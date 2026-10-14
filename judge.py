"""The judge: did the action work, store by store, and is the claim allowed?

Three estimators, weakest to strongest:

  naive   before/after (or action days vs other days) on the treated stores alone
  did     difference-in-differences against the concurrent control stores
  synth   the same, against a synthetic control: non-negative control-store weights
          (summing to 1) fitted on every day before the action first switches on

The verdict is not "p < 0.05". A causal claim has to survive, in order:

  D1 boundary    the treated/control line must separate what customers experience.
                 A cost-allocation line does not -- the other side is not a control.
  D2 placebo     (before/after designs with a control) pretend the action started
                 14 days early and score it against every earlier day. If that fake
                 action "works", the treated stores were already moving (regression
                 to the mean, a differential trend) and the real reading is not clean.
  G2 weekday     weekend share of action-arm days vs reference-arm days, gap <= 20pp
  G3 time        |corr(action label, date)| <= 0.40 over the compared days
  G1 permutation two-sided permutation test, 10,000 shuffles, p < 0.05 after Holm
                 correction across the batch of decisions reviewed together

Output: EFFECT (size + interval) / NO_EFFECT (with the interval) / HOLD (which check).

Two design notes the eval backs with numbers:

* G1 shuffles the unit the action was assigned by. With a concurrent control the
  action was assigned to *stores*, so store labels are shuffled (each store scored by
  its own after-minus-before change). Without one -- an action switched on for some
  days at the same stores -- day labels are shuffled. Shuffling days when stores were
  assigned treats a store's temporary slump as if it were the action and inflates
  false claims several-fold (see `eval.py`, "day-shuffle calibration").
* G2/G3 are measured on the dates each arm is observed on. With a concurrent control
  the reference arm is the control stores *on the same dates*, so G2 = G3 = 0 by
  construction -- that is what a control buys. A before/after comparison without one
  has corr(label, date) ~ 0.87: it compares two periods of time, and the gate says so.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from synth_panel import HISTORY, START, WEEKEND, Decision

ALPHA = 0.05
N_PERM = 10_000
MAX_WEEKEND_GAP = 0.20
MAX_TIME_CORR = 0.40
PLACEBO_SPLIT = 14


@dataclass
class Contrast:
    action: np.ndarray              # outcome (log) per unit in the action arm
    reference: np.ndarray           # outcome (log) per unit in the reference arm
    days_action: np.ndarray         # dates the action arm is observed on (for G2/G3)
    days_reference: np.ndarray      # dates the reference arm is observed on
    unit: str                       # "stores" | "days" -- what G1 shuffles


@dataclass
class Verdict:
    decision_id: str
    scenario: str
    verdict: str                    # "effect" | "no_effect" | "hold"
    reason: str
    estimate: float                 # multiplicative effect, 0.05 = +5%
    ci_low: float
    ci_high: float
    p_raw: float
    p_adj: float
    weekend_gap: float
    time_corr: float
    placebo_p: float
    true_effect: float


# ---------------------------------------------------------------- statistics

def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERM, seed: int = 0):
    """Two-sided permutation test on the difference of means. Returns (obs, p, null)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    x = np.concatenate([a, b])
    na, total = len(a), x.sum()
    obs = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.tile(x, (n_perm, 1)), axis=1)
    sa = perm[:, :na].sum(1)
    null = sa / na - (total - sa) / (len(x) - na)
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n_perm + 1)
    return float(obs), float(p), null


def interval(obs: float, null: np.ndarray, alpha: float = ALPHA):
    """Shift-model interval by inverting the permutation null (log scale -> multiplicative)."""
    lo, hi = obs - np.quantile(null, 1 - alpha / 2), obs - np.quantile(null, alpha / 2)
    return float(np.expm1(lo)), float(np.expm1(hi))


def weekend_gap(days_a, days_b, dow) -> float:
    share = lambda d: float(np.isin(dow[d], WEEKEND).mean()) if len(d) else 0.0
    return abs(share(days_a) - share(days_b))


def time_corr(days_a, days_b) -> float:
    label = np.r_[np.ones(len(days_a)), np.zeros(len(days_b))]
    t = np.r_[days_a, days_b].astype(float)
    if t.std() == 0 or label.std() == 0:
        return 0.0
    return float(np.corrcoef(label, t)[0, 1])


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


# ---------------------------------------------------------------- contrasts

def group_mean(d: Decision, mask) -> np.ndarray:
    """Daily mean log sales over a store group; NaN on days none of them existed yet."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(d.log_sales[mask], axis=0)


def _arm_days(d: Decision, days=None, on=None):
    days = d.window_days if days is None else days
    on = d.action_days[days] if on is None else on
    return days, on


def contrast(d: Decision, method: str = "did", days=None, on=None) -> Contrast:
    """naive -> within the treated stores, day units. did -> store units vs the control."""
    days, on = _arm_days(d, days, on)
    if method == "naive" or not d.control.any():
        y = group_mean(d, d.treated)[days]
        return Contrast(y[on], y[~on], days[on], days[~on], "days")
    # each store's own change, action days minus reference days (same days for everyone)
    score = np.nanmean(d.log_sales[:, days[on]], 1) - np.nanmean(d.log_sales[:, days[~on]], 1)
    return Contrast(score[d.treated], score[d.control], days[on], days[on], "stores")


def contrast_dayshuffle(d: Decision) -> Contrast:
    """DiD on the daily gap treated - control, with *day* labels as the shuffled unit.
    Kept only to measure why it is the wrong unit when stores were assigned."""
    days, on = _arm_days(d)
    gap = (group_mean(d, d.treated) - group_mean(d, d.control))[days]
    return Contrast(gap[on], gap[~on], days[on], days[~on], "days")


def synth_weights(d: Decision) -> np.ndarray:
    """Demeaned synthetic control fitted on every day before the action first switches on:
    w >= 0, sum(w) = 1 (enforced with a heavily weighted extra row)."""
    fit = int(np.argmax(d.action_days))
    y = group_mean(d, d.treated)[:fit]
    X = d.log_sales[d.control][:, :fit]
    y, X = y - y.mean(), X - X.mean(1, keepdims=True)
    lam = 10.0
    w, _ = nnls(np.vstack([X.T, lam * np.ones(X.shape[0])]), np.r_[y, lam])
    return w / w.sum()


def synth_estimate(d: Decision) -> float:
    """Point estimate against the synthetic control (multiplicative). NaN if not applicable."""
    yt = group_mean(d, d.treated)
    if not d.control.any() or not np.isfinite(yt[:HISTORY]).all():
        return float("nan")
    days, on = _arm_days(d)
    gap = (yt - synth_weights(d) @ d.log_sales[d.control])[days]
    return float(np.expm1(gap[on].mean() - gap[~on].mean()))


def estimate(d: Decision, method: str = "did", n_perm: int = N_PERM, seed: int = 0):
    c = contrast_dayshuffle(d) if method == "did_dayshuffle" else contrast(d, method)
    obs, p, null = permutation_test(c.action, c.reference, n_perm, seed)
    return c, obs, p, null


def placebo_p(d: Decision, n_perm: int = N_PERM, seed: int = 0) -> float:
    """Fake action PLACEBO_SPLIT days before the real one, scored against every earlier day."""
    days = np.arange(0, START)
    c = contrast(d, "did", days, days >= START - PLACEBO_SPLIT)
    return permutation_test(c.action, c.reference, n_perm, seed)[1]


# ---------------------------------------------------------------- verdicts

def baseline_claim(d: Decision, method: str, n_perm: int = N_PERM, seed: int = 0):
    """A reading *without* the checks: the estimate, and 'effect' whenever raw p < 0.05."""
    _, obs, p, _ = estimate(d, method, n_perm, seed)
    return float(np.expm1(obs)), p, p < ALPHA


def judge_one(d: Decision, n_perm: int = N_PERM, seed: int = 0) -> Verdict:
    """Gated verdict before the batch-level multiple-comparison step (G1)."""
    c, obs, p, null = estimate(d, "did", n_perm, seed)
    lo, hi = interval(obs, null)
    wg = weekend_gap(c.days_action, c.days_reference, d.dow)
    tc = time_corr(c.days_action, c.days_reference)
    pp = placebo_p(d, n_perm, seed + 1) if (d.design == "step" and d.control.any()) else float("nan")

    reason = ""
    if d.boundary == "cost":
        reason = "D1 boundary: the control line is a cost boundary, not a customer one"
    elif pp == pp and pp < ALPHA:
        reason = f"D2 placebo: treated stores moved before the action (p={pp:.4f})"
    elif wg > MAX_WEEKEND_GAP:
        reason = f"G2 weekday: weekend share differs by {wg:.0%}"
    elif abs(tc) > MAX_TIME_CORR:
        reason = f"G3 time: corr(action, date) = {tc:+.2f}"

    return Verdict(
        decision_id=d.decision_id, scenario=d.scenario,
        verdict="hold" if reason else "pending", reason=reason,
        estimate=float(np.expm1(obs)), ci_low=lo, ci_high=hi, p_raw=p, p_adj=p,
        weekend_gap=wg, time_corr=tc, placebo_p=pp, true_effect=d.true_effect,
    )


def finalize(verdicts: list[Verdict], correction: str = "holm") -> list[Verdict]:
    """G1 across one batch (one review cycle). Only decisions that cleared D1/D2/G2/G3 are tested."""
    live = [v for v in verdicts if v.verdict != "hold"]
    raw = np.array([v.p_raw for v in live])
    adj = holm(raw) if correction == "holm" else raw
    for v, pa in zip(live, adj):
        v.p_adj = float(pa)
        if pa < ALPHA:
            v.verdict, v.reason = "effect", f"G1 passed (p_adj={pa:.4f})"
        else:
            v.verdict, v.reason = "no_effect", f"G1 not significant (p_adj={pa:.4f})"
    return verdicts


def judge_batch(decisions: list[Decision], correction: str = "holm",
                n_perm: int = N_PERM, seed: int = 0) -> list[Verdict]:
    return finalize([judge_one(d, n_perm, seed + 7 * i) for i, d in enumerate(decisions)], correction)
