"""Generate synthetic store x day sales panels with *planted* actions.

Every "decision" is one action a review meeting agreed on (a promotion, an
operating change) applied to some stores. The generator records the true effect
it planted -- including zero -- so the eval can check whether a judge recovers it
or, more importantly, whether it claims an effect that was never there.

Log daily sales for store i on day t:

    log y = level_i
          + beta_i * (weekday + seasonality + chain drift + chain-wide events)
          + regional drift + store AR(1) noise + temporary "bad spells"
          + log(1 + true_effect) on action days of treated stores

`beta_i` gives stores different sensitivity to the common calendar, so an
equal-weighted control group is not a perfect mirror (that is what a synthetic
control can improve on).

Seven scenarios. Three are honest designs; four are the traps that real
before/after readings fall into:

  clean_step        random stores, action from day START on, concurrent controls
  clean_onoff       random stores, action on alternating weeks, concurrent controls
  regional_rollout  a whole region treated; the boundary really separates customers
  mean_reversion    action given to the stores with the worst recent two weeks
  launch_curve      newly launched units on a ramp-up; discounts only after week 3
  weekday_skew      chain-wide weekend-only action, nothing untreated to compare to
  cost_boundary     a cost-allocation boundary used as the control line, while the
                    treated region has its own demand shift

No real data: stores, regions, levels, calendars and effects are all generated.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_STORES = 40
N_REGIONS = 4
HISTORY = 28      # days before the analysis window (synthetic-control fit, selection look-back)
PRE = 28          # analysis window: reference period
POST = 28         # analysis window: action period
T = HISTORY + PRE + POST
START = HISTORY + PRE                     # first action day for step designs
WINDOW = (HISTORY, T)                     # analysis window [lo, hi)
WEEKDAY = np.array([-0.08, -0.10, -0.06, -0.03, 0.06, 0.20, 0.14])   # Mon..Sun, log
WEEKEND = (5, 6)

HONEST = ("clean_step", "clean_onoff", "regional_rollout")
TRAPS = ("mean_reversion", "launch_curve", "weekday_skew", "cost_boundary")
SCENARIOS = HONEST + TRAPS


@dataclass
class Decision:
    decision_id: str
    scenario: str
    design: str                 # "step" (before/after) | "onoff" (action on some days)
    log_sales: np.ndarray       # (n_stores, T); NaN before a unit existed
    treated: np.ndarray         # bool (n_stores,)
    control: np.ndarray         # bool (n_stores,) -- stores offered as the comparison (may be empty)
    action_days: np.ndarray     # bool (T,) -- days the action is on for treated stores
    dow: np.ndarray             # (T,) weekday 0=Mon..6=Sun
    region: np.ndarray          # (n_stores,)
    boundary: str               # "none" | "exposure" | "cost" -- what the treated/control line is
    true_effect: float          # planted multiplicative effect (0.05 = +5%); 0 = no effect

    @property
    def window_days(self) -> np.ndarray:
        return np.arange(*WINDOW)


def _ar1(rng, phi: float, sigma: float, shape) -> np.ndarray:
    """AR(1) along the last axis, started from its stationary distribution."""
    e = rng.normal(0.0, sigma, shape)
    x = np.empty(shape)
    x[..., 0] = e[..., 0] / np.sqrt(1 - phi**2)
    for t in range(1, shape[-1]):
        x[..., t] = phi * x[..., t - 1] + e[..., t]
    return x


SPELL_RATE = 0.005   # daily chance a store starts a temporary dip


def _spells(rng, n: int, t: int, rate: float = SPELL_RATE) -> np.ndarray:
    """Temporary store-level dips (road works, a staff gap): random start, 10-24 days, -10..-30%."""
    out = np.zeros((n, t))
    for i in range(n):
        day = 0
        while day < t:
            if rng.random() < rate:
                length = int(rng.integers(10, 25))
                out[i, day:day + length] -= rng.uniform(0.10, 0.30)
                day += length
            day += 1
    return out


def base_panel(rng, spell_rate: float = SPELL_RATE):
    n, t = N_STORES, T
    dow = (int(rng.integers(7)) + np.arange(t)) % 7
    region = rng.permutation(np.arange(n) % N_REGIONS)
    level = rng.normal(np.log(1000.0), 0.35, n)
    beta = rng.uniform(0.6, 1.4, n)

    season = 0.06 * np.sin(2 * np.pi * np.arange(t) / 365 + rng.uniform(0, 2 * np.pi))
    drift = _ar1(rng, 0.97, 0.006, (t,))
    events = np.zeros(t)
    for day in rng.choice(t, size=rng.poisson(3), replace=False):
        events[day] += rng.normal(0, 0.15)
    common = WEEKDAY[dow] + season + drift + events

    regional = _ar1(rng, 0.95, 0.0015, (N_REGIONS, t))[region]
    noise = _ar1(rng, 0.3, 0.07, (n, t))
    log_sales = level[:, None] + beta[:, None] * common[None, :] + regional + noise + _spells(rng, n, t, spell_rate)
    return log_sales, dow, region


def _effect(rng, p_null: float = 0.5) -> float:
    return 0.0 if rng.random() < p_null else float(np.round(rng.uniform(0.03, 0.10), 4))


def _weeks_mask(on_weeks) -> np.ndarray:
    days = np.zeros(T, dtype=bool)
    for w in on_weeks:
        lo = HISTORY + 7 * int(w)
        days[lo:lo + 7] = True
    return days


def make_decision(scenario: str, seed: int, idx: int = 0, spell_rate: float = SPELL_RATE,
                  force_effect: float | None = None) -> Decision:
    rng = np.random.default_rng([seed, idx, SCENARIOS.index(scenario)])
    log_sales, dow, region = base_panel(rng, spell_rate)
    n = N_STORES
    treated = np.zeros(n, dtype=bool)
    design, boundary, effect = "step", "none", 0.0
    step_days = np.arange(T) >= START

    if scenario == "clean_step":
        treated[rng.choice(n, size=rng.integers(6, 13), replace=False)] = True
        action, effect = step_days, _effect(rng)

    elif scenario == "clean_onoff":
        treated[rng.choice(n, size=rng.integers(6, 13), replace=False)] = True
        design, effect = "onoff", _effect(rng)
        action = _weeks_mask(range(int(rng.integers(2)), 8, 2))   # planned alternating weeks

    elif scenario == "regional_rollout":
        treated = region == rng.integers(N_REGIONS)
        action, boundary, effect = step_days, "exposure", _effect(rng)

    elif scenario == "mean_reversion":
        # pick the stores whose last two weeks fell furthest below their usual level
        recent = log_sales[:, START - 14:START].mean(1)
        usual = log_sales[:, :HISTORY].mean(1)
        k = int(rng.integers(5, 9))
        treated[np.argsort(recent - usual)[:k]] = True
        action = step_days

    elif scenario == "launch_curve":
        # new units opened at the start of the window and are still ramping up;
        # discounts were switched on only once the unit had traffic (after week 3)
        k = int(rng.integers(6, 13))
        treated[rng.choice(n, size=k, replace=False)] = True
        age = np.arange(T) - HISTORY
        ramp = np.where(age >= 0, -3.0 * np.exp(-np.maximum(age, 0) / 12.0), np.nan)
        log_sales[treated] = log_sales[treated] + ramp[None, :]
        later = [3, 4, 5, 6, 7]
        later.remove(int(rng.choice(later)))
        design, action = "onoff", _weeks_mask(later)

    elif scenario == "weekday_skew":
        treated[:] = True                                           # chain-wide, nothing left to compare
        design = "onoff"
        action = np.isin(dow, WEEKEND) & (np.arange(T) >= HISTORY)

    elif scenario == "cost_boundary":
        # the treated/control line is a cost-allocation boundary: customers on both sides
        # see the same thing, so the action has no demand effect -- but the treated
        # region happens to get its own demand lift around the same date
        r = int(rng.integers(N_REGIONS))
        treated = region == r
        action, boundary = step_days, "cost"
        shift_from = START + int(rng.integers(-3, 4))
        log_sales[treated, shift_from:] += rng.uniform(0.04, 0.10)

    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    if force_effect is not None:
        effect = force_effect
    control = ~treated if scenario not in ("launch_curve", "weekday_skew") else np.zeros(n, dtype=bool)
    if effect:
        log_sales[np.ix_(treated, action)] += np.log1p(effect)

    return Decision(
        decision_id=f"{scenario}-{idx:03d}", scenario=scenario, design=design,
        log_sales=log_sales, treated=treated, control=control, action_days=action,
        dow=dow, region=region, boundary=boundary, true_effect=effect,
    )


def generate_suite(n_per: int, seed: int, spell_rate: float = SPELL_RATE) -> list[Decision]:
    return [make_decision(s, seed, i, spell_rate) for s in SCENARIOS for i in range(n_per)]


def ground_truth(decisions: list[Decision]) -> pd.DataFrame:
    return pd.DataFrame([dict(
        decision_id=d.decision_id, scenario=d.scenario, design=d.design, boundary=d.boundary,
        n_treated=int(d.treated.sum()), n_control=int(d.control.sum()), true_effect=d.true_effect,
    ) for d in decisions])


def panel_frame(d: Decision) -> pd.DataFrame:
    n, t = d.log_sales.shape
    return pd.DataFrame(dict(
        store=np.repeat([f"S{i+1:02d}" for i in range(n)], t),
        region=np.repeat([f"R{r+1}" for r in d.region], t),
        day=np.tile(np.arange(t), n),
        weekday=np.tile(d.dow, n),
        treated=np.repeat(d.treated, t),
        control=np.repeat(d.control, t),
        action_on=np.repeat(d.treated, t) & np.tile(d.action_days, n),
        sales=np.round(np.exp(d.log_sales.ravel()), 1),
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="decisions per scenario")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()

    a.out.mkdir(exist_ok=True)
    suite = generate_suite(a.n, a.seed)
    ground_truth(suite).to_csv(a.out / "ground_truth.csv", index=False)
    for s in SCENARIOS:                                  # one inspectable panel per scenario
        panel_frame(next(d for d in suite if d.scenario == s)).to_csv(a.out / f"panel_{s}.csv", index=False)
    print(f"wrote {len(suite)} decisions -> {a.out/'ground_truth.csv'} (+ one example panel per scenario)")


if __name__ == "__main__":
    main()
