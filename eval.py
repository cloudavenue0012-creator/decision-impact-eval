"""Evaluation harness -- the differentiator.

Every decision in the suite carries the effect that was planted (often zero), so
each reading can be scored against the truth:

  false-claim rate   true effect 0, verdict "effect"            (the expensive error)
  catch rate         true effect > 0, verdict "effect"          (1 - catch = missed or held)
  recovery error     |estimate - planted effect|, percentage points
  interval coverage  share of gated claims whose interval contains the planted effect

Three readings are compared on the same decisions:

  naive        before/after (or action days vs other days) on the treated stores, raw p < 0.05
  did          difference-in-differences vs the control stores, raw p < 0.05, no checks
  gated        the judge: D1 boundary, D2 placebo, G2 weekday, G3 time, then G1 with Holm
               correction inside each batch of 10 decisions (one review cycle)

Plus two measurements the README leans on: the launch-curve case (G1 and G2 pass,
G3 is the only thing that stops it) and the calibration of shuffling days when the
action was assigned to stores.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from judge import ALPHA, MAX_TIME_CORR, MAX_WEEKEND_GAP, N_PERM, baseline_claim, finalize, judge_one, synth_estimate
from synth_panel import HONEST, SCENARIOS, SPELL_RATE, generate_suite, make_decision

BATCH = 10


def run(n_per: int = 100, seed: int = 42, n_perm: int = N_PERM, spell_rate: float = SPELL_RATE) -> pd.DataFrame:
    suite = generate_suite(n_per, seed, spell_rate)
    rows, pending = [], []
    for i, d in enumerate(suite):
        s = 1000 * i + seed
        nv_est, nv_p, nv_claim = baseline_claim(d, "naive", n_perm, s)
        dd_est, dd_p, dd_claim = baseline_claim(d, "did", n_perm, s)
        if d.control.any():
            _, ds_p, ds_claim = baseline_claim(d, "did_dayshuffle", n_perm, s)
        else:
            ds_p, ds_claim = float("nan"), False
        rows.append(dict(
            decision_id=d.decision_id, scenario=d.scenario, design=d.design, boundary=d.boundary,
            has_control=bool(d.control.any()), true_effect=d.true_effect,
            naive_est=nv_est, naive_p=nv_p, naive_claim=nv_claim,
            did_est=dd_est, did_p=dd_p, did_claim=dd_claim,
            dayshuffle_p=ds_p, dayshuffle_claim=ds_claim,
            synth_est=synth_estimate(d),
        ))
        pending.append(judge_one(d, n_perm, s + 1))

    # review cycles: decisions arrive in a shuffled order, ten per cycle, Holm inside each
    order = np.random.default_rng(seed).permutation(len(suite))
    batch_of = np.empty(len(suite), dtype=int)
    raw = copy.deepcopy(pending)
    for b, lo in enumerate(range(0, len(suite), BATCH)):
        idx = order[lo:lo + BATCH]
        batch_of[idx] = b
        finalize([pending[j] for j in idx], "holm")
        finalize([raw[j] for j in idx], "none")

    df = pd.DataFrame(rows)
    df["batch"] = batch_of
    for col in ("verdict", "reason", "estimate", "ci_low", "ci_high", "p_raw", "p_adj",
                "weekend_gap", "time_corr", "placebo_p"):
        df[col] = [getattr(v, col) for v in pending]
    df["verdict_uncorrected"] = [v.verdict for v in raw]
    df["gated_claim"] = df["verdict"] == "effect"
    df["uncorrected_claim"] = df["verdict_uncorrected"] == "effect"
    df["hold_check"] = np.where(df["verdict"] == "hold", df["reason"].str[:2], "")
    return df


def null_cycles(n_cycles: int = 50, seed: int = 42, n_perm: int = N_PERM, spell_rate: float = SPELL_RATE) -> dict:
    """Review cycles of 10 honest decisions that all truly did nothing: how many 'effects' appear?"""
    raw_counts, holm_counts = [], []
    for c in range(n_cycles):
        cycle = [make_decision(HONEST[j % len(HONEST)], seed + 1, 10 * c + j, spell_rate, force_effect=0.0)
                 for j in range(BATCH)]
        pending = [judge_one(d, n_perm, 7919 * c + j) for j, d in enumerate(cycle)]
        raw = finalize(copy.deepcopy(pending), "none")
        holm = finalize(pending, "holm")
        raw_counts.append(sum(v.verdict == "effect" for v in raw))
        holm_counts.append(sum(v.verdict == "effect" for v in holm))
    raw_counts, holm_counts = np.array(raw_counts), np.array(holm_counts)
    return dict(cycles=n_cycles, raw_mean=float(raw_counts.mean()), holm_mean=float(holm_counts.mean()),
                raw_any=float((raw_counts > 0).mean()), holm_any=float((holm_counts > 0).mean()))


def summarize(df: pd.DataFrame) -> dict:
    null, pos = df[df.true_effect == 0], df[df.true_effect > 0]
    honest_null = null[null.scenario.isin(HONEST)]
    claims = pos[pos.gated_claim]
    launch = df[df.scenario == "launch_curve"]
    ctrl_null = honest_null[honest_null.has_control]
    out = dict(
        n=len(df), n_null=len(null), n_pos=len(pos),
        false_claim={m: float(null[f"{m}_claim"].mean()) for m in ("naive", "did", "gated")},
        honest_false_claim={m: float(honest_null[f"{m}_claim"].mean()) for m in ("naive", "did", "gated")},
        catch={m: float(pos[f"{m}_claim"].mean()) for m in ("naive", "did", "gated")},
        held_real=float((pos.verdict == "hold").mean()),
        catch_by_size={label: float(pos[(pos.true_effect > lo) & (pos.true_effect <= hi)].gated_claim.mean())
                       for label, lo, hi in (("+3-5%", 0.0, 0.05), ("+5-7.5%", 0.05, 0.075), ("+7.5-10%", 0.075, 0.10))},
        mae={"naive": float((pos.naive_est - pos.true_effect).abs().mean() * 100),
             "did": float((pos.did_est - pos.true_effect).abs().mean() * 100),
             "synth": float((pos.synth_est - pos.true_effect).abs().mean() * 100),
             "gated_claims": float((claims.estimate - claims.true_effect).abs().mean() * 100)},
        coverage=float(((claims.ci_low <= claims.true_effect) & (claims.true_effect <= claims.ci_high)).mean()),
        coverage_all=float(((pos.ci_low <= pos.true_effect) & (pos.true_effect <= pos.ci_high)).mean()),
        launch=dict(
            n=len(launch),
            g1_pass=float((launch.p_raw < ALPHA).mean()),
            g2_pass=float((launch.weekend_gap <= MAX_WEEKEND_GAP).mean()),
            g3_fail=float((launch.time_corr.abs() > MAX_TIME_CORR).mean()),
            only_g3=float(((launch.p_raw < ALPHA) & (launch.weekend_gap <= MAX_WEEKEND_GAP)
                           & (launch.time_corr.abs() > MAX_TIME_CORR)).mean()),
            naive_median=float(launch.naive_est.median()), naive_p_median=float(launch.naive_p.median()),
            corr_min=float(launch.time_corr.abs().min()), corr_median=float(launch.time_corr.abs().median()),
        ),
        multiple=dict(
            reached_g1=int((honest_null.verdict != "hold").sum()),
            raw=int(honest_null.uncorrected_claim.sum()), holm=int(honest_null.gated_claim.sum()),
            batches=int(df.batch.nunique()),
            batches_any_false_raw=int(null[null.uncorrected_claim].batch.nunique()),
            batches_any_false_holm=int(null[null.gated_claim].batch.nunique()),
        ),
        dayshuffle=dict(n=len(ctrl_null), store=float(ctrl_null.did_claim.mean()),
                        day=float(ctrl_null.dayshuffle_claim.mean())),
    )
    return out


def scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SCENARIOS:
        for label, sub in (("effect = 0", df[(df.scenario == s) & (df.true_effect == 0)]),
                           ("effect > 0", df[(df.scenario == s) & (df.true_effect > 0)])):
            if not len(sub):
                continue
            holds = sub.hold_check[sub.hold_check != ""]
            rows.append(dict(
                scenario=s, truth=label, n=len(sub),
                naive=sub.naive_claim.mean(), did=sub.did_claim.mean(), gated=sub.gated_claim.mean(),
                held=(sub.verdict == "hold").mean(),
                blocked_by=(holds.value_counts().index[0] if len(holds) else "-"),
            ))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, s: dict) -> None:
    pct = lambda x: f"{x:6.1%}"
    print("=" * 74)
    print(" Decision-Impact Eval - did the action work, and may we say so?")
    print("=" * 74)
    print(f" decisions: {s['n']}  (true effect 0: {s['n_null']} | true effect > 0: {s['n_pos']})")
    print(" 'claimed' = share of decisions read as EFFECT")
    print()
    print(f" {'scenario':17s} {'truth':10s} {'n':>4s} {'naive':>7s} {'DiD':>7s} {'gated':>7s} {'held':>7s}  blocked by")
    for r in scenario_table(df).itertuples():
        print(f" {r.scenario:17s} {r.truth:10s} {r.n:4d} {pct(r.naive)} {pct(r.did)} {pct(r.gated)} {pct(r.held)}  {r.blocked_by}")
    print("-" * 74)
    fc, hfc, c = s["false_claim"], s["honest_false_claim"], s["catch"]
    print(f" false-claim rate, all effect-0 decisions : naive {pct(fc['naive'])}  DiD {pct(fc['did'])}  gated {pct(fc['gated'])}")
    print(f" false-claim rate, honest designs only    : naive {pct(hfc['naive'])}  DiD {pct(hfc['did'])}  gated {pct(hfc['gated'])}")
    print(f" catch rate, real effects                 : naive {pct(c['naive'])}  DiD {pct(c['did'])}  gated {pct(c['gated'])}"
          f"   (gated held {s['held_real']:.1%})")
    print("   gated catch rate by planted effect      : "
          + "  ".join(f"{k} {v:.1%}" for k, v in s["catch_by_size"].items()))
    m = s["mae"]
    print(f" recovery error, real effects (pp)        : naive {m['naive']:5.2f}  DiD {m['did']:5.2f}  synth {m['synth']:5.2f}"
          f"  gated claims {m['gated_claims']:5.2f}")
    print(f" 95% interval contains the planted effect : gated claims {s['coverage']:.1%}"
          f" | all real-effect decisions {s['coverage_all']:.1%}")
    print("-" * 74)
    L = s["launch"]
    print(f" launch curve ({L['n']} decisions, true effect 0):")
    print(f"   naive reading: median {L['naive_median']:+.0%}, median p {L['naive_p_median']:.4f}")
    print(f"   G1 p<0.05 {pct(L['g1_pass'])} | G2 pass {pct(L['g2_pass'])} | G3 fail {pct(L['g3_fail'])}"
          f"   |corr(action,date)| min {L['corr_min']:.2f} median {L['corr_median']:.2f}")
    print(f"   G1 and G2 pass, only G3 stops it: {pct(L['only_g3'])}")
    M = s["multiple"]
    print(f" multiple comparisons, this suite (honest effect-0 decisions reaching G1: {M['reached_g1']}):")
    print(f"   false claims raw p<0.05 {M['raw']} -> Holm {M['holm']}"
          f"   | review cycles with >=1 false claim: {M['batches_any_false_raw']}/{M['batches']} -> {M['batches_any_false_holm']}/{M['batches']}")
    if "cycles" in s:
        C = s["cycles"]
        print(f" multiple comparisons, {C['cycles']} cycles of 10 decisions that all did nothing:")
        print(f"   'effects' per cycle: raw {C['raw_mean']:.2f} -> Holm {C['holm_mean']:.2f}"
              f"   | cycles with >=1: raw {C['raw_any']:.0%} -> Holm {C['holm_any']:.0%}")
    D = s["dayshuffle"]
    print(f" shuffle unit (effect-0 decisions with a control, n={D['n']}):"
          f" store labels {pct(D['store'])}  vs  day labels {pct(D['day'])} false claims")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="decisions per scenario")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--spell-rate", type=float, default=SPELL_RATE, help="daily chance a store starts a temporary dip")
    ap.add_argument("--cycles", type=int, default=50, help="all-null review cycles for the multiple-comparison check")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--max-false-claim", type=float, default=None, help="fail if gated false-claim rate exceeds this")
    ap.add_argument("--min-catch", type=float, default=None, help="fail if gated catch rate on real effects is below this")
    ap.add_argument("--min-launch-g3", type=float, default=None,
                    help="fail if the share of launch-curve decisions stopped only by G3 is below this")
    a = ap.parse_args()

    df = run(a.n, a.seed, a.n_perm, a.spell_rate)
    s = summarize(df)
    if a.cycles:
        s["cycles"] = null_cycles(a.cycles, a.seed, a.n_perm, a.spell_rate)
    report(df, s)
    a.out.mkdir(exist_ok=True)
    df.to_csv(a.out / "verdicts.csv", index=False)

    failures = []
    if a.max_false_claim is not None:
        for k, v in (("all", s["false_claim"]["gated"]), ("honest", s["honest_false_claim"]["gated"])):
            if v > a.max_false_claim:
                failures.append(f"gated false-claim ({k}) {v:.3f} > {a.max_false_claim}")
    if a.min_catch is not None and s["catch"]["gated"] < a.min_catch:
        failures.append(f"gated catch rate {s['catch']['gated']:.3f} < {a.min_catch}")
    if a.min_launch_g3 is not None and s["launch"]["only_g3"] < a.min_launch_g3:
        failures.append(f"launch curve stopped only by G3 {s['launch']['only_g3']:.3f} < {a.min_launch_g3}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
