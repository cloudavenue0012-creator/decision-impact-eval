"""Render the two README charts from the eval's own output (data/verdicts.csv).

  docs/traps.png         how often a zero effect gets called an effect, per scenario,
                         for the naive reading, an ungated DiD, and the gated judge
  docs/launch_curve.png  one launch-curve decision: the reading that passes G1 and G2
                         and is stopped only by G3

Run `python eval.py` first so the chart shows exactly the numbers the eval printed.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from judge import (ALPHA, MAX_TIME_CORR, MAX_WEEKEND_GAP, baseline_claim, group_mean,  # noqa: E402
                   judge_one, weekend_gap)
from synth_panel import HISTORY, SCENARIOS, T, make_decision  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#ecebe7", "#fcfcfb"
SERIES = {"naive": "#2a78d6", "did": "#eb6834", "gated": "#1baf7a"}     # fixed categorical order
LABEL = {"naive": "naive before/after", "did": "DiD, no checks", "gated": "gated judge"}
HOLD = {"D1": "held: D1 boundary", "D2": "held: D2 placebo", "G2": "held: G2 weekday", "G3": "held: G3 time"}


def _style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def traps(df: pd.DataFrame, out: Path) -> None:
    null = df[df.true_effect == 0]
    order = [s for s in SCENARIOS if (null.scenario == s).any()]
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    h = 0.26
    for k, m in enumerate(("naive", "did", "gated")):
        rates = [null[null.scenario == s][f"{m}_claim"].mean() for s in order]
        y = np.arange(len(order)) + (k - 1) * h
        ax.barh(y, rates, height=h - 0.04, color=SERIES[m], label=LABEL[m], edgecolor=SURFACE, linewidth=1)
        for yi, r in zip(y, rates):
            ax.text(r + 0.012, yi, f"{r:.0%}", va="center", fontsize=8, color=INK)
    for i, s in enumerate(order):
        sub = null[null.scenario == s]
        checks = sub.hold_check.fillna("")
        top = checks[checks != ""].value_counts()
        if len(top) and (sub.verdict == "hold").mean() > 0.5:
            ax.text(1.12, i, HOLD[top.index[0]], va="center", fontsize=8.5, color=MUTED)
    ax.axvline(0.05, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.05, len(order) - 0.35, " 5%", fontsize=8, color=MUTED, va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([s.replace("_", " ") for s in order], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title("How often a zero effect gets reported as an effect", loc="left",
                 fontsize=13, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, f"decisions whose planted effect is 0 ({len(null)} of {len(df)}), share read as EFFECT · 100% synthetic",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def launch_curve(out: Path, seed: int = 42, idx: int = 0) -> dict:
    d = make_decision("launch_curve", seed, idx)
    est, p, _ = baseline_claim(d, "naive", seed=idx)
    v = judge_one(d, seed=idx)
    days = np.arange(HISTORY, T)
    sales = np.exp(group_mean(d, d.treated))[days]
    sales = sales / sales[-14:].mean() * 100
    on = d.action_days[days]

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = days - HISTORY + 1
    for lo in range(0, len(days), 7):
        if on[lo]:
            ax.axvspan(x[lo] - 0.5, x[lo] + 6.5, color=SERIES["did"], alpha=0.10, lw=0)
    ax.plot(x, sales, color=SERIES["naive"], lw=2)
    ax.scatter(x[on], sales[on], s=14, color=SERIES["did"], zorder=3, label="discount days")
    ax.scatter(x[~on], sales[~on], s=14, color=SERIES["naive"], zorder=3, label="other days")
    ax.set_xlabel("days since the units launched", fontsize=9, color=MUTED)
    ax.set_ylabel("daily sales index (last two weeks = 100)", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title("A launch curve read as a discount effect", loc="left", fontsize=13, fontweight="bold",
                 color=INK, pad=24)
    wg = weekend_gap(days[on], days[~on], d.dow)
    ok = lambda passed: "pass" if passed else "FAIL"
    note = (f"naive reading: discount days {est:+.0%} vs other days\n"
            f"G1 permutation p = {v.p_raw:.4f}        {ok(v.p_raw < ALPHA)}\n"
            f"G2 weekend-share gap = {wg:.0%}        {ok(wg <= MAX_WEEKEND_GAP)}\n"
            f"G3 corr(discount, date) = {v.time_corr:+.2f} {ok(abs(v.time_corr) <= MAX_TIME_CORR)}\n"
            f"verdict: {v.verdict.upper()}   (planted effect: 0%)")
    ax.text(0.98, 0.05, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, multialignment="left",
            family="monospace", color=INK, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID))
    ax.text(0, 1.02, "shaded weeks = discount on · the units were still ramping up, and discounts started in week 4",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return dict(naive=est, p=p, weekend_gap=wg, time_corr=v.time_corr, verdict=v.verdict)


def main():
    path = Path("data") / "verdicts.csv"
    if not path.exists():
        raise SystemExit("data/verdicts.csv not found -- run `python eval.py` first")
    df = pd.read_csv(path)
    docs = Path(__file__).with_name("docs")
    docs.mkdir(exist_ok=True)
    traps(df, docs / "traps.png")
    info = launch_curve(docs / "launch_curve.png")
    print(f"wrote {docs/'traps.png'} and {docs/'launch_curve.png'}")
    print(f"launch example: naive {info['naive']:+.1%} (p={info['p']:.4f}), weekend gap {info['weekend_gap']:.0%}, "
          f"corr {info['time_corr']:+.2f} -> {info['verdict']}")


if __name__ == "__main__":
    main()
