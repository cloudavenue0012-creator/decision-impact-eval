"""Tests: the statistics (permutation test, Holm, the two calendar gates), the
generator's ground truth, and that each trap is stopped by the check built for it
while a real effect still gets through."""
import numpy as np
import pytest

from eval import run, summarize
from judge import (contrast, finalize, holm, judge_one, permutation_test, time_corr,
                   weekend_gap)
from synth_panel import HISTORY, START, T, make_decision

FAST = 2_000   # permutations in tests; the eval and CI use 10,000


def test_generator_is_deterministic():
    a = make_decision("clean_step", seed=3, idx=1)
    b = make_decision("clean_step", seed=3, idx=1)
    np.testing.assert_array_equal(a.log_sales, b.log_sales)
    np.testing.assert_array_equal(a.treated, b.treated)
    assert a.true_effect == b.true_effect


def test_planted_effect_is_recorded_and_applied():
    base = make_decision("clean_step", seed=5, idx=0, force_effect=0.0)
    lifted = make_decision("clean_step", seed=5, idx=0, force_effect=0.08)
    assert lifted.true_effect == 0.08 and base.true_effect == 0.0
    diff = lifted.log_sales - base.log_sales
    assert np.allclose(diff[np.ix_(lifted.treated, np.arange(START, T))], np.log1p(0.08))
    assert np.allclose(diff[:, :START], 0) and np.allclose(diff[~lifted.treated], 0)


def test_permutation_test_extremes():
    obs, p, null = permutation_test(np.zeros(20), np.zeros(20), n_perm=500)
    assert obs == 0 and p == pytest.approx(1.0)
    obs, p, _ = permutation_test(np.arange(20) + 100.0, np.arange(20.0), n_perm=500)
    assert obs == pytest.approx(100.0) and p == pytest.approx(1 / 501)   # nothing as extreme


def test_holm_matches_hand_computation():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]).max() <= 1.0


def test_calendar_gates():
    dow = np.arange(T) % 7
    weekend, weekday = np.where(dow >= 5)[0], np.where(dow < 5)[0]
    assert weekend_gap(weekend, weekday, dow) == pytest.approx(1.0)
    same = np.arange(START, T)
    assert time_corr(same, same) == pytest.approx(0.0)                      # concurrent control
    assert time_corr(np.arange(START, T), np.arange(HISTORY, START)) > 0.85  # before/after


def test_control_means_store_shuffle_and_same_dates():
    c = contrast(make_decision("clean_step", seed=1), "did")
    assert c.unit == "stores"
    np.testing.assert_array_equal(c.days_action, c.days_reference)
    assert contrast(make_decision("weekday_skew", seed=1), "did").unit == "days"


@pytest.mark.parametrize("scenario,check", [
    ("cost_boundary", "D1"), ("mean_reversion", "D2"), ("weekday_skew", "G2"), ("launch_curve", "G3"),
])
def test_each_trap_is_held_by_its_check(scenario, check):
    verdicts = [judge_one(make_decision(scenario, seed=11, idx=i), n_perm=FAST, seed=i) for i in range(8)]
    assert all(v.verdict == "hold" and v.reason.startswith(check) for v in verdicts)


def test_launch_curve_passes_g1_and_g2_and_only_g3_stops_it():
    for i in range(5):
        v = judge_one(make_decision("launch_curve", seed=2, idx=i), n_perm=FAST, seed=i)
        assert v.p_raw < 0.05 and v.weekend_gap <= 0.20 and abs(v.time_corr) > 0.40
        assert v.estimate > 0.5                      # a large, "significant", entirely spurious lift


def test_real_effect_is_claimed_with_an_interval_that_covers_it():
    d = make_decision("clean_onoff", seed=9, idx=0, force_effect=0.10)
    v = finalize([judge_one(d, n_perm=FAST)])[0]
    assert v.verdict == "effect"
    assert v.ci_low <= 0.10 <= v.ci_high


def test_small_end_to_end_eval():
    s = summarize(run(n_per=6, seed=42, n_perm=FAST))
    assert s["false_claim"]["gated"] <= 0.05
    assert s["false_claim"]["naive"] > s["false_claim"]["gated"]
    assert s["launch"]["only_g3"] == 1.0
