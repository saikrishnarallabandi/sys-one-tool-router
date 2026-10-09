import math

import numpy as np
import pytest

from decisionbench import metrics as M


def test_ece_hand_computed():
    # bin (0.8667, 0.9333]: conf 0.9, 0.9 correct 1, 0 -> |0.5 - 0.9| = 0.4, weight 2/4
    # bin (0.5333, 0.6]:    conf 0.6, 0.6 correct 1, 1 -> |1.0 - 0.6| = 0.4, weight 2/4
    conf = np.array([0.9, 0.9, 0.6, 0.6])
    correct = np.array([1, 0, 1, 1])
    assert M.ece(conf, correct) == pytest.approx(0.4)


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([0.8] * 5)
    correct = np.array([1, 1, 1, 1, 0])
    assert M.ece(conf, correct) == pytest.approx(0.0)


def test_ece_bin_edges_and_zero_confidence():
    # 1.0 lands in the top bin, 0.0 in the first bin (not dropped).
    assert M.ece(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)


def test_top_label_ece_uses_argmax():
    P = np.array([[0.7, 0.3], [0.2, 0.8]])
    y = np.array([0, 0])
    # conf 0.7 correct, conf 0.8 wrong, different bins -> (0.3 + 0.8) / 2
    assert M.top_label_ece(P, y) == pytest.approx(0.55)


def test_qwk_hand_computed():
    y = np.array([0, 1, 2])
    assert M.quadratic_weighted_kappa(y, y, 3) == pytest.approx(1.0)
    # pred = [0, 2, 2]: sum W*O = 0.25, sum W*E = 1.25 -> kappa = 1 - 0.25 / 1.25 = 0.8
    pred = np.array([0, 2, 2])
    W = np.array([[(i - j) ** 2 / 4 for j in range(3)] for i in range(3)])
    O = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    E = np.outer(O.sum(1), O.sum(0)) / 3
    expected = 1 - (W * O).sum() / (W * E).sum()
    assert M.quadratic_weighted_kappa(y, pred, 3) == pytest.approx(expected)
    assert expected == pytest.approx(0.8)


def test_aurc_hand_computed():
    conf = np.array([0.9, 0.8, 0.7])
    correct = np.array([1, 0, 1])
    # risks at coverage 1/3, 2/3, 3/3: 0, 1/2, 1/3
    assert M.aurc(conf, correct) == pytest.approx((0 + 0.5 + 1 / 3) / 3)


def test_aurc_ties_are_order_invariant():
    conf = np.array([0.5, 0.5])
    # Whichever tied example is "first", the expected risk is 0.5 at both coverages.
    assert M.aurc(conf, np.array([1, 0])) == pytest.approx(0.5)
    assert M.aurc(conf, np.array([0, 1])) == pytest.approx(0.5)


def test_aurc_perfect_ranking_beats_worst():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    good = M.aurc(conf, np.array([1, 1, 0, 0]))
    bad = M.aurc(conf, np.array([0, 0, 1, 1]))
    assert good < bad


def test_coverage_at_accuracy():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7])
    correct = np.array([1, 1, 1, 0, 1])
    assert M.coverage_at_accuracy(conf, correct, 0.95) == pytest.approx(0.6)
    assert M.coverage_at_accuracy(conf, correct, 0.75) == pytest.approx(1.0)
    assert M.coverage_at_accuracy(np.array([0.9]), np.array([0]), 0.95) == 0.0


def test_coverage_at_accuracy_respects_ties():
    # The top two are tied; a threshold cannot keep only one of them.
    conf = np.array([0.9, 0.9, 0.5])
    correct = np.array([1, 0, 1])
    assert M.coverage_at_accuracy(conf, correct, 0.95) == 0.0


def test_temperature_scaling_softens_overconfident():
    rng = np.random.default_rng(0)
    n, k = 400, 4
    y = rng.integers(0, k, n)
    # 60% accurate but always 0.97 confident in the argmax.
    pred = np.where(rng.random(n) < 0.6, y, (y + 1) % k)
    P = np.full((n, k), 0.01)
    P[np.arange(n), pred] = 0.97
    T = M.fit_temperature(P, y)
    assert T > 1
    assert M.log_loss(M.apply_temperature(P, T), y) < M.log_loss(P, y)
    ts = M.temperature_scaling_eval(P, y, seed=0)
    assert ts["temperature"] > 1
    assert ts["ece_eval_after"] < ts["ece_eval_before"]
    assert ts["n_cal"] == 120 and ts["n_eval"] == 280


def test_temperature_sharpens_underconfident():
    y = np.array([0, 1] * 50)
    P = np.where(np.eye(2)[y] == 1, 0.6, 0.4)  # always right, only 60% confident
    assert M.fit_temperature(P, y) < 1


def test_binary_temperature_is_logit_scaling():
    p = np.array([0.9, 0.2, 0.6])
    T = 2.0
    got = M.apply_temperature(M.binary_matrix(p), T)[:, 1]
    logit = np.log(p / (1 - p))
    assert got == pytest.approx(1 / (1 + np.exp(-logit / T)))


def test_temperature_scaling_too_small_returns_none():
    n = 2 * M.MIN_CAL
    assert M.temperature_scaling_eval(np.array([[0.6, 0.4]] * n), np.array([0, 1] * M.MIN_CAL)) is None
    n = 4 * M.MIN_CAL
    assert M.temperature_scaling_eval(np.array([[0.6, 0.4]] * n), np.array([0, 1] * (2 * M.MIN_CAL))) is not None


def test_log_loss_and_brier():
    P = np.array([[0.5, 0.5], [1.0, 0.0]])
    y = np.array([0, 1])
    assert M.log_loss(P, y) == pytest.approx((math.log(2) - math.log(M.EPS)) / 2)
    assert M.brier(P, y) == pytest.approx((0.5 + 2.0) / 2)


def test_tv_distance():
    P = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    Q = np.array([[0.5, 0.0, 0.5], [1.0, 0.0, 0.0]])
    assert M.tv_distance(P, Q) == pytest.approx([0.5, 0.0])
    o = M.order_sensitivity(P, Q)
    assert o["mean_tv"] == pytest.approx(0.25)
    assert o["flip_rate"] == pytest.approx(0.0)
    assert M.order_sensitivity(np.array([[0.9, 0.1]]), np.array([[0.1, 0.9]]))["flip_rate"] == 1.0


def test_negation_consistency():
    g = M.negation_consistency(np.array([0.9, 0.7, 0.3]), np.array([0.1, 0.6, 0.2]))
    assert g["mean_abs_sum_minus_1"] == pytest.approx((0.0 + 0.3 + 0.5) / 3)
    assert g["contradiction_rate"] == pytest.approx(2 / 3)


def test_choice_metrics_basic():
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4], [0.5, 0.4, 0.1]])
    y = np.array([0, 2, 2, 1])
    m = M.choice_metrics(P, y)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["top2_accuracy"] == pytest.approx(1.0)
    assert 0 <= m["aurc"] <= 1


def test_score_metrics_basic():
    P = np.array([[0.8, 0.2, 0.0], [0.0, 0.1, 0.9], [0.1, 0.8, 0.1]])
    y = np.array([0, 2, 2])
    m = M.score_metrics(P, y)
    assert m["exact_accuracy"] == pytest.approx(2 / 3)
    assert m["within1_accuracy"] == pytest.approx(1.0)
    assert m["mae_expected"] == pytest.approx((0.2 + 0.1 + 1.0) / 3)


def test_noul_metrics_basic_and_degenerate():
    p = np.array([0.9, 0.8, 0.3, 0.1])
    y = np.array([1, 1, 0, 0])
    m = M.noul_metrics(p, y)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["best_f1"] == pytest.approx(1.0)
    one_class = M.noul_metrics(np.array([0.9, 0.8]), np.array([1, 1]))
    assert one_class["auroc"] is None


def test_bootstrap_ci_is_seeded_and_brackets():
    x = np.array([1, 0] * 50)
    a = M.bootstrap_ci(lambda v: float(v.mean()), [x], seed=1)
    b = M.bootstrap_ci(lambda v: float(v.mean()), [x], seed=1)
    assert a == b
    assert a["low"] < 0.5 < a["high"]


def test_latency_stats():
    s = M.latency_stats([0.01, 0.02, 0.03])
    assert s["p50_ms"] == pytest.approx(20)
    assert s["throughput_rps"] == pytest.approx(50)
    assert M.latency_stats([]) is None
