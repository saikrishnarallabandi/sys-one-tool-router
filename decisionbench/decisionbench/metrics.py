"""Metrics over normalized predictions. Pure numpy / scipy / sklearn.

Conventions: `P` is an (n, K) array of probabilities aligned with label/level order, `y` an (n,)
int array of gold indices. For noul, `p` is an (n,) array of P(true) and `y` is 0/1. Metrics
that are undefined on the given sample (e.g. AUROC with one class present) are returned as None.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, f1_score, precision_recall_curve, roc_auc_score,
)

EPS = 1e-6
N_BINS = 15
CAL_FRACTION = 0.3
# Below this many calibration examples a fitted temperature is noise (separable folds drive T to a bound).
MIN_CAL = 10


def _finite_or_none(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def binary_matrix(p: np.ndarray) -> np.ndarray:
    """(n,) P(true) -> (n, 2) [P(false), P(true)]."""
    p = np.asarray(p, dtype=float)
    return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------- proper scores
def log_loss(P: np.ndarray, y: np.ndarray, eps: float = EPS) -> float:
    """Mean negative log-likelihood of the gold class, probabilities clipped to [eps, 1]."""
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0))))


def brier(P: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score: mean over examples of sum_k (p_k - 1[k = y])^2 (range 0..2)."""
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


# ---------------------------------------------------------------- calibration
def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error with equal-width bins over (0, 1]; conf == 0 falls in bin 0."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if conf.size == 0:
        raise ValueError("ece of an empty sample")
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / conf.size)


def top_label_ece(P: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    return ece(P.max(axis=1), P.argmax(axis=1) == y, n_bins)


def apply_temperature(P: np.ndarray, T: float, eps: float = EPS) -> np.ndarray:
    """softmax(log p / T). For two columns [1-p, p] this is exactly sigmoid(logit(p) / T)."""
    z = np.log(np.clip(np.asarray(P, dtype=float), eps, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(P: np.ndarray, y: np.ndarray, eps: float = EPS,
                    bounds: tuple[float, float] = (1e-2, 1e2)) -> float:
    """Temperature minimizing NLL on (P, y), searched over log T. T > 1 softens, T < 1 sharpens."""
    y = np.asarray(y, dtype=int)
    res = minimize_scalar(lambda lt: log_loss(apply_temperature(P, math.exp(lt), eps), y, eps),
                          bounds=(math.log(bounds[0]), math.log(bounds[1])), method="bounded",
                          options={"xatol": 1e-5})
    return float(math.exp(res.x))


def temperature_scaling_eval(P: np.ndarray, y: np.ndarray, seed: int = 0, cal_fraction: float = CAL_FRACTION,
                             n_bins: int = N_BINS) -> dict | None:
    """Fit T on a seeded `cal_fraction` fold, report ECE/NLL before and after on the remaining fold."""
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    n = len(y)
    n_cal = int(round(n * cal_fraction))
    if n_cal < MIN_CAL or n - n_cal < MIN_CAL:
        return None
    perm = np.random.default_rng(seed).permutation(n)
    cal, ev = perm[:n_cal], perm[n_cal:]
    T = fit_temperature(P[cal], y[cal])
    P_ts = apply_temperature(P[ev], T)
    return {
        "temperature": T,
        "ece_eval_before": top_label_ece(P[ev], y[ev], n_bins),
        "ece_eval_after": top_label_ece(P_ts, y[ev], n_bins),
        "nll_eval_before": log_loss(P[ev], y[ev]),
        "nll_eval_after": log_loss(P_ts, y[ev]),
        "n_cal": int(n_cal),
        "n_eval": int(n - n_cal),
    }


# ---------------------------------------------------------------- selective prediction
def _tie_groups(conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order by descending confidence; return (order, group end positions exclusive)."""
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    ends = np.flatnonzero(np.diff(c) != 0) + 1
    return order, np.append(ends, len(c))


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Selective risk at coverage k/n for k = 1..n.

    Examples with equal confidence cannot be separated by a threshold, so within a tie group the
    errors are spread evenly (the expectation over orderings of the tied examples).
    """
    conf, err = np.asarray(conf, dtype=float), 1.0 - np.asarray(correct, dtype=float)
    order, ends = _tie_groups(conf)
    err = err[order]
    exp_err = np.empty(len(err))
    start, before = 0, 0.0
    for end in ends:
        rate = err[start:end].mean()
        exp_err[start:end] = before + rate * np.arange(1, end - start + 1)
        before += err[start:end].sum()
        start = end
    return exp_err / np.arange(1, len(err) + 1)


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve: mean selective risk over coverages 1/n..n/n (lower is better)."""
    return float(np.mean(risk_coverage_curve(conf, correct)))


def coverage_at_accuracy(conf: np.ndarray, correct: np.ndarray, target: float = 0.95) -> float:
    """Largest fraction of examples that can be kept, thresholding on confidence, with accuracy >= target."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    order, ends = _tie_groups(conf)
    cum = np.cumsum(correct[order])
    ok = [e for e in ends if cum[e - 1] / e >= target - 1e-12]
    return float(max(ok) / len(conf)) if ok else 0.0


# ---------------------------------------------------------------- per-type suites
def top_k_accuracy(P: np.ndarray, y: np.ndarray, k: int) -> float:
    topk = np.argsort(-np.asarray(P), axis=1, kind="stable")[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def choice_metrics(P: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    pred, conf = P.argmax(axis=1), P.max(axis=1)
    correct = pred == y
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "top2_accuracy": top_k_accuracy(P, y, 2) if P.shape[1] > 2 else None,
        "log_loss": log_loss(P, y),
        "brier": brier(P, y),
        "ece": top_label_ece(P, y),
        "temperature_scaling": temperature_scaling_eval(P, y, seed),
        "aurc": aurc(conf, correct),
        "coverage_at_95_acc": coverage_at_accuracy(conf, correct, 0.95),
    }


def quadratic_weighted_kappa(y: np.ndarray, pred: np.ndarray, n_levels: int) -> float | None:
    return _finite_or_none(cohen_kappa_score(y, pred, labels=list(range(n_levels)), weights="quadratic"))


def score_metrics(P: np.ndarray, y: np.ndarray) -> dict:
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    k = P.shape[1]
    pred = P.argmax(axis=1)
    expected = P @ np.arange(k)
    rho = spearmanr(expected, y).statistic if len(np.unique(y)) > 1 and len(np.unique(expected)) > 1 else None
    return {
        "exact_accuracy": float(np.mean(pred == y)),
        "within1_accuracy": float(np.mean(np.abs(pred - y) <= 1)),
        "mae_expected": float(np.mean(np.abs(expected - y))),
        "spearman": _finite_or_none(rho),
        "qwk": quadratic_weighted_kappa(y, pred, k),
        "ece": top_label_ece(P, y),
        "log_loss": log_loss(P, y),
    }


def noul_metrics(p: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
    both = len(np.unique(y)) == 2
    P2 = binary_matrix(p)
    hard = (p >= 0.5).astype(int)
    best_f1 = best_thr = None
    if y.sum() > 0:
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)[:-1]
        i = int(np.argmax(f1))
        best_f1, best_thr = float(f1[i]), float(thr[i])
    return {
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "auprc": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "accuracy": float(np.mean(hard == y)),
        "f1": float(f1_score(y, hard, zero_division=0)),
        "best_f1": best_f1,
        "best_threshold": best_thr,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": log_loss(P2, y),
        "ece": top_label_ece(P2, y),
        "temperature_scaling": temperature_scaling_eval(P2, y, seed),
    }


# ---------------------------------------------------------------- multi-question (typed-decisions)
def kl_from_gold(gold: np.ndarray, pred: np.ndarray, eps: float = EPS) -> float:
    """KL(gold || model) in nats; model probabilities clipped at eps."""
    g, p = np.asarray(gold, dtype=float), np.clip(np.asarray(pred, dtype=float), eps, 1.0)
    m = g > 0
    return float(np.sum(g[m] * np.log(g[m] / p[m])))


def decision_distribution(qtype: str, probs: Sequence[float]) -> np.ndarray:
    """A decision's full distribution: choice/score as given, noul [P(false), P(true)]."""
    return np.array([1.0 - probs[0], probs[0]]) if qtype == "noul" else np.asarray(probs, dtype=float)


def decision_hit(qtype: str, probs: Sequence[float], gold: int) -> bool:
    """Discrete answer = argmax (noul: p >= 0.5) compared with the gold index."""
    if qtype == "noul":
        return int(probs[0] >= 0.5) == gold
    return int(np.argmax(probs)) == gold


def multi_question_metrics(items: list[dict]) -> dict:
    """Metrics over decisions pooled across cases. Each item: case, group, question (a key unique per
    question schema, e.g. 'workflow/qid'), type, probs (as stored), gold (index), gold_probs (optional)."""
    if not items:
        raise ValueError("no decisions")
    hits = np.array([decision_hit(d["type"], d["probs"], d["gold"]) for d in items])
    out: dict = {"decision_accuracy": float(hits.mean()), "n_decisions": len(items),
                 "n_cases": len({d["case"] for d in items})}
    by_group: dict[str, list[bool]] = {}
    for d, h in zip(items, hits):
        by_group.setdefault(d["group"], []).append(bool(h))
    out["accuracy_by_group"] = {g: float(np.mean(v)) for g, v in sorted(by_group.items())}

    ch = [(d, h) for d, h in zip(items, hits) if d["type"] == "choice"]
    out["choice_accuracy"] = float(np.mean([h for _, h in ch])) if ch else None

    sc = [d for d in items if d["type"] == "score"]
    if sc:
        out["score_exact_accuracy"] = float(np.mean([np.argmax(d["probs"]) == d["gold"] for d in sc]))
        out["score_mae_expected"] = float(np.mean([abs(np.dot(np.arange(len(d["probs"])), d["probs"]) - d["gold"])
                                                   for d in sc]))
        per_q: dict[str, list[dict]] = {}
        for d in sc:
            per_q.setdefault(d["question"], []).append(d)
        qwks = [quadratic_weighted_kappa(np.array([d["gold"] for d in ds]),
                                         np.array([int(np.argmax(d["probs"])) for d in ds]), len(ds[0]["probs"]))
                for ds in per_q.values()]
        qwks = [q for q in qwks if q is not None]
        out["score_qwk_mean"] = float(np.mean(qwks)) if qwks else None
    else:
        out["score_exact_accuracy"] = out["score_mae_expected"] = out["score_qwk_mean"] = None

    nl = [d for d in items if d["type"] == "noul"]
    if nl:
        p = np.array([d["probs"][0] for d in nl])
        y = np.array([d["gold"] for d in nl])
        out["noul_accuracy"] = float(np.mean((p >= 0.5).astype(int) == y))
        out["noul_auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
        out["noul_brier"] = float(np.mean((p - y) ** 2))
    else:
        out["noul_accuracy"] = out["noul_auroc"] = out["noul_brier"] = None

    dists = [decision_distribution(d["type"], d["probs"]) for d in items]
    conf = np.array([x.max() for x in dists])
    correct = np.array([int(np.argmax(x)) == d["gold"] for x, d in zip(dists, items)])
    out["ece"] = ece(conf, correct)
    with_gold = [(x, d) for x, d in zip(dists, items) if d.get("gold_probs") is not None]
    out["kl_from_gold"] = float(np.mean([kl_from_gold(d["gold_probs"], x) for x, d in with_gold])) if with_gold else None
    return out


def case_accuracy_arrays(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-case (correct decisions, decisions), for bootstrapping decision accuracy over cases."""
    cases: dict[str, list[int]] = {}
    for d in items:
        c = cases.setdefault(d["case"], [0, 0])
        c[0] += int(decision_hit(d["type"], d["probs"], d["gold"]))
        c[1] += 1
    arr = np.array(list(cases.values()), dtype=float)
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------- robustness
def tv_distance(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Row-wise total-variation distance 0.5 * sum |p - q|."""
    return 0.5 * np.abs(np.asarray(P, dtype=float) - np.asarray(Q, dtype=float)).sum(axis=-1)


def order_sensitivity(P: np.ndarray, P_reordered_aligned: np.ndarray) -> dict:
    """P_reordered_aligned: answers to the reordered question, mapped back to the original label order."""
    P, Q = np.asarray(P, dtype=float), np.asarray(P_reordered_aligned, dtype=float)
    return {
        "flip_rate": float(np.mean(P.argmax(axis=1) != Q.argmax(axis=1))),
        "mean_tv": float(np.mean(tv_distance(P, Q))),
        "n": int(len(P)),
    }


def negation_consistency(p_pos: np.ndarray, p_neg: np.ndarray) -> dict:
    """A statement and its negation should get probabilities summing to 1 and never agree."""
    a, b = np.asarray(p_pos, dtype=float), np.asarray(p_neg, dtype=float)
    return {
        "mean_abs_sum_minus_1": float(np.mean(np.abs(a + b - 1.0))),
        "contradiction_rate": float(np.mean(((a > 0.5) & (b > 0.5)) | ((a < 0.5) & (b < 0.5)))),
        "n": int(len(a)),
    }


# ---------------------------------------------------------------- uncertainty & systems
def bootstrap_ci(stat: Callable[..., float | None], arrays: Sequence[np.ndarray], n_resamples: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> dict | None:
    """Percentile bootstrap CI of `stat(*arrays)`, resampling examples. Resamples where the
    statistic is undefined are dropped; returns None if fewer than half are usable."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        v = _finite_or_none(stat(*(a[idx] for a in arrays)))
        if v is not None:
            vals.append(v)
    if len(vals) < n_resamples / 2:
        return None
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"low": float(lo), "high": float(hi), "n_resamples": len(vals)}


def latency_stats(latencies_s: Sequence[float]) -> dict | None:
    lat = np.asarray(latencies_s, dtype=float)
    if lat.size == 0:
        return None
    return {
        "p50_ms": float(np.percentile(lat, 50) * 1000),
        "p95_ms": float(np.percentile(lat, 95) * 1000),
        "mean_ms": float(lat.mean() * 1000),
        "throughput_rps": float(lat.size / lat.sum()) if lat.sum() > 0 else None,
        "n": int(lat.size),
    }
