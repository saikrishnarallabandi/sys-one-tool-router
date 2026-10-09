"""Recompute metrics from saved predictions and render report.md.

    python -m decisionbench.report results/<dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from decisionbench import contamination as C
from decisionbench import metrics as M

HEADLINE = {"choice": "accuracy", "score": "qwk", "noul": "auroc", "multi": "decision_accuracy"}
HEADLINE_LABEL = {"choice": "accuracy", "score": "QWK", "noul": "AUROC", "multi": "decision accuracy"}


def _headline_stat(task_type: str, n_options: int):
    if task_type == "choice":
        return lambda P, y: float(np.mean(P.argmax(axis=1) == y))
    return lambda P, y: M.quadratic_weighted_kappa(y, P.argmax(axis=1), n_options)


def _auroc_only(p, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def _robustness(valid: list[dict], by_i: dict) -> dict:
    orig, rev, pos, neg = [], [], [], []
    for r in valid:
        probes = r.get("probes") or {}
        for qid, a in ((probes.get("reversed") or {}).get("answers") or {}).items():
            if "probs_aligned" in a and qid in r["answers"]:
                orig.append(r["answers"][qid]["probs"])
                rev.append(a["probs_aligned"])
        for qid, a in ((probes.get("negation") or {}).get("answers") or {}).items():
            if "p" in a and qid in r["answers"]:
                pos.append(r["answers"][qid]["probs"][0])
                neg.append(a["p"])
    rob: dict = {}
    if orig:
        # Rows can differ in length (several choice questions per case), so compare row by row.
        flips = [int(np.argmax(o) != np.argmax(q)) for o, q in zip(orig, rev)]
        tvs = [float(M.tv_distance(np.array(o), np.array(q))) for o, q in zip(orig, rev)]
        rob["option_order"] = {"flip_rate": float(np.mean(flips)), "mean_tv": float(np.mean(tvs)), "n": len(orig)}
    if pos:
        rob["negation"] = M.negation_consistency(np.array(pos), np.array(neg))
    probe_err = sum(1 for r in by_i.values() for pr in (r.get("probes") or {}).values() if pr.get("error"))
    if rob or probe_err:
        rob["probe_errors"] = probe_err
    return rob


def pair_metrics(meta: dict, records: list[dict], seed: int = 0) -> dict:
    """Metrics for one (model, task) from its prediction records."""
    ttype = meta["task_type"]
    n_total = meta["n_examples"]
    by_i = {r["i"]: r for r in records}
    valid = [by_i[i] for i in sorted(by_i) if not by_i[i].get("error")]
    n_done = len(by_i)
    out: dict = {
        "task_type": ttype, "n_examples": n_total, "n_done": n_done, "n_valid": len(valid),
        "complete": n_done == n_total,
        "error_rate": (n_done - len(valid)) / n_done if n_done else None,
        "errors_sample": sorted({r["error"] for r in by_i.values() if r.get("error")})[:3],
        "headline": HEADLINE[ttype],
    }
    if not valid:
        return out
    if ttype == "multi":
        items = [{"case": r["id"], "group": r.get("group"), "question": f"{r.get('group')}/{qid}", "type": a["type"],
                  "probs": a["probs"], "gold": r["gold"][qid], "gold_probs": (r.get("gold_probs") or {}).get(qid)}
                 for r in valid for qid, a in r["answers"].items()]
        out["metrics"] = M.multi_question_metrics(items)
        hits, counts = M.case_accuracy_arrays(items)
        out["headline_ci"] = M.bootstrap_ci(lambda h, c: float(h.sum() / c.sum()), [hits, counts], seed=seed)
    else:
        qid = meta["question_id"]
        y = np.array([r["gold"][qid] for r in valid], dtype=int)
        if ttype == "noul":
            p = np.array([r["answers"][qid]["probs"][0] for r in valid], dtype=float)
            out["metrics"] = M.noul_metrics(p, y, seed)
            out["headline_ci"] = M.bootstrap_ci(_auroc_only, [p, y], seed=seed)
        else:
            P = np.array([r["answers"][qid]["probs"] for r in valid], dtype=float)
            out["metrics"] = M.choice_metrics(P, y, seed) if ttype == "choice" else M.score_metrics(P, y)
            out["headline_ci"] = M.bootstrap_ci(_headline_stat(ttype, P.shape[1]), [P, y], seed=seed)

    rob = _robustness(valid, by_i)
    if rob:
        out["robustness"] = rob
    warm_ok = [r["latency_s"] for r in valid if r.get("warm") and r.get("latency_s") is not None]
    out["latency"] = M.latency_stats(warm_ok)
    toks = [r["input_tokens"] for r in valid if r.get("input_tokens") is not None]
    out["input_tokens_mean"] = float(np.mean(toks)) if toks else None
    out["decisions_per_request"] = float(np.mean([len(r["answers"]) for r in valid]))
    return out


def systems_metrics(model: str, sidecar: dict | None, pairs: dict[str, tuple[dict, list[dict]]]) -> dict:
    """Pooled over every task the model ran."""
    recs = [r for _, rs in pairs.values() for r in rs]
    ok = [r for r in recs if not r.get("error")]

    def warm_latencies(multi: bool) -> list[float]:
        # One-question requests and multi-question cases differ in size, so they are pooled separately.
        return [r["latency_s"] for meta, rs in pairs.values() if (meta["task_type"] == "multi") == multi
                for r in rs if not r.get("error") and r.get("warm") and r.get("latency_s") is not None]
    with_tokens = [r for r in ok if r.get("input_tokens") is not None]
    sidecar = sidecar or {}
    price = sidecar.get("price_per_1m_input_tokens")
    deployment = sidecar.get("deployment")
    tok_mean = float(np.mean([r["input_tokens"] for r in with_tokens])) if with_tokens else None
    if deployment == "local":
        cost = 0.0
    elif price is not None and with_tokens:
        # Billing is per request; spread it over the decisions each request answered.
        tokens_per_decision = sum(r["input_tokens"] for r in with_tokens) / sum(len(r["answers"]) for r in with_tokens)
        cost = tokens_per_decision * 1000 * price / 1e6
    else:
        cost = None
    served = sorted({a.get("extra", {}).get("served_model") for r in ok for a in r["answers"].values()} - {None})
    return {
        "device": sidecar.get("device"), "deployment": deployment, "info": sidecar.get("info"),
        "load_time_s": sidecar.get("load_time_s"), "first_request_s": sidecar.get("first_request_s"),
        "latency": M.latency_stats(warm_latencies(False)),
        "latency_multi": M.latency_stats(warm_latencies(True)),
        "n_requests": len(recs),
        "error_rate": (len(recs) - len(ok)) / len(recs) if recs else None,
        "input_tokens_per_request": tok_mean,
        "usd_per_1k_decisions": cost,
        "served_models": served,
    }


def _headline_value(m: dict) -> float | None:
    return (m.get("metrics") or {}).get(m.get("headline"))


def summarize(res: dict) -> dict:
    """Per-model summary (mean rank, held-out headline, ECE after TS) and the held-out comparison."""
    tasks = res["tasks"]
    models = sorted(res["systems"])
    ranks: dict[str, list[float]] = {m: [] for m in models}
    for task, t in tasks.items():
        vals = {m: _headline_value(x) for m, x in t["models"].items()}
        vals = {m: v for m, v in vals.items() if v is not None}
        if len(vals) < 2:
            continue
        names = list(vals)
        for m, r in zip(names, rankdata([-vals[m] for m in names])):
            ranks[m].append(float(r))
    summary = {}
    for m in models:
        held = [v for task, t in tasks.items() if m in t["models"]
                and C.overlap(m, task) is None and (v := _headline_value(t["models"][m])) is not None]
        ts = [((t["models"][m].get("metrics") or {}).get("temperature_scaling") or {}).get("ece_eval_after")
              for t in tasks.values() if m in t["models"]]
        ts = [x for x in ts if x is not None]
        summary[m] = {"mean_rank": float(np.mean(ranks[m])) if ranks[m] else None, "n_ranked": len(ranks[m]),
                      "heldout_mean_headline": float(np.mean(held)) if held else None, "n_heldout": len(held),
                      "mean_ece_after_ts": float(np.mean(ts)) if ts else None,
                      "disclosed": C.disclosed(m)}

    general = [m for m in models if not any(C.is_specialist(m, task) for task in tasks)]
    heldout_tasks, excluded = [], {}
    for task in tasks:
        dirty = [m for m in general if C.disclosed(m) and C.overlap(m, task) is not None]
        if task in C.TASK_SOURCES and not dirty:
            heldout_tasks.append(task)
        else:
            excluded[task] = dirty or ["(task not in contamination map)"]
    rows = {}
    for m in general:
        vals = {task: _headline_value(tasks[task]["models"][m]) if m in tasks[task]["models"] else None
                for task in heldout_tasks}
        have = [v for v in vals.values() if v is not None]
        rows[m] = {"values": vals, "mean": float(np.mean(have)) if len(have) == len(heldout_tasks) and have else None}
    return {"models": summary, "heldout": {"tasks": heldout_tasks, "excluded": excluded, "rows": rows,
                                           "specialists": [m for m in models if m not in general]}}


def compute(out: Path) -> dict:
    from decisionbench.run import read_records  # local import: run imports this module
    from decisionbench.tasks import TASKS, typed_decisions_reference

    preds = out / "preds"
    pairs: dict[str, dict[str, tuple[dict, list[dict]]]] = {}
    for meta_path in sorted(preds.glob("*.meta.json")) if preds.exists() else []:
        meta = json.loads(meta_path.read_text())
        ids = meta["example_ids"]
        records = [r for r in read_records(meta_path.with_name(meta_path.name.replace(".meta.json", ".jsonl")), meta)
                   if isinstance(r.get("i"), int) and r["i"] < len(ids) and ids[r["i"]] == r.get("id")]
        pairs.setdefault(meta["model"], {})[meta["task"]] = (meta, records)

    result: dict = {"tasks": {}, "systems": {}, "skipped": {}}
    for model, tasks in pairs.items():
        for task, (meta, records) in tasks.items():
            t = result["tasks"].setdefault(task, {"type": meta["task_type"], "source": meta["source"],
                                                  "split": meta["split"], "n_options": meta.get("n_options"),
                                                  "models": {}})
            t["models"][model] = pair_metrics(meta, records, meta.get("seed", 0))
            if task == "typed-decisions":
                ids = sorted({r["id"] for r in records})
                if ids and len(ids) > len(t.get("reference_ids", [])):
                    t["reference_ids"] = ids
        sidecar_path = out / "models" / f"{model}.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None
        result["systems"][model] = systems_metrics(model, sidecar, tasks)
    rank = {name: i for i, name in enumerate(TASKS)}
    result["tasks"] = dict(sorted(result["tasks"].items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0])))
    td = result["tasks"].get("typed-decisions")
    if td is not None and td.get("reference_ids"):
        td["reference"] = typed_decisions_reference(td.pop("reference_ids"))
    skipped_path = out / "skipped.json"
    if skipped_path.exists():
        result["skipped"] = json.loads(skipped_path.read_text())
    result["summary"] = summarize(result)
    return result


# ---------------------------------------------------------------- markdown
def _f(x, nd: int = 3) -> str:
    if x is None:
        return "–"
    return f"{x:.{nd}f}"


def _ci(m: dict, model: str, task: str) -> str:
    val = _headline_value(m)
    ci = m.get("headline_ci")
    s = f"{_f(val)} [{_f(ci['low'], 2)}, {_f(ci['high'], 2)}]" if ci else _f(val)
    return s + C.mark(model, task)


def _ts(ts: dict | None) -> str:
    if not ts:
        return "–"
    return f"{_f(ts['ece_eval_before'])} → {_f(ts['ece_eval_after'])} (T={ts['temperature']:.2f})"


def _n(m: dict) -> str:
    s = f"{m['n_valid']}/{m['n_examples']}"
    return s if m["complete"] else s + " (incomplete)"


def _cell(s: str) -> str:
    return s.replace("|", "\\|")


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(map(_cell, headers)) + " |", "|" + "|".join("---" for _ in headers) + "|"] + \
        ["| " + " | ".join(map(_cell, r)) + " |" for r in rows]


def _rows(res: dict, ttype: str):
    for task, t in res["tasks"].items():
        if t["type"] == ttype:
            for model, m in sorted(t["models"].items()):
                yield task, t, model, m


def _render_summary(res: dict) -> list[str]:
    S = res["summary"]
    L = ["## Summary", "",
         "Mean rank: rank of the headline metric among models that ran the same task (1 = best), averaged. "
         "Held-out headline: mean headline metric over tasks the model did not train on (per the contamination "
         "map; headline metrics differ by task type, so compare within a row set, not across tables). "
         "ECE after TS: mean over tasks with a temperature-scaling fold.", ""]
    rows = []
    for model, s in sorted(S["models"].items(), key=lambda kv: (kv[1]["mean_rank"] is None, kv[1]["mean_rank"] or 0)):
        sys_ = res["systems"][model]
        lat = sys_.get("latency") or {}
        cost = sys_.get("usd_per_1k_decisions")
        cost_s = "$0 (local)" if sys_.get("deployment") == "local" else (f"${cost:.5f}" if cost is not None else "–")
        held = (f"{_f(s['heldout_mean_headline'])} ({s['n_heldout']} tasks)" if s["disclosed"]
                else "? (training data undisclosed)")
        lat_multi = sys_.get("latency_multi") or {}
        rows.append([model, f"{_f(s['mean_rank'], 2)} ({s['n_ranked']} tasks)", held,
                     _f(s["mean_ece_after_ts"]), _f(lat.get("p50_ms"), 1), _f(lat_multi.get("p50_ms"), 1), cost_s])
    L += _table(["model", "mean rank ↓", "held-out mean headline", "mean ECE after TS ↓",
                 "p50 ms / 1-question request", "p50 ms / typed-decisions case", "$ / 1k decisions"], rows) + [""]
    return L


def _render_heldout(res: dict) -> list[str]:
    H = res["summary"]["heldout"]
    L = ["## Held-out comparison", "",
         "Tasks that no compared model trained on, derived from the contamination map (models whose training "
         "data is undisclosed are included and marked ?). Models fine-tuned on a benchmark's own training split "
         "are left out of this table.", ""]
    if not H["tasks"]:
        L += ["No task is held out for every model in this run.", ""]
    else:
        rows = []
        n_max = {t: max(m["n_valid"] for m in res["tasks"][t]["models"].values()) for t in H["tasks"]}

        def cell(model: str, t: str) -> str:
            m = res["tasks"][t]["models"].get(model)
            v = _f(H["rows"][model]["values"][t])
            return v + (f" (n={m['n_valid']} of {n_max[t]})" if m and m["n_valid"] < n_max[t] else "")

        order = sorted(H["rows"].items(), key=lambda kv: -(kv[1]["mean"] if kv[1]["mean"] is not None else -1))
        for i, (model, r) in enumerate(order, 1):
            name = model + ("" if C.disclosed(model) else " ?")
            rows.append([str(i) if r["mean"] is not None else "–", name]
                        + [cell(model, t) for t in H["tasks"]] + [_f(r["mean"])])
        L += _table(["rank", "model"] + [f"{t} ({HEADLINE_LABEL[res['tasks'][t]['type']]})" for t in H["tasks"]]
                    + ["mean"], rows) + [""]
    if H["excluded"]:
        L += ["Excluded tasks: " + "; ".join(f"{t} ({', '.join(ms)})" for t, ms in H["excluded"].items()) + "."]
    if H["specialists"]:
        L += ["Left out as specialists: " + ", ".join(H["specialists"]) + "."]
    L.append("")
    return L


def _render_typed_decisions(res: dict) -> list[str]:
    t = res["tasks"].get("typed-decisions")
    if t is None:
        return []
    L = ["## typed-decisions", "",
         "All five questions of a case are asked in one request. Decision accuracy: the model's argmax "
         "(noul: p ≥ 0.5) equals the dataset's gold `label` (the argmax of its teacher-averaged gold "
         "distribution, ties already broken). The dataset card does not define its leaderboard accuracy, so "
         "these are not guaranteed to match its numbers. CI: bootstrap over cases. KL: KL(gold ‖ model) "
         "against the full gold distribution, in nats. QWK: mean over score questions.", ""]
    rows = []
    for model, m in sorted(t["models"].items()):
        x = m.get("metrics") or {}
        name = model + (" (specialist)" if C.is_specialist(model, "typed-decisions") else "")
        rows.append([name, _n(m), _ci(m, model, "typed-decisions"), _f(x.get("choice_accuracy")),
                     _f(x.get("score_exact_accuracy")), _f(x.get("score_mae_expected")), _f(x.get("score_qwk_mean")),
                     _f(x.get("noul_accuracy")), _f(x.get("noul_auroc")), _f(x.get("noul_brier")),
                     _f(x.get("kl_from_gold")), _f(x.get("ece"))])
    ref = t.get("reference")
    if ref:
        rows.append(["prior (train label frequencies, no input)", f"{ref['n_decisions']} decisions", _f(ref["prior"])]
                    + ["–"] * 9)
        rows.append(["uniform (expected, no input)", "", _f(ref["uniform"])] + ["–"] * 9)
    L += _table(["model", "cases", "decision acc [95% CI]", "choice acc", "score exact acc", "score MAE",
                 "score QWK", "noul acc", "noul AUROC", "noul Brier ↓", "KL from gold ↓", "ECE ↓"], rows) + [""]
    groups = sorted({g for m in t["models"].values() for g in ((m.get("metrics") or {}).get("accuracy_by_group") or {})})
    if groups:
        rows = [[model] + [_f(((m.get("metrics") or {}).get("accuracy_by_group") or {}).get(g)) for g in groups]
                for model, m in sorted(t["models"].items())]
        if ref:
            rows.append(["prior"] + [_f(ref["prior_by_group"].get(g)) for g in groups])
        L += ["Decision accuracy per workflow (config):", ""]
        L += _table(["model"] + groups, rows) + [""]
    return L


def _render_footnotes(res: dict) -> list[str]:
    cells = [(model, task, C.overlap(model, task)) for task, t in res["tasks"].items() for model in t["models"]]
    marked = [(m, t, k) for m, t, k in cells if k in ("direct", "corpus")]
    unknown = sorted({m for m, t, k in cells if k == "unknown" and not C.disclosed(m)})
    L = ["## Training overlap", "",
         "† the model was trained on this task's dataset (a different split). ‡ the model was trained on a dataset "
         "built from the same source texts (SST-2 / SST-5 / rotten_tomatoes). Marked cells measure in-distribution "
         "performance, not zero-shot generalization.", ""]
    for model in sorted({m for m, _, _ in marked}):
        items = ", ".join(f"{t}{C.MARKS[k]}" for m, t, k in marked if m == model)
        L.append(f"- **{model}**: {items} (source: {C.source(model)})")
    for model in unknown:
        L.append(f"- **{model}**: training data undisclosed (?); no overlap can be ruled out")
    L.append("")
    return L


def render(res: dict) -> str:
    L = ["# decisionbench report", ""]
    L += ["Headline metric per task type (with 95% bootstrap CI, 1000 resamples): choice = accuracy, "
          "score = quadratic weighted kappa, noul = AUROC, typed-decisions = decision accuracy. `n` is "
          "valid/sampled examples; metrics use valid examples only (see error rates in the systems table). "
          "ECE → ECE after temperature scaling is on the 70% evaluation fold, T fitted on the other 30%. "
          "† / ‡ mark training overlap (see the end of the report).", ""]
    L += _render_summary(res)
    L += _render_heldout(res)

    L += ["## Choice tasks", ""]
    rows = []
    for task, t, model, m in _rows(res, "choice"):
        x = m.get("metrics") or {}
        rows.append([f"{task} ({t['n_options']})", model, _n(m), _ci(m, model, task), _f(x.get("macro_f1")),
                     _f(x.get("top2_accuracy")), _f(x.get("log_loss")), _f(x.get("brier")), _f(x.get("ece")),
                     _ts(x.get("temperature_scaling")), _f(x.get("aurc")), _f(x.get("coverage_at_95_acc"))])
    L += _table(["task (labels)", "model", "n", "accuracy [95% CI]", "macro-F1", "top-2 acc", "log-loss", "Brier",
                 "ECE", "ECE eval-fold → after TS", "AURC ↓", "cov@95%acc"], rows) + [""]

    L += ["## Score tasks", ""]
    rows = []
    for task, t, model, m in _rows(res, "score"):
        x = m.get("metrics") or {}
        rows.append([f"{task} ({t['n_options']})", model, _n(m), _f(x.get("exact_accuracy")),
                     _f(x.get("within1_accuracy")), _f(x.get("mae_expected")), _f(x.get("spearman")),
                     _ci(m, model, task), _f(x.get("ece")), _f(x.get("log_loss"))])
    L += _table(["task (levels)", "model", "n", "exact acc", "within-1 acc", "MAE (E[score])", "Spearman ρ",
                 "QWK [95% CI]", "ECE", "log-loss"], rows) + [""]

    L += ["## Noul tasks", ""]
    rows = []
    for task, t, model, m in _rows(res, "noul"):
        x = m.get("metrics") or {}
        best = f"{_f(x.get('best_f1'))} @{_f(x.get('best_threshold'), 2)}" if x.get("best_f1") is not None else "–"
        rows.append([task, model, _n(m), _ci(m, model, task), _f(x.get("auprc")), _f(x.get("accuracy")),
                     _f(x.get("f1")), best, _f(x.get("brier")), _f(x.get("log_loss")), _f(x.get("ece")),
                     _ts(x.get("temperature_scaling"))])
    L += _table(["task", "model", "n", "AUROC [95% CI]", "AUPRC", "acc@0.5", "F1@0.5", "best F1 (oracle thr)",
                 "Brier", "log-loss", "ECE", "ECE eval-fold → after TS"], rows) + [""]

    L += _render_typed_decisions(res)

    order_rows, neg_rows = [], []
    for task, t in res["tasks"].items():
        for model, m in sorted(t["models"].items()):
            rob = m.get("robustness") or {}
            if "option_order" in rob:
                o = rob["option_order"]
                order_rows.append([task, model, str(o["n"]), _f(o["flip_rate"]), _f(o["mean_tv"]),
                                   str(rob.get("probe_errors", 0))])
            if "negation" in rob:
                g = rob["negation"]
                neg_rows.append([task, model, str(g["n"]), _f(g["mean_abs_sum_minus_1"]), _f(g["contradiction_rate"]),
                                 str(rob.get("probe_errors", 0))])
    if order_rows or neg_rows:
        L += ["## Robustness", ""]
    if order_rows:
        L += ["Option-order sensitivity: each choice question re-asked with its options reversed "
              "(typed-decisions: all choice questions of a case reversed in one request; n counts decisions).", ""]
        L += _table(["task", "model", "n", "prediction flip rate ↓", "mean TV distance ↓", "probe errors"],
                    order_rows) + [""]
    if neg_rows:
        L += ["Negation consistency: the statement and its negation asked separately.", ""]
        L += _table(["task", "model", "n", "mean |p_pos + p_neg − 1| ↓", "contradiction rate ↓", "probe errors"],
                    neg_rows) + [""]

    L += ["## Systems", "",
          "Latency is per request, sequential, warm (the first requests of each task are excluded). p50/p95/mean "
          "and req/s pool the one-question tasks; typed-decisions requests (all five questions of a case) are "
          "reported separately. Throughput = 1 / mean latency. Load time "
          "includes any first-time weight download and, for server-backed models, server start-up. Cost uses "
          "Jev's list price ($0.042 / 1M input tokens) spread over the decisions per request; local models cost "
          "$0 in API fees.", ""]
    rows = []
    for model, s in sorted(res["systems"].items()):
        lat = s.get("latency") or {}
        cost = s.get("usd_per_1k_decisions")
        cost_s = "$0 (local)" if s.get("deployment") == "local" else (f"${cost:.5f}" if cost is not None else "–")
        first = s.get("first_request_s")
        info = s.get("info") or {}
        runs = ", ".join(f"{m.get('run')} {m.get('backend', '')} {m.get('dtype', '')}".strip()
                         for m in info.get("models", [])[:1] if m.get("run"))
        answered = ", ".join(s.get("served_models") or []) + (f" ({runs})" if runs else "")
        lm = s.get("latency_multi") or {}
        rows.append([model, str(s.get("device")), answered or "–", _f(s.get("load_time_s"), 1),
                     _f(first * 1000 if first is not None else None, 0), _f(lat.get("p50_ms"), 1),
                     _f(lat.get("p95_ms"), 1), _f(lat.get("mean_ms"), 1), _f(lat.get("throughput_rps"), 1),
                     _f(lm.get("p50_ms"), 1), _f(lm.get("p95_ms"), 1),
                     _f(s.get("error_rate")), _f(s.get("input_tokens_per_request"), 0), cost_s])
    L += _table(["model", "device", "answered by", "load s", "first request ms", "p50 ms", "p95 ms", "mean ms",
                 "req/s", "typed-dec. case p50 ms", "typed-dec. case p95 ms", "error rate", "input tokens/req",
                 "$ / 1k decisions"], rows) + [""]

    tasks = list(res["tasks"])
    models = sorted(res["systems"])
    if tasks and models:
        L += ["Mean warm latency (ms) per request, per task:", ""]
        rows = []
        for model in models:
            row = [model]
            for task in tasks:
                lat = ((res["tasks"][task]["models"].get(model) or {}).get("latency") or {})
                row.append(_f(lat.get("mean_ms"), 1))
            rows.append(row)
        L += _table(["model"] + tasks, rows) + [""]

    errs = [(task, model, m) for task, t in res["tasks"].items() for model, m in sorted(t["models"].items())
            if m.get("errors_sample")]
    if errs:
        L += ["## Errors", ""]
        for task, model, m in errs:
            L.append(f"- {model} / {task}: error rate {_f(m['error_rate'])}; e.g. `{m['errors_sample'][0][:200]}`")
        L.append("")

    L += _render_footnotes(res)

    L += ["## Skipped models", ""]
    if res["skipped"]:
        L += [f"- **{model}**: {' '.join(reason.split())}" for model, reason in sorted(res["skipped"].items())]
    else:
        L.append("None.")
    L.append("")
    return "\n".join(L)


def build(out: Path) -> dict:
    out = Path(out)
    res = compute(out)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "report.md").write_text(render(res))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m decisionbench.report", description=__doc__)
    ap.add_argument("out", help="results directory written by decisionbench.run")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "preds").is_dir() and not (out / "skipped.json").exists():
        ap.error(f"{out} does not look like a decisionbench results directory")
    build(out)
    print(out / "report.md")


if __name__ == "__main__":
    main()
