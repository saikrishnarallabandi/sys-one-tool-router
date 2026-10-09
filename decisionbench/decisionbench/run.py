"""Benchmark runner.

    python -m decisionbench.run --models laya-english,nli-zeroshot,jev --tasks all --n 300 --robustness

Layout of the output directory:
    preds/<model>__<task>.jsonl       one record per example (probabilities, latency, errors, probes)
    preds/<model>__<task>.meta.json   the sample and settings the records belong to
    models/<model>.json               load time, first-request latency, device, pricing
    skipped.json                      models that could not run, with the reason
    metrics.json, report.md           written by decisionbench.report (recomputable from preds/)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from decisionbench import report
from decisionbench.models import create, registry
from decisionbench.models.base import DecisionModel
from decisionbench.tasks import Example, Task, get_tasks
from decisionbench.types import InvalidOutput, ModelUnavailable

# A (model, task) whose first this-many requests all fail is abandoned: the model is recorded as
# skipped with the error, instead of burning through the remaining examples.
ABORT_AFTER_FAILURES = 5
RECORD_FORMAT = 2


def pair_stem(model: str, task: str) -> str:
    return f"{model}__{task}"


def _json_default(o):
    if hasattr(o, "tolist"):  # numpy scalars and arrays
        return o.tolist()
    return str(o)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(path)


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def upgrade_record(r: dict, meta: dict) -> dict:
    """Round-1 files held one question per record at the top level; lift them into `answers`."""
    if "answers" in r:
        return r
    qid, ttype = meta["question_id"], meta["task_type"]
    out = {k: r[k] for k in ("i", "id", "error", "latency_s", "input_tokens", "warm") if k in r}
    out["gold"] = {qid: r["gold"]}
    out["answers"] = {} if r.get("error") else {
        qid: {"type": ttype, "probs": r["probs"], "pred": r["pred"], "raw": r.get("raw"), "extra": r.get("extra") or {}}}
    if "probes" in r:
        out["probes"] = {name: ({"error": pr["error"]} if pr.get("error") else {"error": None, "answers": {qid: pr}})
                         for name, pr in r["probes"].items()}
    return out


def read_records(path: Path, meta: dict | None = None) -> list[dict]:
    """Parse a predictions file, ignoring a torn final line from an interrupted run."""
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [upgrade_record(r, meta) for r in records] if meta is not None else records


def reusable(record: dict, ex: Example, i: int, robustness: bool, retry_errors: bool) -> bool:
    if record.get("i") != i or record.get("id") != ex.id:
        return False
    if retry_errors and record.get("error"):
        return False
    probes = record.get("probes") or {}
    for name in (ex.probes if robustness else {}):
        if name not in probes or (retry_errors and probes[name].get("error")):
            return False
    return True


def plan_pair(out: Path, model: str, task: Task, examples: list[Example], seed: int, robustness: bool,
              model_opts: dict, retry_errors: bool) -> tuple[dict[int, dict], list[int]]:
    """Existing reusable records by index, and the indices still to run."""
    stem = out / "preds" / pair_stem(model, task.name)
    meta = _read_json(stem.with_suffix(".meta.json"), {})
    keep: dict[int, dict] = {}
    if meta.get("seed") == seed and meta.get("model_opts", {}) == model_opts:
        for r in read_records(stem.with_suffix(".jsonl"), meta):
            i = r.get("i")
            if isinstance(i, int) and 0 <= i < len(examples) and reusable(r, examples[i], i, robustness, retry_errors):
                keep[i] = r
    todo = [i for i in range(len(examples)) if i not in keep]
    return keep, todo


def _answer_probe(model: DecisionModel, ex: Example, name: str) -> dict:
    specs = ex.probes[name]
    try:
        answers = model.predict(ex.state, specs)
        missing = [qid for qid in specs if qid not in answers]
        if missing:
            raise InvalidOutput(f"no answer for {missing}")
    except ModelUnavailable:
        raise
    except Exception as e:  # recorded per probe; counted in the robustness n
        return {"error": f"{type(e).__name__}: {e}"}
    out = {}
    for qid, spec in specs.items():
        ans = answers[qid]
        if spec.type == "choice":
            # Map back to the original label order so it is comparable with the primary answer.
            by_label = dict(zip(spec.labels, ans.probs_list()))
            out[qid] = {"probs_aligned": [by_label[lab] for lab in ex.questions[qid].labels]}
        else:
            out[qid] = {"p": float(ans.probs)} if spec.type == "noul" else {"probs": ans.probs_list()}
    return {"error": None, "answers": out}


def run_example(model: DecisionModel, ex: Example, i: int, robustness: bool) -> dict:
    record: dict = {"i": i, "id": ex.id, "group": ex.group, "gold": ex.gold, "error": None, "answers": {}}
    if ex.gold_probs:
        record["gold_probs"] = ex.gold_probs
    t0 = time.perf_counter()
    try:
        answers = model.predict(ex.state, ex.questions)
        record["latency_s"] = time.perf_counter() - t0
        missing = [qid for qid in ex.questions if qid not in answers]
        if missing:
            raise InvalidOutput(f"no answer for {missing}")
        record["input_tokens"] = next(iter(answers.values())).input_tokens
        record["answers"] = {qid: {"type": a.type, "probs": a.probs_list(), "pred": a.pred, "raw": a.raw,
                                   "extra": a.extra} for qid, a in answers.items() if qid in ex.questions}
    except ModelUnavailable:
        raise
    except Exception as e:  # a failed request is a data point (error rate), not a crash
        record.setdefault("latency_s", time.perf_counter() - t0)
        record["error"] = f"{type(e).__name__}: {e}"
        record["answers"] = {}
    if robustness and ex.probes:
        record["probes"] = {name: _answer_probe(model, ex, name) for name in ex.probes}
    return record


def run_pair(out: Path, model: DecisionModel, task: Task, examples: list[Example], keep: dict[int, dict],
             todo: list[int], seed: int, n: int | None, robustness: bool, model_opts: dict, warmup: int) -> None:
    stem = out / "preds" / pair_stem(model.name, task.name)
    stem.parent.mkdir(parents=True, exist_ok=True)
    _write_json(stem.with_suffix(".meta.json"), {
        "format": RECORD_FORMAT, "model": model.name, "task": task.name, "task_type": task.type,
        "question_id": task.question_id, "source": task.source, "split": task.split, "n_requested": n,
        "n_examples": len(examples), "seed": seed, "robustness": robustness, "model_opts": model_opts,
        "example_ids": [ex.id for ex in examples],
        "n_options": examples[0].questions[task.question_id].n_options if examples and task.question_id else None,
    })
    path = stem.with_suffix(".jsonl")
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for i in sorted(keep):
            f.write(json.dumps(keep[i]) + "\n")
    tmp.replace(path)

    failures, abort = 0, None
    with path.open("a") as f:
        for k, i in enumerate(tqdm(todo, desc=f"{model.name} / {task.name}", unit="ex", file=sys.stderr)):
            rec = run_example(model, examples[i], i, robustness)
            # The first requests of a session on a task pay one-off costs (kernel compilation for
            # new shapes, connection setup); they are kept as predictions but excluded from latency.
            rec["warm"] = k >= warmup
            f.write(json.dumps(rec, default=_json_default) + "\n")
            f.flush()
            if rec["error"]:
                failures += 1
                if failures == k + 1 >= ABORT_AFTER_FAILURES:
                    abort = f"first {failures} requests on {task.name} failed; last: {rec['error']}"
                    break
    if abort:
        # Drop this session's failures so a later resume retries them instead of reusing them.
        with tmp.open("w") as f:
            for i in sorted(keep):
                f.write(json.dumps(keep[i]) + "\n")
        tmp.replace(path)
        raise ModelUnavailable(abort)


def run(models: list[str], tasks: list[Task], n: int | None, seed: int, robustness: bool, out: Path, warmup: int,
        model_opts: dict[str, dict], retry_errors: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    history = _read_json(out / "runs.json", [])
    history.append({"started": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:]})
    _write_json(out / "runs.json", history)

    examples = {t.name: t.load(n, seed) for t in tasks}
    skipped = _read_json(out / "skipped.json", {})

    for name in models:
        opts = model_opts.get(name, {})
        plans = {t.name: plan_pair(out, name, t, examples[t.name], seed, robustness, opts, retry_errors) for t in tasks}
        pending = [t for t in tasks if plans[t.name][1]]
        for t in tasks:
            if t not in pending:
                print(f"[skip] {name} / {t.name}: {len(examples[t.name])} predictions already complete", file=sys.stderr)
        if not pending:
            continue

        model = None
        try:
            t0 = time.perf_counter()
            model = create(name, **opts)
            load_time = time.perf_counter() - t0
            sidecar = {"name": name, "deployment": model.deployment, "device": model.device,
                       "price_per_1m_input_tokens": model.price_per_1m_input_tokens, "info": model.info(),
                       "model_opts": opts, "load_time_s": load_time,
                       "loaded_at": datetime.now().isoformat(timespec="seconds")}
            print(f"[load] {name} on {model.device} in {load_time:.1f}s", file=sys.stderr)
            first = True
            for t in pending:
                keep, todo = plans[t.name]
                if first:
                    # Cold first request: a throwaway call on the first example, before any timing.
                    ex0 = examples[t.name][0]
                    t1 = time.perf_counter()
                    try:
                        model.predict(ex0.state, ex0.questions)
                        sidecar["first_request_s"] = time.perf_counter() - t1
                    except ModelUnavailable:
                        raise
                    except Exception as e:  # the same failure will be recorded per example below
                        sidecar["first_request_error"] = f"{type(e).__name__}: {e}"
                    _write_json(out / "models" / f"{name}.json", sidecar)
                    first = False
                run_pair(out, model, t, examples[t.name], keep, todo, seed, n, robustness, opts, warmup)
            skipped.pop(name, None)
        except ModelUnavailable as e:
            print(f"[skip] {name}: {e}", file=sys.stderr)
            skipped[name] = str(e)
        finally:
            if model is not None:
                model.close()
            _write_json(out / "skipped.json", skipped)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m decisionbench.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="laya-english,nli-zeroshot,jev", help="comma-separated registered model names")
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated task names")
    ap.add_argument("--n", type=int, default=None,
                    help="examples per task (seeded sample of the eval split); default 300, except typed-decisions, "
                         "which uses its whole test split (400 cases) unless --n is smaller")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robustness", action="store_true",
                    help="also run option-order reversal (choice questions) and negation (sst2) probes")
    ap.add_argument("--out", default=None, help="output directory (default results/<timestamp>); reuse it to resume")
    ap.add_argument("--warmup", type=int, default=3, help="requests per (model, task) excluded from latency stats")
    ap.add_argument("--model-opts", default="{}",
                    help='JSON of per-model constructor overrides, e.g. \'{"laya-english": {"head_max_len": 384}}\'')
    ap.add_argument("--retry-errors", action="store_true", help="on resume, re-run examples that previously errored")
    ap.add_argument("--list", action="store_true", help="list registered models and tasks, then exit")
    args = ap.parse_args(argv)

    if args.list:
        from decisionbench.tasks import TASKS
        print("models:", ", ".join(sorted(registry())))
        print("tasks: ", ", ".join(f"{k} ({t.type})" for k, t in TASKS.items()))
        return
    if args.n is not None and args.n < 1:
        ap.error("--n must be >= 1")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in registry()]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; registered: {', '.join(sorted(registry()))}")
    try:
        tasks = get_tasks(args.tasks)
    except KeyError as e:
        ap.error(str(e))
    try:
        model_opts = json.loads(args.model_opts)
    except json.JSONDecodeError as e:
        ap.error(f"--model-opts is not valid JSON: {e}")
    if not isinstance(model_opts, dict) or not all(isinstance(v, dict) for v in model_opts.values()):
        ap.error("--model-opts must map model name -> object of constructor kwargs")

    out = Path(args.out or f"results/{datetime.now():%Y%m%d-%H%M%S}")
    run(models, tasks, args.n, args.seed, args.robustness, out, args.warmup, model_opts, args.retry_errors)
    report.build(out)
    print(f"wrote {out / 'report.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
