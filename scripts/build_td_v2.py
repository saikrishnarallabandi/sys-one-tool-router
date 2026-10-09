"""Build comprehensive typed-decisions v2 train set from external sources + v1.

Each output line matches kev --data format:
  {"state": {...}, "questions": {qid: {type, instructions, criteria, label, target}}}

    .venv/bin/python scripts/build_td_v2.py --out .cache/decisionbench/typed-decisions-v2.jsonl

Sources (all skipped gracefully on download failure):
  v1 (local typed-decisions-kev-train), chaosnli, helpsteer2, ultrafeedback,
  rewardbench, jigsaw, ethics, mmlu, hellaswag, fever.
"""

from __future__ import annotations

import argparse
import json
import random

CAPS = {
    "chaosnli": 1500, "helpsteer2": 2000, "ultrafeedback": 2000,
    "rewardbench": 1500, "jigsaw": 3000, "ethics": 2000, "mmlu": 2000,
    "hellaswag": 2000, "fever": 2000, "pubmedqa": 2000,
}


def conv_pubmedqa(rng):
    from datasets import load_dataset
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        print(f"pubmedqa: SKIP ({e})"); return []
    out = []
    for r in ds:
        ans = r.get("final_decision", "")
        if ans == "maybe":
            continue
        ctx = " ".join(r.get("context", {}).get("contexts", [])[:3])
        st = {"task": "answer the biomedical question", "source": "pubmedqa"}
        out.append(rec(st, "answer", noul_q(
            f"Question: {r.get('question', '')}\nContext: {ctx[:1200]}",
            ans == "yes",
            crit={"false": "No.", "true": "Yes."})))
    rng.shuffle(out)
    print(f"pubmedqa: {len(out)}")
    return out[:CAPS["pubmedqa"]]


def rec(state, qid, q):
    return {"state": state, "questions": {qid: q}}


def choice_q(instructions, options, label, target=None):
    return {"type": "choice", "instructions": instructions,
            "criteria": {o: o for o in options}, "label": label,
            "target": target or {o: (1.0 if o == label else 0.0) for o in options}}


def noul_q(instructions, label, target=None, crit=None):
    crit = crit or {"false": "No.", "true": "Yes."}
    return {"type": "noul", "instructions": instructions, "criteria": crit,
            "label": bool(label),
            "target": target or {"false": 1.0 - float(label), "true": float(label)}}


def score_q(instructions, levels, label, target=None):
    return {"type": "score", "instructions": instructions, "criteria": levels,
            "label": int(label),
            "target": target or {str(i): (1.0 if i == label else 0.0) for i in range(len(levels))}}


def one_hot_choice(options, label):
    return {o: (1.0 if o == label else 0.0) for o in options}


def conv_v1(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"v1: {len(rows)} records")
    return rows


def conv_chaosnli(rng):
    from datasets import load_dataset
    out = []
    # Eval-only: upstream marks these validation; training on them would
    # contaminate future ChaosNLI evals. Caller routes output to eval.
    cfgs = ["snli", "mnli_m"]
    for cfg in cfgs:
        try:
            ds = load_dataset("earino/chaosnli", cfg, split="validation")
        except Exception as e:
            print(f"chaosnli/{cfg}: SKIP ({e})"); continue
        opts = ["entailment", "neutral", "contradiction"]
        mmap = {"e": "entailment", "n": "neutral", "c": "contradiction"}
        for r in ds:
            dist = r.get("label_dist") or []
            if len(dist) != 3 or sum(dist) <= 0:
                continue
            s = sum(dist)
            tgt = {o: dist[i] / s for i, o in enumerate(opts)}
            lab = mmap.get(r.get("majority_label"), max(opts, key=lambda o: tgt[o]))
            ex = r.get("example", {}) or {}
            st = {"task": "decide the logical relation", "source": f"chaosnli-{cfg}"}
            ctx = f"Premise: {ex.get('premise', '')}\nHypothesis: {ex.get('hypothesis', '')}"
            out.append(rec(st, "relation", choice_q(
                "What is the logical relation? " + ctx, opts, lab, tgt)))
    rng.shuffle(out)
    print(f"chaosnli (eval-only): {len(out)}")
    return out


def conv_helpsteer2(rng):
    from datasets import load_dataset
    try:
        ds = load_dataset("nvidia/HelpSteer2", split="train")
    except Exception as e:
        print(f"helpsteer2: SKIP ({e})"); return []
    levels = [f"L{i}: helpfulness level {i} of 4" for i in range(5)]
    out = []
    for r in ds:
        try:
            s = int(r["helpfulness"])
        except Exception:
            continue
        st = {"task": r["prompt"][:500], "source": "helpsteer2"}
        out.append(rec(st, "helpfulness", score_q(
            "How helpful is this response?\nResponse: " + r["response"][:1500],
            levels, s)))
    rng.shuffle(out)
    print(f"helpsteer2: {len(out)}")
    return out[:CAPS["helpsteer2"]]


def conv_ultrafeedback(rng):
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        col = "binarized"
    except Exception as e:
        print(f"ultrafeedback: SKIP ({e})"); return []
    out = []
    for r in ds:
        ch, rj = r["chosen"], r["rejected"]
        prompt = (ch[0]["content"] if isinstance(ch, list) else r.get("prompt", ""))[:500]
        co = (ch[-1]["content"] if isinstance(ch, list) else str(ch))[:1200]
        ro = (rj[-1]["content"] if isinstance(rj, list) else str(rj))[:1200]
        opts = ["A", "B"]
        if rng.random() < 0.5:
            opts, co, ro = ["B", "A"], ro, co
            lab = "B"
        else:
            lab = "A"
        st = {"task": prompt, "source": "ultrafeedback"}
        out.append(rec(st, "better", choice_q(
            f"Which response is better?\nA: {co}\nB: {ro}", ["A", "B"], lab)))
    rng.shuffle(out)
    print(f"ultrafeedback: {len(out)}")
    return out[:CAPS["ultrafeedback"]]


def conv_rewardbench(rng, eval_only=True):
    from datasets import load_dataset
    try:
        ds = load_dataset("allenai/reward-bench", split="filtered")
    except Exception:
        try:
            ds = load_dataset("allenai/reward-bench", split="raw")
        except Exception as e:
            print(f"rewardbench: SKIP ({e})"); return []
    out = []
    for r in ds:
        try:
            ch, rj = r["chosen"], r["rejected"]
        except KeyError:
            continue
        prompt = str(r.get("prompt", ""))[:500]
        st = {"task": prompt, "source": "rewardbench"}
        out.append(rec(st, "better", choice_q(
            f"Which response is better?\nA: {str(ch)[:1200]}\nB: {str(rj)[:1200]}",
            ["A", "B"], "A")))
    rng.shuffle(out)
    print(f"rewardbench: {len(out)}")
    return out[:CAPS["rewardbench"]]


def conv_jigsaw(rng):
    from datasets import load_dataset
    try:
        ds = load_dataset("ucberkeley-dlab/measuring-hate-speech",
                          "default", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"jigsaw: SKIP ({e})"); return []
    levels = ["Not hateful.", "Maybe hateful.", "Hateful."]
    from collections import defaultdict
    votes = defaultdict(list)
    for r in ds:
        try:
            votes[str(r["text"])].append(max(0, min(2, int(r["hatespeech"]))))
        except Exception:
            continue
    out = []
    for txt, vs in votes.items():
        tgt = {str(i): vs.count(i) / len(vs) for i in range(3)}
        lab = max(range(3), key=lambda i: tgt[str(i)])
        st = {"task": "moderate this post", "source": "hatespeech"}
        out.append(rec(st, "hateful", score_q(
            "How hateful is this post?\n" + txt[:1200], levels, lab, tgt)))
    rng.shuffle(out)
    print(f"hatespeech: {len(out)} unique texts")
    return out[:CAPS["jigsaw"]]


def conv_ethics(rng):
    # No script-free moral-judgment source found (moral_stories needs a dataset
    # script; MMLU aux rows carry no subject field to filter). Dropped by design.
    print("ethics: SKIP (no script-free source; covered by noul/score elsewhere)")
    return []
    out = []
    letters = ["A", "B", "C", "D"]
    for r in ds:
        chs = list(r["choices"])
        lab = letters[int(r["answer"])]
        st = {"task": "judge the scenario", "source": "mmlu-moral_scenarios"}
        body = r["question"] + "\n" + "\n".join(f"{L}: {c}" for L, c in zip(letters, chs))
        out.append(rec(st, "answer", choice_q(body, letters, lab)))
    rng.shuffle(out)
    print(f"moral_scenarios: {len(out)}")
    return out[:CAPS["ethics"]]


def conv_mmlu(rng):
    from datasets import load_dataset
    try:
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    except Exception as e:
        print(f"mmlu: SKIP ({e})"); return []
    out = []
    letters = ["A", "B", "C", "D"]
    for r in ds:
        chs = list(r["choices"])
        lab = letters[int(r["answer"])]
        st = {"task": "answer the question", "source": "mmlu-" + str(r.get("subject", ""))}
        body = r["question"] + "\n" + "\n".join(f"{L}: {c}" for L, c in zip(letters, chs))
        out.append(rec(st, "answer", choice_q(body, letters, lab)))
    rng.shuffle(out)
    print(f"mmlu: {len(out)}")
    return out[:CAPS["mmlu"]]


def conv_hellaswag(rng):
    from datasets import load_dataset
    try:
        ds = load_dataset("Rowan/hellaswag", split="train")
    except Exception as e:
        print(f"hellaswag: SKIP ({e})"); return []
    out = []
    letters = ["A", "B", "C", "D"]
    for r in ds:
        chs = list(r["endings"])
        try:
            lab = letters[int(r["label"])]
        except Exception:
            continue
        st = {"task": "complete the scenario", "source": "hellaswag"}
        body = r["ctx"] + "\n" + "\n".join(f"{L}: {c}" for L, c in zip(letters, chs))
        out.append(rec(st, "ending", choice_q(body, letters, lab)))
    rng.shuffle(out)
    print(f"hellaswag: {len(out)}")
    return out[:CAPS["hellaswag"]]


def conv_fever(rng):
    from datasets import load_dataset
    try:
        ds = load_dataset("facebook/anli", split="train_r1")
    except Exception as e:
        print(f"anli: SKIP ({e})"); return []
    opts = ["entailment", "neutral", "contradiction"]
    out = []
    for r in ds:
        try:
            lab = opts[int(r["label"])]
        except Exception:
            continue
        st = {"task": "decide the logical relation", "source": "anli"}
        ctx = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
        out.append(rec(st, "relation", choice_q(
            "What is the logical relation? " + ctx, opts, lab)))
    rng.shuffle(out)
    print(f"anli: {len(out)}")
    return out[:CAPS["fever"]]


EVAL_FRAC = 0.12
EVAL_MIN, EVAL_MAX = 100, 400


def norm_text(r):
    import re
    parts = [json.dumps(r.get("state", ""), sort_keys=True)]
    for q in r.get("questions", {}).values():
        parts.append(q.get("instructions", ""))
        c = q.get("criteria", {})
        parts.append(json.dumps(c, sort_keys=True))
    s = " ".join(parts).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v1", default=".cache/decisionbench/typed-decisions-kev-train.jsonl")
    ap.add_argument("--v1-test", default=".cache/decisionbench/typed-decisions-kev-test.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    train_rows, eval_rows = conv_v1(a.v1), []
    for fn in [conv_helpsteer2, conv_ultrafeedback,
               conv_jigsaw, conv_ethics, conv_mmlu, conv_hellaswag, conv_fever,
               conv_pubmedqa]:
        try:
            rows = fn(rng)
        except Exception as e:
            print(f"{fn.__name__}: SKIP ({type(e).__name__}: {e})")
            continue
        if not rows:
            print(f"{fn.__name__}: empty, skipped")
            continue
        n_eval = min(EVAL_MAX, max(EVAL_MIN, int(len(rows) * EVAL_FRAC)))
        eval_rows.extend(rows[:n_eval])
        train_rows.extend(rows[n_eval:])
        print(f"{fn.__name__}: train={len(rows) - n_eval} eval={n_eval}")
    for fn in [conv_chaosnli, conv_rewardbench]:
        try:
            rows = fn(rng)
        except Exception as e:
            print(f"{fn.__name__}: SKIP ({type(e).__name__}: {e})")
            continue
        eval_rows.extend(rows)
        print(f"{fn.__name__}: eval-only={len(rows)}")
    # Leakage audit: v2 train vs v1 test, and within v2.
    ref_keys = set()
    try:
        for line in open(a.v1_test):
            if line.strip():
                ref_keys.add(norm_text(json.loads(line)))
        print(f"v1-test refs: {len(ref_keys)}")
    except FileNotFoundError:
        print("v1-test: MISSING, audit vs test skipped")
    seen, dup_in, leak_test, clean = set(), 0, 0, []
    for r in train_rows:
        k = norm_text(r)
        if k in ref_keys:
            leak_test += 1
            continue
        if k in seen:
            dup_in += 1
            continue
        seen.add(k)
        clean.append(r)
    print(f"audit: dropped {leak_test} v1-test leaks, {dup_in} in-train dups")
    rng.shuffle(clean)
    rng.shuffle(eval_rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r) + "\n")
    with open(a.eval_out, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"TRAIN {len(clean)} -> {a.out}")
    print(f"EVAL {len(eval_rows)} -> {a.eval_out}")


if __name__ == "__main__":
    main()
