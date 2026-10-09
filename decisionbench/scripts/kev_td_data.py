"""Convert cached typed-decisions rows to kev `--data` JSONL (label + soft target).

    .venv/bin/python scripts/kev_td_data.py \
        --in .cache/decisionbench/typed-decisions-train.jsonl \
        --out .cache/decisionbench/typed-decisions-kev-train.jsonl

kev's `load_records` takes request-shaped rows with a `label` per question
(choice: option name; noul: bool; score: level index) and an optional `target`
(choice: {option: p}; noul: {"false": p, "true": p}; score: {str(level): p}).
kev's `question_loss` trains cross-entropy against the soft target when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_row(r: dict) -> dict:
    questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
    gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
    out_q = {}
    for qid, q in questions.items():
        g = gold[qid]
        p = g["probabilities"]
        if q["type"] == "choice":
            out_q[qid] = {"type": "choice", "instructions": q["instructions"],
                          "criteria": dict(q["criteria"]),
                          "label": g["label"], "target": dict(p)}
        elif q["type"] == "score":
            out_q[qid] = {"type": "score", "instructions": q["instructions"],
                          "criteria": list(q["criteria"]),
                          "label": int(g["label"]), "target": {str(k): v for k, v in p.items()}}
        elif q["type"] == "noul":
            lab = g["label"]
            if lab not in ("true", "false"):
                raise ValueError(f"noul label {lab!r} in {r['id']}/{qid}")
            entry: dict = {"type": "noul", "instructions": q["instructions"],
                           "label": lab == "true", "target": dict(p)}
            if q.get("criteria"):
                entry["criteria"] = dict(q["criteria"])
            out_q[qid] = entry
        else:
            raise ValueError(f"unknown type {q['type']!r}")
    return {"state": r["state"], "questions": out_q}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in Path(args.inp).open() if l.strip()]
    if args.n is not None:
        rows = rows[:args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(convert_row(r)) + "\n")
    print(f"wrote {len(rows)} records to {out}")


if __name__ == "__main__":
    main()
