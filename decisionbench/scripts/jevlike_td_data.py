"""Convert typed-decisions kev JSONL to jevlike eval JSONL (context/options/label).

    .venv/bin/python scripts/jevlike_td_data.py \\
        --in .cache/decisionbench/typed-decisions-kev-test.jsonl \\
        --out .cache/decisionbench/typed-decisions-jevlike-test.jsonl
"""

from __future__ import annotations

import argparse
import json


def convert_row(r: dict) -> list[dict]:
    state = json.dumps(r["state"], sort_keys=True)
    out = []
    for q in r["questions"].values():
        ctx = state + "\n" + q["instructions"]
        if q["type"] == "choice":
            options = [f"{k}: {v}" for k, v in q["criteria"].items()]
            label = list(q["criteria"]).index(q["label"])
        elif q["type"] == "noul":
            if "criteria" in q:
                options = [q["criteria"]["false"], q["criteria"]["true"]]
            else:
                options = [f"No: {q['instructions']}", f"Yes: {q['instructions']}"]
            label = 1 if q["label"] else 0
        else:  # score
            options = list(q["criteria"])
            label = int(q["label"])
        out.append({"context": ctx, "options": options, "label": label,
                    "type": q["type"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    n = 0
    with open(a.inp, encoding="utf-8") as fh, open(a.out, "w", encoding="utf-8") as fo:
        for line in fh:
            if not line.strip():
                continue
            for ex in convert_row(json.loads(line)):
                fo.write(json.dumps(ex) + "\n")
                n += 1
    print(f"wrote {n} examples -> {a.out}")


if __name__ == "__main__":
    main()
