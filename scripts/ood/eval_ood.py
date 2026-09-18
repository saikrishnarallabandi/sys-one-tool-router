"""Evaluate the frozen Sys One router on the OOD challenge set.

- Loads ood/ood_challenge.jsonl (fields: text, label, acceptable, category, context)
- Runs the trained classifier with T=1.791
- Reports overall + per-category accuracy (with acceptable-label credit),
  risk-coverage, abstention behavior, template-overlap stats.
"""
import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "/home2/srallaba/projects/system-one-tool-router"
CKPT = f"{BASE}/exp/checkpoints/qwen3-0.6b-tool-router/best"

def norm(s):
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()

def main():
    label_map = json.load(open(f"{BASE}/data/label_map.json"))["label2id"]
    id2label = {v: k for k, v in label_map.items()}
    T = json.load(open(f"{BASE}/exp/temperature.json"))["temperature"]
    items = [json.loads(l) for l in open(f"{BASE}/exp/ood/ood_challenge.jsonl")]

    texts = [(it["context"] + "\n" + it["text"] if it.get("context") else it["text"])
             for it in items]
    y_true = np.array([label_map[it["label"]] for it in items])
    accept = [set(label_map[a] for a in it.get("acceptable", [it["label"]])) for it in items]
    cats = [it["category"] for it in items]

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT).cuda().eval()

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits.float()
            all_logits.append(logits.cpu())
    logits = torch.cat(all_logits).numpy() / T
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    conf = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)

    strict = (y_pred == y_true)
    lenient = np.array([yp in a for yp, a in zip(y_pred, accept)])

    out = {"n": len(items), "temperature": T,
           "accuracy_strict": round(float(strict.mean()), 4),
           "accuracy_lenient": round(float(lenient.mean()), 4)}
    by_cat = {}
    for c in sorted(set(cats)):
        m = np.array(cats) == c
        by_cat[c] = {"n": int(m.sum()),
                     "acc_strict": round(float(strict[m].mean()), 4),
                     "acc_lenient": round(float(lenient[m].mean()), 4),
                     "mean_conf": round(float(conf[m].mean()), 4)}
    out["by_category"] = by_cat

    # risk-coverage on lenient correctness
    order = np.argsort(-conf)
    curve = []
    for cov in [1.00, 0.99, 0.95, 0.90, 0.80]:
        k = max(1, int(len(items) * cov))
        sel = order[:k]
        curve.append({"coverage": cov,
                      "selective_acc": round(float(lenient[sel].mean()), 4),
                      "escalated": int(len(items) - k)})
    out["risk_coverage"] = curve
    out["mean_confidence"] = round(float(conf.mean()), 4)
    out["frac_below_0_9"] = round(float((conf < 0.9).mean()), 4)

    # template overlap: share of 5-grams also seen in train
    train_grams = set()
    for l in open(f"{BASE}/data_train.jsonl"):
        toks = norm(json.loads(l)["text"]).split()
        train_grams.update(tuple(toks[i:i + 5]) for i in range(len(toks) - 4))
    overlap = []
    for it in items:
        toks = norm(it["text"]).split()
        grams = [tuple(toks[i:i + 5]) for i in range(len(toks) - 4)]
        overlap.append(sum(g in train_grams for g in grams) / max(1, len(grams)))
    out["fivegram_overlap_with_train"] = {
        "mean": round(float(np.mean(overlap)), 4),
        "median": round(float(np.median(overlap)), 4)}

    # exact dup check (normalized) vs train/val/test
    seen = set()
    for f in ["data/data_train.jsonl", "data/data_val.jsonl", "data/data_test.jsonl"]:
        for l in open(f"{BASE}/{f}"):
            seen.add(norm(json.loads(l)["text"]))
    dups = sum(norm(it["text"]) in seen for it in items)
    out["exact_dups_vs_splits"] = dups

    json.dump(out, open(f"{BASE}/exp/ood/ood_eval_report.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
