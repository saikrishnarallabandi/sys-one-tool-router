"""Frozen-router eval on Glaive v2 (CPU, background-friendly).

Eval set: mappable items in the constructed 5% test split whose EXACT text does
not appear in our train/val/test (avoids train-set contamination, since our
dataset was built from this same Glaive file).
Saves per-item predictions incrementally to glaive_preds.json; final metrics
go into glaive_report.json.
"""
import json, time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "/home2/srallaba/projects/system-one-tool-router"
torch.set_num_threads(4)

label_map = json.load(open(f"{BASE}/data/label_map.json"))["label2id"]
id2label = {v: k for k, v in label_map.items()}
no_tool_id = label_map["no_tool"]
T = json.load(open(f"{BASE}/exp/temperature.json"))["temperature"]

ours = set()
for f in ["data/data_train.jsonl", "data/data_val.jsonl", "data/data_test.jsonl"]:
    for line in open(f"{BASE}/{f}"):
        ours.add(json.loads(line)["text"].strip().lower())

rows = json.load(open(f"{BASE}/data/glaive/items.jsonl"))
work = [r for r in rows if r["mappable"] and r["split"] == "test"
        and r["text"].strip().lower() not in ours]
print(f"work items: {len(work)}", flush=True)

CKPT = f"{BASE}/exp/checkpoints/qwen3-0.6b-tool-router/best"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT).eval()
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

preds = []
def save_preds():
    json.dump(preds, open(f"{BASE}/exp/glaive/preds.json", "w"))

with torch.no_grad():
    for i in range(0, len(work), 4):
        batch = work[i:i+4]
        enc = tok([b["text"] for b in batch], padding=True, truncation=True,
                  max_length=256, return_tensors="pt")
        logits = model(**enc).logits.numpy() / T
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        order = np.argsort(-probs, axis=1)
        for b, pr, od in zip(batch, probs, order):
            preds.append({"id": b["id"], "gt_classes": b["gt_classes"],
                          "top1": id2label[int(od[0])],
                          "top3": [id2label[int(x)] for x in od[:3]],
                          "conf": round(float(pr[od[0]]), 4)})
        if len(preds) % 40 == 0:
            save_preds()
            print(f"  {len(preds)}/{len(work)} done", flush=True)
save_preds()

# ---- metrics ----
correct = np.array([p["top1"] in p["gt_classes"] for p in preds])
conf = np.array([p["conf"] for p in preds])
sel = conf >= 0.9
out = {
    "n": len(preds),
    "accuracy": round(float(correct.mean()), 4),
    "selective_acc_at_0_9": round(float(correct[sel].mean()), 4) if sel.sum() else None,
    "coverage_at_0_9": round(float(sel.mean()), 4),
    "conf_mean": round(float(conf.mean()), 4),
}
# no_tool behavior
is_nt = np.array([p["gt_classes"] == ["no_tool"] for p in preds])
pred_nt = np.array([p["top1"] == "no_tool" for p in preds])
out["no_tool"] = {
    "n_gt": int(is_nt.sum()),
    "recall": round(float((is_nt & pred_nt).sum() / max(1, is_nt.sum())), 4),
    "precision": round(float((is_nt & pred_nt).sum() / max(1, pred_nt.sum())), 4),
    "conf_mean_on_gt": round(float(conf[is_nt].mean()), 4) if is_nt.sum() else None,
}
per_class = {}
for p in preds:
    for c in p["gt_classes"]:
        a, b = per_class.get(c, (0, 0))
        per_class[c] = (a + (p["top1"] == c), b + 1)
out["per_class"] = {k: {"correct": a, "n": b, "acc": round(a/b, 3)}
                    for k, (a, b) in sorted(per_class.items())}

rep = json.load(open(f"{BASE}/exp/glaive/report.json"))
rep["results"] = out
rep["inference"] = {"device": "cpu", "temperature": T, "threshold": 0.9,
                    "note": "GPU0 unhealthy; CPU fallback. Eval on novel texts only."}
json.dump(rep, open(f"{BASE}/exp/glaive/report.json", "w"), indent=2)
print(json.dumps(out, indent=2))
