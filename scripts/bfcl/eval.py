"""BFCL evaluation for the Sys One router (CPU, background-friendly).

Loads bfcl_items.jsonl, runs the frozen classifier on mappable items +
all irrelevance items, and updates bfcl_report.json incrementally.
Saves per-item predictions to bfcl_preds.json every 20 items.
"""
import json, time, os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "/home2/srallaba/projects/system-one-tool-router"
torch.set_num_threads(4)

label_map = json.load(open(f"{BASE}/data/label_map.json"))["label2id"]
id2label = {v: k for k, v in label_map.items()}
no_tool_id = label_map["no_tool"]
T = json.load(open(f"{BASE}/exp/temperature.json"))["temperature"]

items = json.load(open(f"{BASE}/data/bfcl/items.jsonl"))
work = [it for it in items if it["mappable"]]
print(f"work items: {len(work)}", flush=True)

CKPT = f"{BASE}/exp/checkpoints/qwen3-0.6b-tool-router/best"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT).eval()
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

preds = []
def save_preds():
    json.dump(preds, open(f"{BASE}/exp/bfcl/preds.json", "w"))

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
            preds.append({"id": b["id"], "split": b["split"],
                          "gt_classes": b["gt_classes"],
                          "top3": [id2label[int(x)] for x in od[:3]],
                          "top3_probs": [round(float(pr[x]), 4) for x in od[:3]],
                          "conf": round(float(pr[od[0]]), 4)})
        if len(preds) % 20 == 0:
            save_preds()
            print(f"  {len(preds)}/{len(work)} done", flush=True)
save_preds()

# ---- metrics ----
rep = json.load(open(f"{BASE}/exp/bfcl/report.json"))
out = {}
# irrelevance -> all GT are no_tool
irr = [p for p in preds if p["split"] == "irrelevance"]
irr_pred_notool = [p for p in irr if p["top3"][0] == "no_tool"]
irr_conf = np.array([p["conf"] for p in irr])
out["irrelevance"] = {
    "n": len(irr),
    "no_tool_rate": round(len(irr_pred_notool)/max(1,len(irr)), 4),
    "conf_mean": round(float(irr_conf.mean()), 4),
    "conf_median": round(float(np.median(irr_conf)), 4),
    "conf_hist": np.histogram(irr_conf, bins=10, range=(0,1))[0].tolist(),
    "top_wrong_preds": {k: v for k, v in
        __import__("collections").Counter(p["top3"][0] for p in irr if p["top3"][0] != "no_tool").most_common(10)},
}
# single-GT items (simple + multiple splits, exactly one gt class)
sg = [p for p in preds if p["split"] in ("simple","multiple") and len(p["gt_classes"])==1]
sg_correct = np.array([p["top3"][0] == p["gt_classes"][0] for p in sg])
sg_conf = np.array([p["conf"] for p in sg])
sel = sg_conf >= 0.9
out["single_gt"] = {
    "n": len(sg),
    "accuracy": round(float(sg_correct.mean()), 4) if len(sg) else None,
    "selective_acc_at_0_9": round(float(sg_correct[sel].mean()), 4) if sel.sum() else None,
    "coverage_at_0_9": round(float(sel.mean()), 4) if len(sg) else None,
    "conf_mean": round(float(sg_conf.mean()), 4) if len(sg) else None,
}
per_class = {}
for p in sg:
    c = p["gt_classes"][0]
    per_class.setdefault(c, [0,0])
    per_class[c][1] += 1
    per_class[c][0] += (p["top3"][0] == c)
out["single_gt"]["per_class"] = {k: {"correct": a, "n": b, "acc": round(a/b,3)} for k,(a,b) in sorted(per_class.items())}
# parallel diagnostic: top-1 in GT set? any GT in top-3? all GT in top-3?
par = [p for p in preds if p["split"] in ("parallel","parallel_multiple")]
def stats(ps):
    if not ps: return {"n": 0}
    top1 = np.mean([p["top3"][0] in p["gt_classes"] for p in ps])
    any3 = np.mean([any(c in p["top3"] for c in p["gt_classes"]) for p in ps])
    all3 = np.mean([all(c in p["top3"] for c in p["gt_classes"]) for p in ps])
    return {"n": len(ps), "top1_in_gt_set": round(float(top1),4),
            "any_gt_in_top3": round(float(any3),4), "all_gt_in_top3": round(float(all3),4)}
out["parallel"] = {"parallel": stats([p for p in par if p["split"]=="parallel"]),
                     "parallel_multiple": stats([p for p in par if p["split"]=="parallel_multiple"])}
rep["results"] = out
rep["inference"] = {"device": "cpu", "temperature": T, "threshold": 0.9,
                      "note": "GPU0 unhealthy; CPU fallback"}
json.dump(rep, open(f"{BASE}/exp/bfcl/report.json", "w"), indent=2)
print(json.dumps(out, indent=2))
