"""Selective routing analysis: risk-coverage curves for the Sys One router.

Loads the test set, runs the trained classifier, applies the fitted
temperature, and computes coverage vs. selective accuracy at confidence
thresholds, plus no_tool-specific abstention behavior.
"""
import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "/home2/srallaba/projects/system-one-tool-router"
CKPT = f"{BASE}/exp/checkpoints/qwen3-0.6b-tool-router/best"

def main():
    label_map = json.load(open(f"{BASE}/data/label_map.json"))["label2id"]
    id2label = {v: k for k, v in label_map.items()}
    no_tool_id = label_map["no_tool"]
    T = json.load(open(f"{BASE}/exp/temperature.json"))["temperature"]

    rows = [json.loads(l) for l in open(f"{BASE}/data_test.jsonl")]
    texts = [r["text"] for r in rows]
    y_true = np.array([int(r["label"]) for r in rows])

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT).cuda().eval()

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits.float()
            all_logits.append(logits.cpu())
    logits = torch.cat(all_logits).numpy() / T
    # softmax
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    conf = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)
    correct = (y_pred == y_true)

    out = {"n": len(rows), "temperature": T,
           "overall_acc": float(correct.mean())}

    # Risk-coverage at fixed coverage levels: pick threshold achieving coverage
    order = np.argsort(-conf)  # descending confidence
    curve = []
    for cov in [1.00, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80]:
        k = max(1, int(len(rows) * cov))
        sel = order[:k]
        acc = float(correct[sel].mean())
        thr = float(conf[order[k - 1]])
        curve.append({"coverage": cov, "threshold": round(thr, 4),
                      "selective_acc": round(acc, 4), "n_selected": k,
                      "n_errors": int((~correct[sel]).sum())})
    out["risk_coverage"] = curve

    # Threshold sweep (fixed thresholds)
    sweep = []
    for thr in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
        sel = conf >= thr
        cov = float(sel.mean())
        acc = float(correct[sel].mean()) if sel.sum() else None
        sweep.append({"threshold": thr, "coverage": round(cov, 4),
                      "selective_acc": round(acc, 4) if acc is not None else None,
                      "n_selected": int(sel.sum())})
    out["threshold_sweep"] = sweep

    # no_tool behavior: among abstained (low-conf) items, what fraction are no_tool or errors?
    is_notool_true = (y_true == no_tool_id)
    is_notool_pred = (y_pred == no_tool_id)
    out["no_tool"] = {
        "prevalence": float(is_notool_true.mean()),
        "pred_rate": float(is_notool_pred.mean()),
        # precision/recall of no_tool as predicted class
        "precision": float((is_notool_true & is_notool_pred).sum() / max(1, is_notool_pred.sum())),
        "recall": float((is_notool_true & is_notool_pred).sum() / max(1, is_notool_true.sum())),
        # errors the router makes: how many are low confidence (catchable by abstention)?
    }
    err = ~correct
    out["errors_catchable"] = {
        "n_errors": int(err.sum()),
        "err_conf_mean": float(conf[err].mean()),
        "ok_conf_mean": float(conf[~err].mean()),
        # fraction of errors below 0.9 confidence
        "frac_errors_below_0_9": float((conf[err] < 0.9).mean()),
        "frac_correct_below_0_9": float((conf[~err] < 0.9).mean()),
    }

    # Confidence histogram (10 bins) for the paper
    hist, edges = np.histogram(conf, bins=10, range=(0, 1))
    out["conf_histogram"] = {"counts": hist.tolist(),
                             "edges": [round(float(x), 2) for x in edges]}

    json.dump(out, open(f"{BASE}/exp/selective_report.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
