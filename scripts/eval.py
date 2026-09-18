#!/usr/bin/env python3
"""Evaluate the tool-router + temperature scaling on the test set.

Temperature is fit on the VALIDATION logits only (never the test set).
Reports accuracy, macro-F1, multi-class NLL, Brier, ECE(15) before/after
scaling, per-class precision/recall/F1, and the most-confused tool pairs.
Saves temperature.json and eval_report.json.
"""
import json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_recall_fscore_support, confusion_matrix)

CKPT = "exp/checkpoints/qwen3-0.6b-tool-router/best"
MAX_LEN = 128

def load(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line); texts.append(r["text"]); labels.append(r["label"])
    return texts, np.array(labels)

@torch.no_grad()
def logits_for(model, tok, texts, bs=64):
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, padding=True,
                  max_length=MAX_LEN, return_tensors="pt").to(model.device)
        outs.append(model(**enc).logits.float().cpu())
    return torch.cat(outs)

def ece_score(probs, labels, n_bins=15):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    acc = (pred == labels)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / len(labels)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)

def brier_multi(probs, labels):
    oh = np.zeros_like(probs); oh[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))

def report(name, logits, labels):
    probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "nll": float(log_loss(labels, probs)),
        "brier": brier_multi(probs, labels),
        "ece15": ece_score(probs, labels),
    }
    print(f"--- {name} ---")
    for k, v in out.items():
        print(f"  {k:9s}: {v:.4f}")
    return probs, out

def main():
    with open("data/label_map.json") as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

    # fit temperature on VALIDATION logits (NLL minimization)
    vtexts, vlabels = load("data/data_val.jsonl")
    vlogits = logits_for(model, tok, vtexts)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.5, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(vlogits / T, torch.tensor(vlabels))
        loss.backward(); return loss
    opt.step(closure)
    T_val = float(T.detach().clamp_min(1e-3))
    print(f"fitted temperature T = {T_val:.4f} (on val, n={len(vlabels)})")
    with open("exp/temperature.json", "w") as f:
        json.dump({"temperature": T_val}, f)

    report("val, uncalibrated (T=1.0)", vlogits, vlabels)
    report(f"val, scaled (T={T_val:.3f})", vlogits / T_val, vlabels)

    ttexts, tlabels = load("data/data_test.jsonl")
    tlogits = logits_for(model, tok, ttexts)
    _, before = report("test, uncalibrated (T=1.0)", tlogits, tlabels)
    probs_after, after = report(f"test, scaled (T={T_val:.3f})", tlogits / T_val, tlabels)

    # per-class analysis on the CALIBRATED test predictions
    pred = probs_after.argmax(axis=1)
    prec, rec, f1, sup = precision_recall_fscore_support(tlabels, pred, zero_division=0)
    per_class = []
    for i in range(len(id2label)):
        per_class.append({"tool": id2label[i], "precision": round(float(prec[i]), 4),
                          "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4),
                          "support": int(sup[i])})
    per_class.sort(key=lambda d: d["f1"])
    print("\n--- 5 weakest tools by F1 ---")
    for d in per_class[:5]:
        print(f"  {d['tool']:28s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} n={d['support']}")
    print("--- 5 strongest tools by F1 ---")
    for d in per_class[-5:][::-1]:
        print(f"  {d['tool']:28s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} n={d['support']}")

    # most confused pairs (true -> predicted)
    cm = confusion_matrix(tlabels, pred)
    pairs = []
    n = len(id2label)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], id2label[i], id2label[j]))
    pairs.sort(reverse=True)
    print("\n--- top 12 confusion pairs (true -> predicted) ---")
    for cnt, a, b in pairs[:12]:
        print(f"  {a:28s} -> {b:28s}  x{cnt}")

    with open("exp/eval_report.json", "w") as f:
        json.dump({"temperature": T_val, "n_test": len(tlabels),
                   "test_before": before, "test_after": after,
                   "per_class": per_class,
                   "confusions": [{"true": a, "pred": b, "count": int(c)} for c, a, b in pairs[:25]]},
                  f, indent=2)
    print("\nwrote temperature.json, eval_report.json")

if __name__ == "__main__":
    main()
