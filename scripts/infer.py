#!/usr/bin/env python3
"""System One tool-router inference: utterance in -> typed JSON routing decision.

Usage:
    python infer.py "What is 20% tip on a $85 bill?"
    echo "Send an email to the team" | python infer.py

Output: {"tool": "calculate_tip",
         "probabilities": {"calculate_tip": 0.93, ...},
         "confidence": 0.93, "latency_ms": 28.4}
"""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CKPT = "exp/checkpoints/qwen3-0.6b-tool-router/best"
MAX_LEN = 128

_tok, _model, _T, _id2label = None, None, 1.0, None

def load():
    global _tok, _model, _T, _id2label
    if _model is not None:
        return
    _tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    _model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
    _id2label = {int(k): v for k, v in json.load(open("data/label_map.json"))["id2label"].items()}
    try:
        _T = json.load(open("exp/temperature.json"))["temperature"]
    except FileNotFoundError:
        _T = 1.0

def route(text: str) -> dict:
    load()
    enc = _tok(text, truncation=True, padding=True, max_length=MAX_LEN,
               return_tensors="pt").to(_model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = _model(**enc).logits.float() / _T
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(logits, dim=1)[0].tolist()
    best = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {"tool": _id2label[best],
            "probabilities": { _id2label[i]: round(p, 4) for i, p in enumerate(probs)},
            "confidence": round(max(probs), 4),
            "latency_ms": round(ms, 2)}

def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read().strip()
    print(json.dumps(route(text), indent=2))

if __name__ == "__main__":
    main()
