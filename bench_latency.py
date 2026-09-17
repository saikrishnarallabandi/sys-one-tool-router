#!/usr/bin/env python3
"""Latency benchmark: single-sample (incl. tokenization) + batch throughput.
fp16 on a single GTX 1080 Ti. Mirrors system-one-poc/bench_latency.py."""
import json, time, statistics, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CKPT = "checkpoints/qwen3-0.6b-tool-router/best"
MAX_LEN = 128

SAMPLES = [
    "What is a 20 percent tip on an $85 restaurant bill?",
    "Convert 150 US dollars to euros at the current rate",
    "Send an email to Priya about the meeting tomorrow at 10am",
    "What are the latest news headlines for India?",
    "Generate a secure password for my new account",
    "What is my BMI if I weigh 70kg and am 175cm tall?",
    "Remind me to buy milk tomorrow morning",
    "Translate 'good morning' to Spanish",
    "What is the current stock price of Tesla?",
    "Can you book a flight for me from New York to London?",
]

def main():
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

    # warmup
    for s in SAMPLES[:4]:
        enc = tok(s, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            model(**enc)
    torch.cuda.synchronize()

    lat = []
    for s in SAMPLES * 10:  # 100 single-sample runs
        t0 = time.perf_counter()
        enc = tok(s, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            model(**enc)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    print(f"single-sample (n=100, incl. tokenization):")
    print(f"  mean {statistics.mean(lat):.1f} ms  p50 {lat[50]:.1f} ms  p99 {lat[99]:.1f} ms")

    for bs in (16, 64):
        enc = tok(SAMPLES * (bs // len(SAMPLES) + 1), padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(model.device)
        enc = {k: v[:bs] for k, v in enc.items()}
        t0 = time.perf_counter(); reps = 20
        with torch.no_grad():
            for _ in range(reps):
                model(**enc)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        print(f"batch {bs}: {bs * reps / secs:.0f} samples/sec")

if __name__ == "__main__":
    main()
