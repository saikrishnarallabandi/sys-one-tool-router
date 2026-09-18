"""Matched generative baseline: Qwen3-0.6B as a generative tool router.

Same test set, same 25-way decision, but the model must GENERATE the tool
name instead of classifying. Measures accuracy, macro-F1, latency (p50/p99),
parse-failure rate, and confidence availability.
"""
import json, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score

BASE = "/home2/srallaba/projects/system-one-tool-router"

TOOLS = [
    "analyze_sentiment", "calculate_age", "calculate_area", "calculate_bmi",
    "calculate_discount", "calculate_distance", "calculate_loan_payment",
    "calculate_mortgage_payment", "calculate_tip", "convert_currency",
    "create_calendar_event", "create_todo", "generate_password",
    "generate_qr_code", "generate_random_number", "get_definition",
    "get_movie_details", "get_news", "get_stock_price", "no_tool",
    "search_books", "search_movies", "search_recipes", "send_email",
    "translate_text",
]

PROMPT = (
    "You are a tool router. Given the user message, output EXACTLY ONE tool name "
    "from the list below, and nothing else.\n\nTools: {tools}\n\n"
    "User message: {text}\nTool:"
)

def normalize(s):
    s = s.strip().lower()
    s = re.split(r"[\s,.;:!\n\"']", s)[0]
    return s

def main():
    label_map = json.load(open(f"{BASE}/data/label_map.json"))["label2id"]
    rows = [json.loads(l) for l in open(f"{BASE}/data_test.jsonl")]
    texts = [r["text"] for r in rows]
    y_true = np.array([int(r["label"]) for r in rows])
    tool_list = ", ".join(TOOLS)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", trust_remote_code=True,
        torch_dtype=torch.float16).cuda().eval()

    y_pred, lat, n_parse_fail = [], [], 0
    with torch.no_grad():
        for i, text in enumerate(texts):
            prompt = PROMPT.format(tools=tool_list, text=text)
            enc = tok(prompt, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            dt = time.perf_counter() - t0
            gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            name = normalize(gen)
            lat.append(dt * 1000)
            if name in label_map:
                y_pred.append(label_map[name])
            else:
                n_parse_fail += 1
                y_pred.append(-1)
            if (i + 1) % 100 == 0:
                print(f"done {i+1}/{len(texts)}", flush=True)

    y_pred = np.array(y_pred)
    valid = y_pred >= 0
    acc = float((y_pred[valid] == y_true[valid]).mean()) if valid.sum() else 0.0
    # macro-F1 over the 25 real classes; parse failures count as errors
    # (false negatives for their true class) without inventing a 26th class
    ncls = len(TOOLS)
    tp = np.zeros(ncls); fp = np.zeros(ncls); fn = np.zeros(ncls)
    for t, p in zip(y_true, y_pred):
        if p < 0:
            fn[t] += 1
        elif p == t:
            tp[t] += 1
        else:
            fp[p] += 1; fn[t] += 1
    denom = 2 * tp + fp + fn
    f1s = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    macro_f1 = float(f1s.mean())
    lat = np.array(lat)
    report = {
        "n": len(texts),
        "accuracy": round(acc, 4),
        "accuracy_incl_parse_fail_as_wrong": round(float((np.where(valid, y_pred, -99) == y_true).mean()), 4),
        "macro_f1": round(macro_f1, 4),
        "latency_ms": {"mean": round(float(lat.mean()), 1),
                       "p50": round(float(np.percentile(lat, 50)), 1),
                       "p99": round(float(np.percentile(lat, 99)), 1)},
        "parse_failures": n_parse_fail,
        "parse_failure_rate": round(n_parse_fail / len(texts), 4),
        "confidence_available": False,
        "throughput_per_sec": round(1000.0 / float(lat.mean()), 1),
    }
    json.dump(report, open(f"{BASE}/exp/baseline_gen_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
