"""Audit glaive-function-calling-v2 schema: splits, functions, labels. Read-only."""
import json, re
from collections import Counter

BASE = "/home2/srallaba/projects/system-one-tool-router"
DATA = f"{BASE}/data/glaive/raw/glaive-function-calling-v2.json"

print("loading...", flush=True)
items = json.load(open(DATA))
print(f"n_items={len(items)}", flush=True)

spec_fn_counts = Counter()       # function names declared in system spec
called_fn_counts = Counter()     # function names actually called in chat
n_multi_spec = 0
n_zero_calls = 0
n_multi_called = 0
user_lens = []
chat_lens = []
spec_mismatch = 0

NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
FNCALL = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?=<\|endoftext\|>)", re.S)

for idx, it in enumerate(items):
    system, chat = it["system"], it["chat"]
    # --- spec: embedded function-spec JSON after the first '{' ---
    m = system.find("{")
    spec_names = []
    if m >= 0:
        try:
            spec, _ = json.JSONDecoder().raw_decode(system[m:])
            if isinstance(spec, dict) and "name" in spec:
                spec_names = [spec["name"]]
        except Exception:
            pass
    # detect items whose spec mentions more than one function name
    n_names_in_spec = len(NAME_RE.findall(system[m:] if m >= 0 else ""))
    if n_names_in_spec > 1:
        n_multi_spec += 1
    for n in spec_names:
        spec_fn_counts[n] += 1

    # --- chat: first user turn ---
    parts = chat.split("USER:")
    first_user = parts[1].split("ASSISTANT:")[0].strip() if len(parts) > 1 else ""
    user_lens.append(len(first_user))
    chat_lens.append(len(chat))

    # --- called functions ---
    called = []
    for fm in FNCALL.finditer(chat):
        try:
            called.append(json.loads(fm.group(1))["name"])
        except Exception:
            pass
    for n in called:
        called_fn_counts[n] += 1
    if not called:
        n_zero_calls += 1
    if len(set(called)) > 1:
        n_multi_called += 1
    if spec_names and called and spec_names[0] not in called:
        spec_mismatch += 1

print(f"distinct spec functions: {len(spec_fn_counts)}")
print(f"distinct called functions: {len(called_fn_counts)}")
print(f"items with >1 fn name in spec: {n_multi_spec}")
print(f"items with zero function calls (no_tool-style): {n_zero_calls}")
print(f"items with >1 distinct called fn: {n_multi_called}")
print(f"spec/called name mismatch: {spec_mismatch}")
print(f"avg first-user chars: {sum(user_lens)/len(user_lens):.0f} | avg chat chars: {sum(chat_lens)/len(chat_lens):.0f}")
print("top 30 called functions:")
for n, c in called_fn_counts.most_common(30):
    print(f"  {c:6d}  {n}")
json.dump({"spec_fn_counts": dict(spec_fn_counts), "called_fn_counts": dict(called_fn_counts)},
          open(f"{BASE}/data/glaive/fn_counts.json", "w"))
print("saved glaive_fn_counts.json")
