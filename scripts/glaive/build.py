"""Build glaive_items.jsonl: parse spec/called functions, first user turn, split.
Writes per-item rows with mappable flag (mapping applied later by glaive_map.py).
Also dumps full called-function list for mapping review.
"""
import json, re, hashlib
from collections import Counter

BASE = "/home2/srallaba/projects/system-one-tool-router"
items = json.load(open(f"{BASE}/data/glaive/raw/glaive-function-calling-v2.json"))

NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
# function name right after <functioncall>, tolerant of multiline JSON
CALL_RE = re.compile(r'<functioncall>\s*\{\s*"name"\s*:\s*"([^"]+)"')

called_counts = Counter()
spec_counts = Counter()
rows = []
n_multi_spec = 0
n_nospec_call = 0
n_nospec_nocall = 0
n_spec_nocall = 0
n_multicall = 0

for idx, it in enumerate(items):
    system, chat = it["system"], it["chat"]
    m = system.find("{")
    # all top-level function specs: find each '"name"' that belongs to a spec object
    # (specs are concatenated JSON objects; parameter dicts rarely contain "name")
    spec_names = []
    if m >= 0:
        # parse successive JSON objects from the system string
        pos = m
        dec = json.JSONDecoder()
        while pos < len(system):
            # skip to next '{'
            nxt = system.find("{", pos)
            if nxt < 0:
                break
            try:
                obj, end = dec.raw_decode(system[nxt:])
                if isinstance(obj, dict) and "name" in obj and "description" in obj:
                    spec_names.append(obj["name"])
                pos = nxt + end
            except Exception:
                pos = nxt + 1
            if len(spec_names) > 4:
                break
    for n in spec_names:
        spec_counts[n] += 1
    if len(spec_names) > 1:
        n_multi_spec += 1

    called = CALL_RE.findall(chat)
    for n in called:
        called_counts[n] += 1
    if len(set(called)) > 1:
        n_multicall += 1

    parts = chat.split("USER:")
    first_user = parts[1].split("ASSISTANT:")[0].strip() if len(parts) > 1 else ""

    has_spec = len(spec_names) > 0
    has_call = len(called) > 0
    if not has_spec and has_call:
        n_nospec_call += 1
    if not has_spec and not has_call:
        n_nospec_nocall += 1
    if has_spec and not has_call:
        n_spec_nocall += 1

    # constructed split: hash-based 5% test (no official split exists)
    h = int(hashlib.md5(str(idx).encode()).hexdigest(), 16)
    split = "test" if h % 20 == 0 else "train"

    rows.append({
        "id": f"glaive-{idx}",
        "split": split,
        "text": first_user,
        "spec_names": spec_names,
        "called_names": called,
        "has_spec": has_spec,
    })

json.dump(rows, open(f"{BASE}/data/glaive/items.jsonl", "w"))
print(f"n={len(rows)} test={[r['split'] for r in rows].count('test')} "
      f"train={[r['split'] for r in rows].count('train')}")
print(f"distinct spec fns: {len(spec_counts)} | distinct called fns: {len(called_counts)}")
print(f"multi-spec items: {n_multi_spec} | multi-distinct-call items: {n_multicall}")
print(f"no-spec+call: {n_nospec_call} | no-spec+no-call: {n_nospec_nocall} | spec+no-call: {n_spec_nocall}")
print("called fn counts (all):")
for n, c in called_counts.most_common():
    print(f"  {c:6d}  {n}")
json.dump({"spec": dict(spec_counts), "called": dict(called_counts)},
          open(f"{BASE}/data/glaive/fn_counts.json", "w"))
