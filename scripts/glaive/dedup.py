"""no_tool diversity + exact-dedup vs our train/test data."""
import json
from collections import Counter

BASE = "/home2/srallaba/projects/system-one-tool-router"
rows = json.load(open(f"{BASE}/data/glaive/items.jsonl"))

nontool_texts = Counter()
for r in rows:
    if r["mappable"] and r["gt_classes"] == ["no_tool"]:
        nontool_texts[r["text"]] += 1
print(f"no_tool items: {sum(nontool_texts.values())}, distinct texts: {len(nontool_texts)}")
print("top 10 no_tool user texts:")
for t, c in nontool_texts.most_common(10):
    print(f"  {c:6d}  {t[:100]}")

# exact dedup vs our data
ours = set()
for f in ["data/data_train.jsonl", "data/data_val.jsonl", "data/data_test.jsonl"]:
    for line in open(f"{BASE}/{f}"):
        ours.add(json.loads(line)["text"].strip().lower())
overlap = sum(1 for r in rows if r["mappable"] and r["text"].strip().lower() in ours)
print(f"\nexact text overlap with our train/val/test: {overlap} / {sum(1 for r in rows if r['mappable'])}")
