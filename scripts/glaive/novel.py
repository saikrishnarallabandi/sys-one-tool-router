"""Count novel (not in our train/val/test) mapped items per split/class."""
import json
from collections import Counter

BASE = "/home2/srallaba/projects/system-one-tool-router"
rows = json.load(open(f"{BASE}/data/glaive/items.jsonl"))

ours = set()
for f in ["data/data_train.jsonl", "data/data_val.jsonl", "data/data_test.jsonl"]:
    for line in open(f"{BASE}/{f}"):
        ours.add(json.loads(line)["text"].strip().lower())

novel = Counter()
novel_test_class = Counter()
for r in rows:
    if not r["mappable"]:
        continue
    if r["text"].strip().lower() not in ours:
        novel[r["split"]] += 1
        if r["split"] == "test":
            for c in r["gt_classes"]:
                novel_test_class[c] += 1
print("novel mapped items by split:", dict(novel))
print("novel TEST items by class:")
for c, n in novel_test_class.most_common():
    print(f"  {n:5d}  {c}")
