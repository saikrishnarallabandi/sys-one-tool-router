"""Train-split assessment: novel per-class counts + unmapped-function pool sizes."""
import json
from collections import Counter

BASE = "/home2/srallaba/projects/system-one-tool-router"
rows = json.load(open(f"{BASE}/data/glaive/items.jsonl"))

ours = set()
for f in ["data/data_train.jsonl", "data/data_val.jsonl", "data/data_test.jsonl"]:
    for line in open(f"{BASE}/{f}"):
        ours.add(json.loads(line)["text"].strip().lower())

novel_train_class = Counter()
novel_train_total = 0
unmapped_train_fn = Counter()
unmapped_train_total = 0
for r in rows:
    if r["split"] != "train":
        continue
    if r["text"].strip().lower() in ours:
        continue
    if r["mappable"]:
        novel_train_total += 1
        for c in r["gt_classes"]:
            novel_train_class[c] += 1
    elif r["called_names"]:
        unmapped_train_total += 1
        for c in set(c.lower() for c in r["called_names"]):
            unmapped_train_fn[c] += 1

print(f"novel mapped TRAIN items (unique texts): {novel_train_total}")
print("per class:")
for c, n in novel_train_class.most_common():
    print(f"  {n:6d}  {c}")
print(f"\nunmapped-call TRAIN items (catalog-expansion pool): {unmapped_train_total}")
print(f"distinct unmapped functions in train: {len(unmapped_train_fn)}")
print("top 25 unmapped functions by train items:")
for n, c in unmapped_train_fn.most_common(25):
    print(f"  {c:6d}  {n}")
