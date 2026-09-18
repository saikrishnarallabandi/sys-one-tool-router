"""Characterize the exact-overlap between Glaive-mapped items and our data."""
import json
from collections import Counter

BASE = "/home2/srallaba/projects/system-one-tool-router"
rows = json.load(open(f"{BASE}/data/glaive/items.jsonl"))

ours = {}  # text -> (file, label)
for f in ["data/data_train.jsonl", "data/data_val.jsonl", "data/data_test.jsonl"]:
    for line in open(f"{BASE}/{f}"):
        d = json.loads(line)
        ours[d["text"].strip().lower()] = (f, d["label"])

overlap_by_class = Counter()
overlap_by_file = Counter()
examples = []
for r in rows:
    if not r["mappable"]:
        continue
    key = r["text"].strip().lower()
    if key in ours:
        f, lab = ours[key]
        overlap_by_file[f] += 1
        for c in r["gt_classes"]:
            overlap_by_class[c] += 1
        if len(examples) < 8:
            examples.append((r["gt_classes"], lab, r["text"][:120]))

print("overlap by our file:", dict(overlap_by_file))
print("overlap by glaive class (top 12):")
for c, n in overlap_by_class.most_common(12):
    print(f"  {n:6d}  {c}")
print("examples (glaive gt -> our label):")
for gt, lab, t in examples:
    print(f"  {gt} -> {lab}: {t}")
