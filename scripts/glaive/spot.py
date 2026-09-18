"""Spot-check label quality: 1 sample per mapped class + 15 no_tool samples."""
import json, random

BASE = "/home2/srallaba/projects/system-one-tool-router"
rows = json.load(open(f"{BASE}/data/glaive/items.jsonl"))
random.seed(11)

by_class = {}
notools = []
for r in rows:
    if not r["mappable"] or r["split"] != "train":
        continue
    if r["gt_classes"] == ["no_tool"]:
        notools.append(r)
    elif len(r["gt_classes"]) == 1:
        by_class.setdefault(r["gt_classes"][0], []).append(r)

print("=== 1 sample per tool class (train) ===")
for c in sorted(by_class):
    r = random.choice(by_class[c])
    print(f"[{c}] called={r['called_names']}")
    print(f"  USER: {r['text'][:160].replace(chr(10),' ')}")

print("\n=== 15 no_tool samples (train) ===")
for r in random.sample(notools, 15):
    print(f"spec={r['spec_names'][:2]}")
    print(f"  USER: {r['text'][:150].replace(chr(10),' ')}")
