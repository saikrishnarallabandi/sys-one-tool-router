import json
BASE = "/home2/srallaba/projects/system-one-tool-router"
preds = json.load(open(f"{BASE}/exp/glaive/preds.json"))
rows = {r["id"]: r for r in json.load(open(f"{BASE}/data/glaive/items.jsonl"))}
for cls in ["calculate_loan_payment", "create_calendar_event"]:
    print(f"=== errors in {cls} ===")
    n = 0
    for p in preds:
        if p["gt_classes"] == [cls] and p["top1"] != cls and n < 6:
            r = rows[p["id"]]
            print(f"called={r['called_names']} pred={p['top1']} conf={p['conf']}")
            print(f"  USER: {r['text'][:130]}")
            n += 1
