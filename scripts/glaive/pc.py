import json
rep = json.load(open("/home2/srallaba/projects/system-one-tool-router/exp/glaive/report.json"))
pc = rep["results"]["per_class"]
print("per-class sorted by acc (worst first):")
for k, v in sorted(pc.items(), key=lambda kv: kv[1]["acc"]):
    print(f"  {v['acc']:.3f}  n={v['n']:4d}  correct={v['correct']:4d}  {k}")
