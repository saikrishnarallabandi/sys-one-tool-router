#!/bin/bash
cd /home2/srallaba/projects/system-one-tool-router
VENV=/home2/srallaba/projects/system-one-poc/venv/bin/python
echo "[stage2] waiting for few-shot baseline..."
while pgrep -f "gen_baseline_fewshot\.py" >/dev/null; do sleep 60; done
if [ -f baseline_gen_fewshot_report.json ]; then echo "[stage2] few-shot OK"; else echo "[stage2] FEWSHOT REPORT MISSING"; fi
echo "[stage2] launching OOD generation (ollama/phi4)..."
$VENV ood/generate_ood.py > ood/generate_ood.log 2>&1
echo "[stage2] ood exit=$? examples=$(wc -l < ood/ood_challenge.jsonl 2>/dev/null)"
date > ood/STAGE2_DONE
echo "[stage2] ALL DONE"