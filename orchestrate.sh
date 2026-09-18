#!/bin/bash
# Sys One paper pipeline: zero-shot baseline -> few-shot baseline -> OOD generation.
# Each stage writes its own log + report; this script just sequences them.
set -u
cd /home2/srallaba/projects/system-one-tool-router
VENV=/home2/srallaba/projects/system-one-poc/venv/bin/python

echo "[pipeline] waiting for zero-shot baseline (pid via pgrep)..."
while pgrep -f "gen_baseline\.py" >/dev/null; do sleep 60; done
if [ ! -f baseline_gen_report.json ]; then
  echo "[pipeline] ZERO-SHOT FAILED: baseline_gen_report.json missing"; exit 1
fi
echo "[pipeline] zero-shot done: $(python3 -c "import json;print(json.load(open('baseline_gen_report.json'))['accuracy'])")"

echo "[pipeline] launching few-shot baseline on GPU0..."
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 $VENV gen_baseline_fewshot.py > gen_baseline_fewshot.log 2>&1
if [ ! -f baseline_gen_fewshot_report.json ]; then
  echo "[pipeline] FEWSHOT FAILED: report missing"; tail -5 gen_baseline_fewshot.log; exit 1
fi
echo "[pipeline] few-shot done"

echo "[pipeline] launching OOD generation via ollama/phi4..."
$VENV ood/generate_ood.py > ood/generate_ood.log 2>&1
if [ ! -f ood/ood_challenge.jsonl ]; then
  echo "[pipeline] OOD FAILED: challenge file missing"; tail -5 ood/generate_ood.log; exit 1
fi
echo "[pipeline] OOD done: $(wc -l < ood/ood_challenge.jsonl) examples"
echo "[pipeline] ALL STAGES COMPLETE"
