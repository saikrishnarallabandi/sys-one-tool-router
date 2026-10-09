# typed-decisions delta post-training — run log

## Goal
RL post-training (no SFT) for calibrated typed decisions: kev-4b + soft gold
distributions + Brier/RPS calibration terms. Primitives kept: choice/score/noul.

## Data (local, checksummed both sides)
- Converter: `scripts/kev_td_data.py` (label + soft `target` per question)
- `.cache/decisionbench/typed-decisions-kev-train.jsonl`: 1200 recs, md5 ccbc192ce11e7e6dfea7797084a9a4af
- `.cache/decisionbench/typed-decisions-kev-test.jsonl`: 400 recs, md5 f36733fb87d1716e8d52dee2bfa8eca7
- Train stats: 300 cases × 4 workflows, 5 Qs each; 5977/6000 non-degenerate gold
  dists (mean entropy 0.75 nats); rare labels: action=stop (3), execute_refund (9),
  disposition=reject (13), outcome=failure (15).

## Baselines (results/smoke2-n40, full 400-case test, 2000 decisions)
| model | dec-acc | KL | ECE |
|---|---|---|---|
| prior (train freqs) | 0.484 | — | — |
| laya-english | 0.363 | 0.570 | 0.174 |
| nli-zeroshot | 0.532 | 0.398 | 0.085 |
| laya-typed-decisions (specialist) | 0.767 | 0.117 | 0.214 |

## Env image (replication deliverable)
- `docker/Dockerfile` → built on judith → `srallaba/kev-train:cu128-py313`
- Digest: `sha256:1581329196f5a38aacfd915f66dca108aee7d78a90869a83213d159e5d2937d1`
- Use on Vast: launch instance with image `srallaba/kev-train:cu128-py313`.

## Box 1 (DEAD — wedged, destroyed per Vast gotchas)
- 52642713, RTX 3090 $0.153/hr. Setup: CUDA 12.8 image + uv + py3.13 + kev clone.
- Smoke `runs/smoke-vast` passed (loss 1.98 → 1.48).
- Delta attempt: OOM at batch 4 → batch 1/accum 8 OOM → fixed with
  `--checkpointing 1 --perm_frac 0`. Reached ep0 step ~220, loss ~0.80.
- Lost: uncompleted run dir (nothing rsynced; LoRA saves at end only).
- Lesson: `--save_every_steps` is full-weight-only; LoRA runs have no mid-run
  rescue. Reproduce-from-scratch is the recovery path (all inputs versioned).

## Box 2 (TRAINED — eval running)
- Training finished: 290 optimizer steps, 2316 records seen (1158 × 2 epochs),
  grad-norm mean 2.14 (ep0) → 1.06 (ep1), peak 19.9 GB. Weights rescued to
  `/tmp/td-delta-rescue/` (adapter 130 MB + head.pt + tokenizer).
- One transient SSH blip post-save (box unreachable ~10 min, API said running);
  recovered on its own — NOT destroyed (contrast box 1: API `offline` + clean
  connection-refused × 3).

## Box 2 (ACTIVE)
- 52664976, RTX 3090 $0.149/hr, ssh6.vast.ai:24976, key ~/.ssh/vast_ed25519.
- kev @ ce089c6, torch 2.8.0+cu128, transformers 5.17.0.
- Command:
  `kev.train --data /root/typed-decisions-kev-train.jsonl --init_from jaredpalmer/kev-4b
   --base Qwen/Qwen3.5-4B-Base --epochs 2 --lr 2e-5 --batch 1 --accum 8
   --checkpointing 1 --perm_frac 0 --brier_w 0.5 --ord_w 0.5 --device cuda
   --out runs/td-delta` (+ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
- Start-of-train loss 1.201 (identical to box 1 — deterministic setup).

## Eval results (kev benchmark, 400 held-out cases / 1945 clean decisions)
| metric | delta (td-delta) | base (kev-4b) |
|---|---|---|
| acc | **0.800** | 0.665 |
| ece | 0.162 | **0.091** |
| brier | **0.327** | 0.468 |
| nll | **0.576** | 0.819 |
| aurc | **0.071** | 0.195 |
| score_mae | **0.390** | 0.603 |
| RPS | **0.066** | 0.117 |
| coverage@0.9 | **0.130** | 0.059 |
| accuracy@0.9 | **1.000** | 0.983 |
| choice acc (ece) | 0.756 (0.149) | 0.640 (0.104) |
| noul acc (ece) | 0.861 (0.107) | 0.762 (0.111) |
| score acc (ece) | 0.787 (0.212) | 0.611 (0.074) |

Reading: post-training wins accuracy, Brier, NLL, AURC, selective
prediction everywhere — but ECE regresses overall, driven by score
questions (0.212 vs 0.074). Same shape as the laya specialist tradeoff:
optimizing proper scores sharpened distributions without fixing ECE.

Independent verification (local, from rows.json + gold):
- KL(gold||pred): delta 0.079 vs base 0.226 (3x closer to gold).
- Per workflow acc — delta: agent_trace 0.754, customer_service 0.822,
  invoice 0.840, security 0.786; base: 0.568 / 0.706 / 0.692 / 0.700.
  Gains are broad, not one-workflow luck.
- Per primitive KL — choice 0.109/0.247, noul 0.042/0.180, score 0.085/0.245.

Verdict: post-training works for accuracy + proper scores + selective
prediction; score-level ECE is the open problem (next: higher ord_w,
temperature fit, or score-specific Brier).

Score-ECE diagnosis (local reliability, /tmp/score_ece.py): the delta is
UNDERconfident on score, not overconfident — e.g. conf~0.6 bin has acc 0.91
(0.32 gap), conf~0.7 bin acc 0.94. Base tracks the diagonal (0.72@0.59).
Cause: training against soft gold distributions spreads mass (KL 0.085),
so argmax confidence understates argmax accuracy. ECE punishes the gap
symmetrically. Fix direction: temperature < 1 (sharpen) fit on held-out —
cheap, no retraining. Middle levels (L1/L2) carry most of the error mass.

## Eval plan (decisionbench scoring)
1. `kev.benchmark --run runs/td-delta --data /root/typed-decisions-kev-test.jsonl --out runs/td-eval`
2. Same with `--run jaredpalmer/kev-4b` → paired baseline.
3. Pull rows back, score via `decisionbench` (per-primitive ECE, KL, robustness).
4. Gate: accuracy near specialist (0.767) with ECE near NLI (0.085).

## Box 3 — 9B replication (ACTIVE)
- 52757934, RTX 3090 CA $0.149/hr, ssh5.vast.ai:37934. (A6000 52755349 wedged in
  `loading` 15+ min → destroyed per gotchas; 3090 fallback offer 43861061.)
- Model decision: `Qwen/Qwen3.5-9B-Base` (Feb 2026). No Qwen4-9B-Base exists, no
  Gemma-3-9B base — nothing better in the 9B class. `jaredpalmer/kev-9b` on HF is
  NOT our replication (provenance: trained on `dates_unknowable.jsonl`), so training
  fresh from base, no --init_from.
- Data: scp'd local `.cache` files; train md5 ccbc192c… (1200 recs, matches 4B run).
- Command (mirrors 4B box-2 recipe + bf16 weights for 24GB fit):
  `kev.train --data /root/typed-decisions-kev-train.jsonl --base Qwen/Qwen3.5-9B-Base
   --epochs 2 --lr 2e-5 --batch 1 --accum 8 --checkpointing 1 --perm_frac 0
   --brier_w 0.5 --ord_w 0.5 --device cuda --dtype bf16 --weights_dtype bf16
   --out runs/td-delta-9b` (+ expandable_segments), tmux `td9b`, log /root/td-delta-9b.log.
- Eval plan (same as 4B): `kev.benchmark --run runs/td-delta-9b --data
  /root/typed-decisions-kev-test.jsonl --out runs/td-eval-9b` + base baseline.

## Box 3 — 9B training live (update)
- HF Hub/XET downloads throttled to ~0 (≈0.3 MB/s) from this host; fixed with
  aria2c -x16 direct shard fetch (~50 MB/s) to /root/qwen9b, verified 775/775
  tensors via safetensors headers, training with `--base /root/qwen9b`.
  (Also: hf_transfer deprecated for XET repos; hf-xet installed but unnecessary
  once aria2c route worked. HF token relayed judith→box via pipe, stored
  /root/.hf_token mode 600.)
- Start: loss 1.216 (4B was 1.201), 1158 recs, 290 steps/ep, 2.33 s/rec →
  ~23 min for 2 epochs. 45.4M trainable params, 18.8GB VRAM.

## Box 3 — 9B results (DONE, box destroyed)
- Train: loss 1.216 → 0.824, 2 epochs, ~75 min. Adapter saved to results/td-delta-9b/.
- Held-out eval (400 cases / 1945 clean decisions), 9B delta vs 4B delta vs 4B base:
  acc 0.774 / 0.800 / 0.665; ece 0.149 / 0.162 / 0.091; brier 0.350 / 0.327 / 0.468;
  nll 0.612 / 0.576 / 0.819; aurc 0.087 / 0.071 / 0.195; score_mae 0.416 / 0.390 / 0.603;
  RPS 0.070 / 0.066 / 0.117; cov@0.9 0.120 / 0.130 / 0.059; acc@0.9 1.000 / 1.000 / 0.983.
  Per primitive acc (ece): choice 0.715 (0.129), noul 0.847 (0.104), score 0.763 (0.199).
- Independent local verification (from rows.json + gold): KL choice 0.133, noul 0.048,
  score 0.099 (all far below 4B-base 0.247/0.180/0.245, near 4B-delta 0.109/0.042/0.085).
  Reliability: underconfident in every bin (same shape as 4B).
- Temperature on hard-label ECE: all T=0.45 → 0.020; score T=0.40 → 0.027;
  choice T=0.55 → 0.052; noul T=0.40 → 0.021. Replicates the 4B T*=0.35 → 0.031 finding.
- Caveat: 9B trained cold from base (no trusted 9B adapter to warm-start, unlike 4B
  from kev-4b); single runs each, so the 0.774-vs-0.800 gap is within run noise +
  warm-start difference, not evidence against scaling.

## Sweep — all kev variants, same held-out set (A6000 52819959, DONE)
- kev05 0.455/0.114/0.680/1.257; kev06 0.447/0.227/0.727/1.362; kev08 0.458/0.047/0.608/1.065;
  kev4b 0.665/0.091/0.467/0.819; td-delta-4b 0.800/0.162/0.327/0.576;
  td-delta-9b 0.774/0.149/0.349/0.612 (reproduced on 2nd box, same 0.774);
  kev8b 0.638/0.112/0.513/0.908; kev9b-hf 0.718/0.091/0.414/0.738 (acc/ece/brier/nll).
  n=1945 (kev05/06/8b: 1955, fewer rejects). Reports: results/sweep-8way/sweep/.
- Surprises: kev8b < kev4b on acc (0.638 vs 0.665); sub-1B cluster ~0.45.
- jevlike (vinnylarouge, 1307 stars) is an independent option-attention implementation
  with game checkpoints only — not kev.benchmark-compatible, not scored.
- Ops note: td-delta-9b needed head.pt Meta.base repointed from /root/qwen9b to
  Qwen/Qwen3.5-9B-Base@68c46c4b (box copy only; local results/ copy keeps faithful path).

## jevlike benchmark (Astra CPU/MPS, DONE)
- jevlike's released checkpoint (joint-imitation.pt) is a vision/game weights file,
  incompatible with text eval — not scored. Instead trained their TinyScorer
  (byte-level, width 64/rank 64) on our train conversion (6000 ex, hard labels)
  via `jevlike.train` (8 epochs, MPS, ~2 min, best val NLL 1.016) → runs/jevlike-td.pt.
- Test (2000 ex): top1 0.511, ECE 0.037. Per type: choice 0.407, noul 0.673, score 0.466.
- Shuffled-context control: 0.494 top1 (vs 0.511) — the scorer barely uses the state,
  mostly option priors. Converter: scripts/jevlike_td_data.py.

## v2 dataset (DONE, local)
- Builder: scripts/build_td_v2.py (seed 0). TRAIN 13,423 recs (md5 af9a9c1e…),
  EVAL 6,279 (md5 1a88210a…): .cache/decisionbench/typed-decisions-v2[-eval].jsonl.
- Train: v1 1200 + helpsteer2 1760 (score) + ultrafeedback-binarized 1760 (choice) +
  hatespeech 2639 grouped-soft (score) + mmlu-aux 1760 (choice) + hellaswag 1760 (choice) +
  anli-r1 1760 (choice) + pubmedqa 784 (noul). Types: choice 8840 / score 6799 / noul 2584.
- Eval: per-source 12% held-out + chaosnli snli/mnli_m 3113 soft (eval-only) +
  rewardbench 1500 (eval-only). Types: choice 5573 / score 600 / noul 106.
- Leakage: normalized exact-match audit vs v1-test (400) → 0 leaks; 1 in-train dup dropped.
  MMLU from auxiliary_train, HellaSwag/ANLI/UltraFeedback/HelpSteer2 from train splits;
  ChaosNLI + RewardBench kept eval-only (upstream eval sets).
- Dropped: jigsaw (bad ID), hendrycks/ethics + fever + moral_stories (dataset scripts
  unsupported in datasets 5.x), Qwen-side: ethics covered by noul/score elsewhere.

## Box 4 — delta-v2 9B (ACTIVE)
- 52882272, A6000 Virginia $0.482/hr, ssh4.vast.ai:12272.
- Warm start --init_from /root/td-delta-9b (mirrors 4B recipe), base pinned 68c46c4b,
  same flags, --out runs/td-delta-v2-9b. Base pulled from hub in 59s (no throttle here).
- Data md5s match local. Adapter base refs patched on box (hub id + pin).

## Box 4 — delta-v2 9B results (DONE)
- Train: warm start td-delta-9b, loss 0.867 → 0.535, 2 epochs × 3346 steps (~20h A6000).
  Adapter + reports: results/td-delta-v2-9b/.
- v1-test (comparability): acc 0.783 (v1-delta 0.774), ece 0.140, brier 0.334, nll 0.585,
  aurc 0.079. Per primitive: choice 0.737 (was 0.715), noul 0.840 (0.847), score 0.774 (0.763).
- v2-eval (6264 clean): acc 0.786, ece 0.107, brier 0.331. Per task: choice 0.801 (5558),
  noul 0.783 (106, noisy), score 0.647 (600, harder 5-pt/3-pt scales).
- Reading: v2 data gives small broad in-domain gains; external choice strong at 0.80.

## Docker image refreshed (DONE)
- srallaba/kev-train:cu128-py313 rebuilt on judith + pushed
  (digest sha256:6ac54f47…). Now bakes in aria2, hf-xet, hf_transfer.
  Dockerfile updated in repo; runtime convention (HF_TOKEN file, scp data) unchanged.

## Test studio (LIVE on Astra)
- Runner: `.venv/bin/python ui/server.py --run results/td-delta-v2-9b --port 8010`
  → http://127.0.0.1:8010/ (kev.serve torch/MPS/bf16 + static UI, one process).
- UI: 6 bundled examples (2/type from v1-test) + custom state JSON + choice/noul/score
  builders, probability bars + raw JSON. Verified: /v1/models OK, live choice
  inference OK (human_review 0.540, 4.7s cold). Browser rendering unverified (no chromium).
- Serving required: peft+safetensors installed; v2 adapter base refs → hub id + pin 68c46c4b;
  Qwen3.5-9B-Base snapshot cached locally (~19GB).
