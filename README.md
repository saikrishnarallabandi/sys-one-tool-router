# System One POC #2 — tool-call routing with Qwen3-0.6B

Second proof-of-concept in the "System One" series (see `../system-one-poc/`):
a small model that emits **typed probabilistic routing decisions** (which tool to
call + calibrated confidence) via a **single forward pass — no token generation**.
This is the "fast router that scores candidate tools with calibrated confidence"
pattern: one forward pass, ~29 ms, instead of an LLM-as-judge tool-selection call.

## Task

**Multi-class tool routing.** Input = a user utterance; output = which of 25
classes (24 canonical tools + `no_tool`) should handle it, as typed JSON:

```json
{"tool": "calculate_tip",
 "probabilities": {"calculate_tip": 0.93, "calculate_discount": 0.04, ...},
 "confidence": 0.93, "latency_ms": 28.4}
```

## Data — real, open-source (not template-synthesized by us)

- Source: `glaiveai/glaive-function-calling-v2` (Hugging Face, apache-2.0,
  112,960 rows). Chosen over `Salesforce/xlam-function-calling-60k` because
  Glaive's v2 split contains natural no-call turns (user asks for something the
  assistant can't do -> refusal), which become the `no_tool` class.
- Extraction (`make_data.py`): split each chat into turns; every ASSISTANT turn
  containing `<functioncall>` yields (the immediately preceding USER utterance,
  tool name). Rows with no functioncall yield (last USER utterance, `no_tool`).
  Multi-turn chats can yield several examples (e.g. "news for the US?" then
  "what about France?" -> two `get_news` examples).
- Normalization: 949 distinct raw tool names (heavy synonym tail:
  `get_news`/`get_news_headlines`/`search_news`, `generate_password`/
  `generate_random_password`, `create_event`/`schedule_meeting`, …) merged via
  a synonym map; kept the top 24 canonical tools + `no_tool` = **25 classes**.
- Dedup: exact (text, label) dedup — Glaive v2 is heavily templated, so this
  collapsed e.g. `calculate_bmi` 3103 occurrences -> 171 unique utterances.
  Without it, train/test would leak identical strings. 128 texts mapped to
  multiple tools (generic clarifications); kept the majority label.
- Final: 10,936 examples, stratified 70/15/15 -> train 7,655 / val 1,640 /
  test 1,641. Per-class: 37–700 tool examples, `no_tool` 2,000.

## Model & training

- Qwen3-0.6B + classification head, LoRA r=16 on all attention+MLP projections,
  score head fully trained (~10.1M trainable params / 1.67%).
- Same stability recipe as POC1: fp32 frozen backbone + fp32 adapters, fp16
  autocast for compute (full fp16 diverged on Pascal); single GPU
  (CUDA_VISIBLE_DEVICES=0; ollama lives on GPU0, jataayu serving on GPU1).
- 3 epochs, batch 16, lr 2e-4 cosine, warmup 100, early stopping on macro-F1.
- **Class-weighted loss** (inverse frequency): the tail is long
  (`generate_random_number`: 37 train examples vs `no_tool`: 1400).
- Adapters merged into a plain HF checkpoint: `checkpoints/qwen3-0.6b-tool-router/best/`.

## Results (GTX 1080 Ti, single GPU)

Test set n=1,641, 25 classes. Temperature T=1.7912 fit on the validation split
(n=1,640) — **not** on the test set.

| Metric | Test, uncalibrated (T=1.0) | Test, scaled (T=1.791) |
|---|---:|---:|
| Accuracy | 0.9927 | 0.9927 |
| Macro-F1 | 0.9894 | 0.9894 |
| NLL | 0.0310 | 0.0261 |
| Brier | 0.0131 | 0.0120 |
| ECE (15 bins) | 0.0067 | 0.0077 |

Calibration story (the POC1 lesson applied): unlike POC1's binary task, where
the easy val split gave T=1.00 and hid overconfidence, this 25-class real-data
task produced **T=1.79 — genuine overconfidence detected on held-out val**.
Temperature scaling cut NLL 0.0310 → 0.0261 and Brier 0.0131 → 0.0120 on test.
ECE ticked *up* slightly (0.0067 → 0.0077): with ECE already near zero, that is
noise — but it is an honest reminder that temperature improves likelihood, not
necessarily every calibration statistic.

Per-class error analysis (test, calibrated):
- Weakest: `calculate_mortgage_payment` F1=0.933 (n=32), `get_movie_details`
  F1=0.964, `translate_text` F1=0.966, `calculate_loan_payment` F1=0.967,
  `get_definition` F1=0.970 — the small/near-duplicate classes.
- Top confusions are **dataset-level tool duplicates**, not model failures:
  `calculate_mortgage_payment` → `calculate_loan_payment` (x4),
  `search_movies` → `get_movie_details` (x2). These tool pairs are near-synonyms
  in the source data; a production router would merge them into one canonical
  tool. `translate_text` → `no_tool` (x1) and two `no_tool` → tool misses are
  the genuinely interesting boundary cases.

Latency (fp16, incl. tokenization, single GTX 1080 Ti):
- Single sample: **mean 29.8 ms, p50 29.3 ms, p99 47.9 ms**
- Throughput: **175 samples/sec** (batch 16), **190 samples/sec** (batch 64)
- For comparison: an LLM-as-judge tool-selection call is typically 2–30 s.
  ~100x faster, typed output, calibrated confidence.

## How to run

```bash
pip install -r requirements.txt
python make_data.py   # build datasets from glaiveai/glaive-function-calling-v2
python train.py       # train Qwen3-0.6B + LoRA (~20 min on a GTX 1080 Ti)
python eval.py        # metrics + temperature scaling + per-class analysis
python infer.py "What is 20% tip on a $85 bill?"
echo "Send an email to Priya" | python infer.py
python bench_latency.py
```

`checkpoints/` (trained weights, ~2.7 GB) is git-ignored; run `train.py` to
reproduce. The committed JSONL/JSON files are the exact datasets, label map,
and eval artifacts from the reference run described above.

## Honest limitations

1. **Templated source data.** Glaive v2 utterances are synthetic and templated;
   dedup removed exact repeats, but paraphrase-level template artifacts remain.
   The model partly learns "which template family is this" rather than deep
   intent. Real Jataayu/jetsons traffic with teacher labels would be strictly
   better.
2. **Utterance-only input.** The router sees only the final user utterance, no
   conversation history and no tool descriptions. Follow-ups like "what about
   France?" are learnable but genuinely ambiguous without context.
3. **Small tail classes.** `generate_random_number` (37 train), `translate_text`
   (69), `get_definition` (74) have few unique utterances; expect weak F1 there.
4. **One decision per forward pass.** Like POC1, the "parallel joint decisions"
   part of the Jev claim is not replicated.
5. **No teacher labeling / human review.** Labels come from the dataset's own
   functioncall annotations, taken at face value.
