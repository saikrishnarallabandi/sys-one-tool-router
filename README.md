# Tool-call routing with Qwen3-0.6B — one forward pass, no token generation

When an AI assistant needs to decide **which tool to call** for a user request,
the default approach is to ask a large LLM to choose ("LLM-as-judge"): slow
(typically 2–30 s), expensive, and free-text output you have to parse.

This repo demonstrates a different pattern: a **small model that emits typed,
probabilistic routing decisions** — which tool, per-class probabilities, and
calibrated confidence — in a **single forward pass (~30 ms), with no token
generation**. Roughly 100x faster than LLM-as-judge, with structured output
and honest uncertainty.

Input is a user utterance; output is typed JSON:

```json
{"tool": "calculate_tip",
 "probabilities": {"calculate_tip": 0.93, "calculate_discount": 0.04, "...": "..."},
 "confidence": 0.93, "latency_ms": 28.4}
```

This is the second in an informal "Sys One" series of experiments: small,
fast, calibrated models that make one structured decision per forward pass
(the first was a binary prompt-injection detector). Each repo stands alone —
you don't need the first to use this one.

## The task

**Multi-class tool routing over 25 classes**: 24 canonical tools
(calculators, messaging, search, scheduling, …) plus `no_tool` for "no tool
applies" (chit-chat, out-of-scope questions). The router sees only the final
user utterance — no conversation history, no tool descriptions.

## Data — real, open-source (not template-synthesized by us)

- Source: [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
  (Hugging Face, Apache-2.0, 112,960 rows). Chosen over
  `Salesforce/xlam-function-calling-60k` because Glaive's v2 split contains
  natural no-call turns (user asks for something the assistant can't do →
  refusal), which become the `no_tool` class.
- Extraction (`make_data.py`): split each chat into turns; every ASSISTANT turn
  containing `<functioncall>` yields (the immediately preceding USER utterance,
  tool name). Rows with no functioncall yield (last USER utterance, `no_tool`).
  Multi-turn chats can yield several examples.
- Normalization: 949 distinct raw tool names (heavy synonym tail, e.g.
  `get_news`/`get_news_headlines`/`search_news`,
  `generate_password`/`generate_random_password`,
  `create_event`/`schedule_meeting`) merged via a synonym map into the top 24
  canonical tools + `no_tool` = **25 classes**.
- Dedup: exact (text, label) dedup — Glaive v2 is heavily templated, so this
  collapsed e.g. `calculate_bmi` from 3,103 occurrences to 171 unique
  utterances. Without it, train/test would leak identical strings. 128 texts
  mapped to multiple tools (generic clarifications); kept the majority label.
- Final: **10,936 examples**, stratified 70/15/15 → train 7,655 / val 1,640 /
  test 1,641. Per-class: 37–700 tool examples, `no_tool` 2,000.

## Model & training

- **Qwen3-0.6B** + sequence-classification head, LoRA r=16 on all
  attention+MLP projections, score head fully trained (~10.1M trainable params /
  1.67%).
- Stability recipe for older GPUs: fp32 frozen backbone + fp32 adapters, fp16
  autocast for compute (full-fp16 training diverged on Pascal); single GPU.
- 3 epochs, batch 16, lr 2e-4 cosine, warmup 100, early stopping on macro-F1.
- **Class-weighted loss** (inverse frequency) for the long tail
  (`generate_random_number`: 37 train examples vs `no_tool`: 1,400).

## Results

Test set n=1,641, 25 classes. Temperature T=1.7912 fit on the **validation**
split (n=1,640) — never on the test set.

| Metric | Test, uncalibrated (T=1.0) | Test, scaled (T=1.791) |
|---|---:|---:|
| Accuracy | 0.9927 | 0.9927 |
| Macro-F1 | 0.9894 | 0.9894 |
| NLL | 0.0310 | **0.0261** |
| Brier | 0.0131 | **0.0120** |
| ECE (15 bins) | 0.0067 | 0.0077 |

**Calibration story.** An earlier binary experiment on easy data produced
T=1.00 — temperature looked perfect because the validation set was too easy,
hiding real overconfidence. This 25-class task on real data produced
**T=1.79: genuine overconfidence, detected on held-out validation**.
Temperature scaling cut test NLL 0.0310 → 0.0261 and Brier 0.0131 → 0.0120.
ECE ticked *up* slightly (0.0067 → 0.0077); with ECE already near zero that is
noise — an honest reminder that temperature improves likelihood, not
necessarily every calibration statistic. Lesson: calibrate against
representative data, or not at all.

**Per-class error analysis** (test, calibrated):
- Weakest: `calculate_mortgage_payment` F1=0.933 (n=32), `get_movie_details`
  F1=0.964, `translate_text` F1=0.966 — the small/near-duplicate classes.
- Top confusions are **dataset-level tool duplicates**, not model failures:
  `calculate_mortgage_payment` → `calculate_loan_payment` (x4),
  `search_movies` → `get_movie_details` (x2). These pairs are near-synonyms in
  the source data; a production router would merge them into one canonical
  tool.

**Latency** (fp16 incl. tokenization, single GTX 1080 Ti):
- Single sample: **mean 29.8 ms, p50 29.3 ms, p99 47.9 ms**
- Throughput: **175 samples/sec** (batch 16), **190 samples/sec** (batch 64)

## Repo contents

| Path | What it is |
|---|---|
| `scripts/` | Pipeline: data build, training, eval, generative baselines, inference, latency bench |
| `scripts/bfcl/` | BFCL mapping + frozen-transfer eval |
| `scripts/glaive/` | Glaive v2 audit tools (mapping, novel-text held-out eval) |
| `scripts/ood/` | OOD challenge generation + eval |
| `data/` | Curated datasets (`data_train/val/test.jsonl`), label map, stats |
| `data/bfcl/` | BFCL items + raw dump |
| `data/glaive/` | Glaive audit items + function counts (raw 259M dump git-ignored) |
| `exp/` | Run artifacts: eval reports, temperature, baseline reports |
| `exp/bfcl/`, `exp/glaive/`, `exp/ood/` | Per-experiment reports + predictions |
| `docs/` | Design notes |
| `paper/` | Paper source + PDF |
| `requirements.txt` | Pinned dependencies |

All scripts run from the repo root, e.g. `python scripts/train.py`.

## Requirements

- Python 3.10+
- A CUDA GPU is recommended (reference numbers measured on a single GTX 1080
  Ti, 11 GB). CPU works but is much slower.

## How to run

```bash
pip install -r requirements.txt
python scripts/make_data.py   # build datasets from glaiveai/glaive-function-calling-v2
python scripts/train.py       # train Qwen3-0.6B + LoRA (~20 min on a GTX 1080 Ti)
python scripts/eval.py        # metrics + temperature scaling + per-class analysis
python scripts/infer.py "What is 20% tip on a $85 bill?"
echo "Send an email to Priya" | python scripts/infer.py
python scripts/bench_latency.py
```

`exp/checkpoints/` (trained weights, ~2.7 GB) is git-ignored; run
`scripts/train.py` to reproduce. The committed JSONL/JSON files under `data/`
and `exp/` are the exact datasets, label map, and eval artifacts from the
reference run described above.

## Honest limitations

1. **Templated source data.** Glaive v2 utterances are synthetic and templated;
   dedup removed exact repeats, but paraphrase-level template artifacts remain.
   The model partly learns "which template family is this" rather than deep
   intent. Real assistant traffic with human-verified labels would be strictly
   better.
2. **Utterance-only input.** The router sees only the final user utterance, no
   conversation history and no tool descriptions. Follow-ups like "what about
   France?" are genuinely ambiguous without context.
3. **Small tail classes.** `generate_random_number` (37 train), `translate_text`
   (69), `get_definition` (74) have few unique utterances; expect weak F1 there.
4. **One decision per forward pass.** The router makes a single routing decision
   per pass; it does not score multiple parallel tool calls jointly.
5. **No teacher labeling / human review.** Labels come from the dataset's own
   function-call annotations, taken at face value.

## License

MIT. The training data derives from
[glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2),
which is Apache-2.0.
