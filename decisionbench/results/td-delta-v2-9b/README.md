---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B-Base
base_model_revision: 68c46c4b3498877f3ef123c856ecfde50c39f404
library_name: peft
tags:
- lora
- typed-decisions
- calibration
- choice
- qwen
---

# td-delta-v2-9b

LoRA adapter (r16) + scorer head on `Qwen/Qwen3.5-9B-Base` for **typed decisions**:
answer typed questions about a JSON state in one non-autoregressive pass, with
calibrated probabilities. Three primitives:

- `choice` — distribution over named labels (classification, MCQs, preferences)
- `score` — distribution over ordered levels 0..K-1 (ratings, severity)
- `noul` — P(statement is true) (yes/no entailment-style judgments)

Served through [`kev.serve`](https://github.com/jaredpalmer/kev) (`/v1/systemone`
API). This is a decision scorer, not a chat model.

## Eval results

Held-out `typed-decisions` test (400 cases / 1945 decisions):

| metric | td-delta-v2-9b | v1 9b delta | kev-4b base |
|---|---|---|---|
| decision acc | **0.783** | 0.774 | 0.665 |
| ECE | 0.140 | 0.149 | **0.091** |
| Brier | **0.334** | 0.350 | 0.468 |
| NLL | **0.585** | 0.612 | 0.819 |
| AURC | **0.079** | 0.087 | 0.195 |

Per primitive acc: choice 0.737, noul 0.840, score 0.774.
Full numbers (incl. per-task and selective-prediction metrics) are in
`eval-v1test.json`.

Broader v2 eval (6264 decisions, mostly eval-only sources incl. ChaosNLI and
RewardBench): acc 0.786, ECE 0.107 — choice 0.801 (5558), score 0.647 (600),
noul 0.783 (106). See `eval-v2eval.json`.

## Use

```bash
# per kev's instructions: clone + uv sync --extra serve (PyPI `kev` is unrelated)
git clone https://github.com/jaredpalmer/kev
cd kev && uv sync --extra serve
uv run python -m kev.serve --run srallaba/td-delta-v2-9b --port 8008
```

```bash
curl -s localhost:8008/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"note": "The staging deploy is failing with a TLS error; rollback takes 20 minutes."},
  "questions": {"q": {"type": "choice", "instructions": "What should be done?",
    "criteria": {"roll back": "revert to the last good build", "wait": "hold and retry"}}}
}' | python -m json.tool
```

Programmatic (same backend the server uses):

```python
from kev.serve import Checkpoint, LoadOptions, Server, default_device
ck = Checkpoint("srallaba/td-delta-v2-9b")
tok, model = ck.load(default_device(), LoadOptions.from_env())
server = Server(ck, tok, model, default_device())
```

## Training

- Warm start from a v1 run on typed-decisions only; then 2 epochs over the v2
  mix (13,423 recs): typed-decisions 1200 + HelpSteer2 1760 (score) +
  UltraFeedback-binarized 1760 (choice) + HateSpeech 2639 grouped-soft (score) +
  MMLU-aux 1760, HellaSwag 1760, ANLI-r1 1760 (choice) + PubMedQA 784 (noul).
- Objective: soft gold distributions + Brier (`brier_w 0.5`) + ordinal
  ranked-probability-score (`ord_w 0.5`) calibration terms.
- Hyperparams: lr 2e-5, batch 1 / accum 8, bf16, LoRA r16/α32 on all
  projections, seed 0; 2 × 3346 optimizer steps, loss 0.867 → 0.535 (~20h on
  one A6000). Exact config: `training_config.json`, `training_metrics.json`.
- Leakage audit: normalized exact-match of train vs the v1 test split → 0 leaks.

## Limitations

- **Underconfident out of the box.** Post-training against soft targets spreads
  probability mass, so argmax confidence understates argmax accuracy (worst on
  `score`). Fit a temperature on held-out data (T ≈ 0.4 on the v1 test cuts ECE
  to ~0.03) before trusting the probabilities.
- `score` is the weakest primitive (0.647 on harder 3/5-point eval scales).
- High-cardinality choice (70+ options) and non-English states are untested.
- Intended for scoring decisions, not open-ended generation.

## Files

| file | what |
|---|---|
| `adapter_model.safetensors` (165M) + `adapter_config.json` | LoRA weights (PEFT) |
| `head.pt` (8M) | kev scorer head + run metadata |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | tokenizer |
| `training_config.json`, `training_metrics.json` | exact training setup + curves |
| `eval-v1test.json`, `eval-v2eval.json` | full eval reports |

## License

Apache-2.0 (same as the Qwen3.5 base model and the kev training code).
