# decisionbench

A benchmark harness for "System One" typed-decision models: models that answer typed questions
about a piece of text (`state`) in one non-autoregressive pass, with calibrated probabilities.
Three question primitives are covered:

| primitive | the model returns | example |
|---|---|---|
| `choice` | a distribution over named labels | news topic, intent |
| `score`  | a distribution over ordered levels 0..K-1 | 1–5 sentiment |
| `noul`   | P(statement is true) | "The text expresses positive sentiment." |

Models compared:
- **TypeSafe Jev** (hosted API).
- **Kev** (open Jev reconstruction, local server).
- **Convai Laya** (local checkpoints, or its own HTTP server).
- **open-jev-deberta**.
- A **zero-shot NLI baseline** (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`).

Any other `/v1/systemone` endpoint can be plugged in via `systemone-custom`.

## Running

```bash
# everything with robustness probes: 300 examples per task, typed-decisions' whole test split (400 cases)
.venv/bin/python -m decisionbench.run --models laya-english,nli-zeroshot,jev,kev-4b --tasks all --robustness

# a quick smoke run into a named directory
.venv/bin/python -m decisionbench.run --models laya-english,nli-zeroshot,jev --tasks all --n 40 --robustness --out results/smoke-n40

# regenerate metrics.json + report.md from saved predictions (no inference)
.venv/bin/python -m decisionbench.report results/smoke-n40

# list registered models and tasks
.venv/bin/python -m decisionbench.run --list

# tests
.venv/bin/python -m pytest
```

`--n` caps every task. Without it, tasks take 300 examples, except typed-decisions, which takes
its whole test split. Other options: `--seed` (sample + bootstrap + calibration split seed, default 0), `--warmup` (requests
per model/task excluded from latency, default 3), `--retry-errors` (on resume, re-run examples
that failed), `--model-opts` (JSON constructor overrides per model, e.g.
`'{"laya-english": {"head_max_len": 384}}'`).

**Resume.** Re-running with the same `--out` reuses every stored prediction whose example id,
seed and model options match; a (model, task) with all its predictions is skipped without
loading the model. Increasing `--n` keeps the existing examples (the sample for a smaller `n` is
a prefix of the sample for a larger one) and only runs the new ones. Adding `--robustness` later
re-runs only the examples missing probes.

**Jev.** Export your key before running; without it Jev is skipped and the reason recorded in the
report:

```bash
export TYPESAFE_API_KEY=...          # optional: TYPESAFE_BASE_URL
```

`jev` pins `jev-1.13.0`; `jev-latest` follows the alias. Bad credentials (401/403) or an unknown
model (404) also skip the model with the server's message. Requests are sequential and use the
SDK's retry policy (3 retries, 30 s timeout).

**One code path for every System One server.** `jev`, `kev-*`, `laya-serve` and
`systemone-custom` are all `SystemOneModel` (`decisionbench/models/systemone.py`). It drives
`typesafe-sdk`'s `TypeSafeClient` with a configurable `base_url` and `model`. Differences between
Jev, Kev and Laya therefore come from the models, not from separate client code. For every request
the report records the `model` the server says answered. At load it records the server's
`GET /v1/models` (for Kev: checkpoint, base, backend, dtype).

**Local servers (Kev, laya-serve).**
- **Auto-start:** the runner starts the server when the model is requested, waits until it is
  healthy, and stops it afterwards. If a healthy server is already on the port it is reused and
  left running.
- **Wrong checkpoint:** a Kev server on the port that reports a different checkpoint in
  `/v1/models` (including Kev's silent fallback to `runs/smoke` when `--run` is not found) is
  refused.
- **Manual start:**

```bash
.venv/bin/python -m decisionbench.serve kev-4b      # start, wait until healthy, serve until Ctrl-C
.venv/bin/python -m decisionbench.serve --list      # kev-0.8b :8007, kev-4b :8008, kev-9b :8009, laya-serve :8000
```

Launching never installs anything. Set Kev up once, from git (the PyPI package named `kev` is
unrelated). It needs Python 3.12–3.13 and uses its own venv, and it serves through MLX on Apple
Silicon:

```bash
git clone https://github.com/jaredpalmer/kev third_party/kev
(cd third_party/kev && uv sync --extra serve)
```

Server logs go to `.cache/decisionbench/servers/<name>.log`. For `laya-serve`:
`uv pip install --python .venv/bin/python 'laya[serve]'`.

**Any other endpoint** (meraGPT, an Open-Jev server, a Kev on Modal):

```bash
.venv/bin/python -m decisionbench.run --models systemone-custom --model-opts \
  '{"systemone-custom": {"base_url": "https://…", "model": "sd-1", "api_key_env": "MERAGPT_API_KEY", "price_per_1m_input_tokens": 0.03}}'
```

**open-jev-deberta.** Its inference code ships as Python inside the Hub repo, so the adapter
refuses to import it unless you opt in, after reviewing
`typed_decisions/*.py` at the pinned revision:
`--model-opts '{"open-jev-deberta": {"allow_hub_code": true}}'`.

### Output directory

```
preds/<model>__<task>.jsonl      one line per example: gold, aligned probabilities, prediction,
                                 latency, input tokens, raw model answer, error, robustness probes
preds/<model>__<task>.meta.json  the sample (example ids), seed and settings behind the lines
models/<model>.json              device, load time, first-request latency, pricing, config
skipped.json                     models that could not run, with the reason
runs.json                        invocation history
metrics.json, report.md          recomputed from the above by decisionbench.report
```

## Tasks

| task | source (split) | type | question |
|---|---|---|---|
| `agnews` | fancyzhx/ag_news (test) | choice, 4 labels with descriptions | topic of the article |
| `emotion` | dair-ai/emotion (test) | choice, 6 labels with descriptions | emotion expressed |
| `banking77` | PolyAI/banking77 (test) | choice, 77 humanized intent labels | high-cardinality stress test |
| `sst5` | SetFit/sst5 (test) | score, 5 levels very negative..very positive | sentiment |
| `sst2` | stanfordnlp/sst2 (validation; test is unlabelled) | noul | "The text expresses positive sentiment." |
| `boolq` | google/boolq (validation) | noul, state = passage | "The answer to the question '<q>?' is yes." |
| `typed-decisions` | LocalLLaMA/typed-decisions `all` (test, 400 cases) | 5 typed questions per case (choice, score, noul with descriptions), JSON state | the dataset's own questions, all asked in one request |

Each task takes the first `n` examples of a seeded permutation of its split. Split rows are cached
under `.cache/decisionbench/` (override with `DECISIONBENCH_CACHE`). Label orders are checked
against each dataset's `ClassLabel` names at load time. `PolyAI/banking77` on the Hub is a
loading script, which `datasets` 4+ refuses to run. So its label names come from the repo's
`dataset_infos.json`, and its rows come from the `test.csv` that script downloads.

## Metrics

Metrics are computed on examples whose request succeeded. The error rate is reported separately,
per task and pooled per model. Headline metrics have 95% percentile-bootstrap CIs (1000 seeded
resamples of examples).

**choice.** accuracy (headline); macro-F1 over the classes present in gold or predictions; top-2
accuracy; log-loss (mean −log p_gold, probabilities clipped at 1e-6); multiclass Brier (mean
Σ_k (p_k − 1[k=y])², range 0–2); ECE (15 equal-width bins on max probability); ECE after
temperature scaling; AURC; and coverage at 95% accuracy.
- **Temperature scaling:** T is fitted to minimize NLL on a seeded 30% calibration fold, by
  bounded search over log T in [0.01, 100]. The report shows ECE on the remaining 70% before and
  after, plus T. T > 1 means the model was overconfident. At least 10 examples are needed in each
  fold, otherwise it is omitted. It is applied as softmax(log p / T), which for a noul is exactly
  sigmoid(logit(p) / T).
- **AURC:** mean selective risk (error rate among the k most-confident examples) over coverages
  1/n..n/n. Examples with tied confidence can't be split by a threshold, so their errors are
  spread evenly across the tie. Lower is better.
- **Coverage@95%acc:** the largest fraction of examples a confidence threshold can keep while
  holding accuracy ≥ 0.95.

**typed-decisions** (multi-question). Scored over all decisions of the sampled cases, overall and
per workflow (config):
- **Decision accuracy (headline):** the model's argmax (noul: p ≥ 0.5) equals the gold `label`.
  The CI comes from a bootstrap over cases.
  - The dataset card has no formula for its leaderboard "Accuracy". Its `label` field is the
    argmax of the teacher-averaged gold distribution, with the 31 exact ties already broken, so
    this is the natural reading.
  - Using it, a train-split label-frequency prior scores 0.4835 on the full test split; the card
    lists 0.470 for "Prior". Treat leaderboard numbers as approximately, not exactly, comparable.
- **Per primitive:**
  - choice accuracy;
  - score exact accuracy, MAE of the expected score, and QWK (averaged over score questions);
  - noul accuracy, AUROC and Brier.
- **Distribution metrics:** KL(gold ‖ model) against the full gold distribution, and pooled
  top-label ECE.
- **Reference rows:** a prior (train label frequencies, ignores the input) and a uniform guess,
  shown for the same cases.

**score.** exact accuracy (argmax level), within-1 accuracy, MAE of the expected score
(Σ i·p_i), Spearman ρ between expected score and gold, quadratic weighted kappa on the argmax
(headline), ECE, log-loss.

**noul.** AUROC (headline), AUPRC, accuracy and F1 at 0.5, best-threshold F1, Brier, log-loss,
ECE and ECE after temperature scaling. The best-threshold F1 is an oracle: the threshold is chosen
on the same examples. ECE is top-label ECE on [1−p, p].

**Training overlap.** `decisionbench/contamination.py` maps each model family to the datasets it
was trained on, with source URLs. In report.md:
- **† and ‡:** † marks headline cells where the model trained on that task's dataset. ‡ marks
  those where it trained on a dataset built from the same source texts (SST-2 / SST-5 /
  rotten_tomatoes).
- **Undisclosed data:** a model whose training data is undisclosed (Jev) is marked ?.
- **Held-out table:** ranks models on the tasks no compared model trained on, derived from the
  map. Models fine-tuned on a benchmark's own train split (laya-typed-decisions) are left out.
- **Summary table (top of the report):** per model, the mean rank across tasks, the mean
  headline metric on held-out tasks, the mean ECE after temperature scaling, p50 latency and
  $ / 1k decisions.

**robustness** (`--robustness`)
- *Option-order sensitivity* (choice tasks): each question is asked again with its options
  reversed, and the answer is mapped back to the original label order. Reports the prediction flip
  rate and the mean total-variation distance ½Σ|p − p'|. The NLI baseline scores each option
  independently, so it is order-invariant by construction and scores 0 here.
- *Negation consistency* (sst2): "The text expresses negative sentiment." is asked separately.
  Reports mean |p_pos + p_neg − 1| and the contradiction rate (both > 0.5 or both < 0.5).

**systems.** These are per-request latencies: sequential, one question per request, measured
around `predict()`. Warm latency p50/p95/mean excludes the first `--warmup` requests of each
task; throughput is 1 / mean warm latency.
- **Load time:** constructing the adapter. It includes any first-time weight download.
- **First request:** one untimed-for-stats request made right after load.
- **Other columns:** error/invalid-output rate, input tokens per request (as reported by the model:
  Jev's billed tokens, Laya's attended tokens, the NLI baseline's tokens summed over hypotheses),
  and $ per 1k decisions at Jev's $0.042 per 1M input tokens. Local models are listed as
  "$0 (local)".

## Models

| name | what |
|---|---|
| `laya-english` | `convaiinnovations/laya` root checkpoint (ModernBERT-large, 512 tokens) |
| `laya-english-wide` | same, `head_max_len=384` so long option lists are not truncated |
| `laya-multilingual` | `convaiinnovations/laya`, subfolder `multilingual` |
| `laya-typed-decisions` | `convaiinnovations/laya`, subfolder `typed-decisions` |
| `laya-router` | `laya.Router()` with automatic language routing |
| `jev`, `jev-latest` | TypeSafe API, `jev-1.13.0` / `jev-latest` (`SystemOneModel`) |
| `kev-0.8b`, `kev-4b`, `kev-9b` | `jaredpalmer/kev-*` served by `kev.serve` on :8007 / :8008 / :8009 (`SystemOneModel`, auto-started) |
| `laya-serve` | `laya-serve` HTTP server on :8000, English checkpoint (`SystemOneModel`, auto-started); protocol parity with `laya-english` |
| `systemone-custom` | any `/v1/systemone` endpoint configured via `--model-opts` |
| `open-jev-deberta` | `com-kotobalabs/open-jev-deberta-v3-large` @ pinned revision. Option texts are "label: description"; noul descriptions are not supported by the model; packed requests over its 512-token limit fall back to one pass per question (marked in the record) |
| `nli-zeroshot` | DeBERTa-v3-large zero-shot NLI. choice/score: softmax over entailment logits of one hypothesis per option ("{instructions} Answer: {label} ({description})."), scored in one batch. noul: P(entailment) of the statement. |

Laya options (`--model-opts`): `max_len` and `head_max_len`. `head_max_len` (default 192) is the
token budget for instructions plus all option texts. With 77 options each option gets cut to 4
tokens, so raise it for high-cardinality questions.

## Adding a model

Add one file under `decisionbench/models/`. The registry imports every module in that package and
collects its `VARIANTS`:

```python
# decisionbench/models/kev.py
from decisionbench.models.base import DecisionModel
from decisionbench.types import ModelUnavailable, choice_answer, score_answer, noul_answer

class KevModel(DecisionModel):
    deployment = "local"                 # or "api" (then set price_per_1m_input_tokens)
    price_per_1m_input_tokens = None

    def __init__(self, checkpoint="org/kev-9b", device=None):
        import torch, transformers       # import heavy deps lazily
        ...                              # raise ModelUnavailable("why") if it cannot run here

    @property
    def device(self): return "mps"

    def predict(self, state, questions):     # questions: dict[str, QuestionSpec]
        out = {}
        for qid, spec in questions.items():
            if spec.type == "choice":        # spec.criteria = ((label, description|None), ...)
                out[qid] = choice_answer(spec, {"label": 0.7, ...}, raw=..., input_tokens=...)
            elif spec.type == "score":       # spec.criteria = (level0 description, level1, ...)
                out[qid] = score_answer(spec, {0: 0.1, 1: 0.9}, raw=..., input_tokens=...)
            else:                            # noul: spec.instructions is the statement
                out[qid] = noul_answer(0.83, raw=..., input_tokens=...)
        return out

VARIANTS = {"kev-9b": (KevModel, {"checkpoint": "org/kev-9b"})}
```

If the model serves `POST /v1/systemone`, no new file is needed: register a `SystemOneModel`
variant in `decisionbench/models/systemone.py`, or use `systemone-custom`.

`choice_answer` / `score_answer` / `noul_answer` align probabilities with the spec's order and
raise `InvalidOutput` for missing or extra labels, non-finite or negative values, or totals off by
more than 0.02. Anything `predict` raises counts as an error for that example, except
`ModelUnavailable`, which skips the model. Add a mapping test next to the existing ones in
`tests/test_adapters.py`.

## Adding a task

Add a `Task` to `TASKS` in `decisionbench/tasks.py`. It needs:
- `load_rows()`: returns every row of the split as dicts with `id`, `gold` and the fields `build`
  uses.
- `build(row) -> Example`:
  - one question: `single(id, state, qid, QuestionSpec(...), gold, probes)`;
  - several questions per state: `Example(id, state, {qid: spec}, {qid: gold}, gold_probs, probes, group)`,
    with `type="multi"`.

`probes` enable robustness checks: `{"reversed": spec.reversed()}` for choice, or
`{"negation": QuestionSpec("noul", ...)}` for a negated statement. Add the task's datasets to
`TASK_SOURCES` in `contamination.py` so overlap marks and the held-out table cover it.

## Known caveats

- Laya rounds its probabilities to 4 decimals. Options below 5e-5 come back as exactly 0 and are
  clipped to 1e-6 in log-loss, which inflates Laya's log-loss on confidently wrong answers.
- The English Laya checkpoint ships a choice temperature for 11+ options (0.10) that the library
  clamps to 0.5, with a warning that confidence there is uncalibrated. This affects banking77
  calibration only; argmax is unchanged.
- `laya-router` latency includes language detection and routing.
- The NLI baseline is not a clean zero-shot baseline on most tasks here. Its training mix includes
  banking77, dair-ai/emotion, ag_news and rotten_tomatoes (the source of the SST sentences); the
  report marks those cells.
- For server-backed models, latency is measured end to end from the client, so it includes local
  HTTP overhead. The server-reported `latency_ms`, when a server sends one (Kev does), is stored
  per answer in the prediction files as `server_latency_ms`.
