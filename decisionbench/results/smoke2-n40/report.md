# decisionbench report

Headline metric per task type (with 95% bootstrap CI, 1000 resamples): choice = accuracy, score = quadratic weighted kappa, noul = AUROC, typed-decisions = decision accuracy. `n` is valid/sampled examples; metrics use valid examples only (see error rates in the systems table). ECE → ECE after temperature scaling is on the 70% evaluation fold, T fitted on the other 30%. † / ‡ mark training overlap (see the end of the report).

## Summary

Mean rank: rank of the headline metric among models that ran the same task (1 = best), averaged. Held-out headline: mean headline metric over tasks the model did not train on (per the contamination map; headline metrics differ by task type, so compare within a row set, not across tables). ECE after TS: mean over tasks with a temperature-scaling fold.

| model | mean rank ↓ | held-out mean headline | mean ECE after TS ↓ | p50 ms / 1-question request | p50 ms / typed-decisions case | $ / 1k decisions |
|---|---|---|---|---|---|---|
| laya-typed-decisions | 1.00 (1 tasks) | – (0 tasks) | – | – | 166.9 | $0 (local) |
| nli-zeroshot | 1.43 (7 tasks) | 0.720 (2 tasks) | 0.133 | 39.8 | 1533.9 | $0 (local) |
| laya-english | 2.50 (7 tasks) | 0.523 (5 tasks) | 0.179 | 37.4 | 177.9 | $0 (local) |
| laya-serve | 2.50 (7 tasks) | 0.523 (5 tasks) | 0.179 | 38.7 | 170.0 | $0 (local) |

## Held-out comparison

Tasks that no compared model trained on, derived from the contamination map (models whose training data is undisclosed are included and marked ?). Models fine-tuned on a benchmark's own training split are left out of this table.

| rank | model | typed-decisions (decision accuracy) | mean |
|---|---|---|---|
| 1 | nli-zeroshot | 0.531 | 0.531 |
| 2 | laya-english | 0.363 | 0.363 |
| 3 | laya-serve | 0.363 | 0.363 |

Excluded tasks: agnews (laya-english, laya-serve, nli-zeroshot); emotion (nli-zeroshot); banking77 (nli-zeroshot); sst5 (nli-zeroshot); sst2 (nli-zeroshot); boolq (laya-english, laya-serve).
Left out as specialists: laya-typed-decisions.

## Choice tasks

| task (labels) | model | n | accuracy [95% CI] | macro-F1 | top-2 acc | log-loss | Brier | ECE | ECE eval-fold → after TS | AURC ↓ | cov@95%acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews (4) | laya-english | 40/40 | 0.900 [0.80, 0.97]† | 0.883 | 1.000 | 0.217 | 0.117 | 0.108 | 0.146 → 0.093 (T=0.69) | 0.010 | 0.900 |
| agnews (4) | laya-serve | 40/40 | 0.900 [0.80, 0.97]† | 0.883 | 1.000 | 0.217 | 0.117 | 0.108 | 0.146 → 0.093 (T=0.69) | 0.010 | 0.900 |
| agnews (4) | nli-zeroshot | 40/40 | 0.875 [0.78, 0.97]† | 0.852 | 1.000 | 0.514 | 0.201 | 0.096 | 0.104 → 0.110 (T=2.10) | 0.121 | 0.775 |
| emotion (6) | laya-english | 40/40 | 0.575 [0.42, 0.70] | 0.463 | 0.850 | 1.416 | 0.627 | 0.297 | 0.276 → 0.235 (T=2.30) | 0.240 | 0.200 |
| emotion (6) | laya-serve | 40/40 | 0.575 [0.42, 0.70] | 0.463 | 0.850 | 1.416 | 0.627 | 0.297 | 0.276 → 0.235 (T=2.30) | 0.240 | 0.200 |
| emotion (6) | nli-zeroshot | 40/40 | 0.800 [0.68, 0.93]† | 0.700 | 0.950 | 0.529 | 0.263 | 0.128 | 0.168 → 0.104 (T=1.14) | 0.045 | 0.675 |
| banking77 (77) | laya-english | 40/40 | 0.300 [0.17, 0.45] | 0.230 | 0.500 | 2.698 | 0.858 | 0.261 | 0.293 → 0.256 (T=1.04) | 0.455 | 0.100 |
| banking77 (77) | laya-serve | 40/40 | 0.300 [0.17, 0.45] | 0.230 | 0.500 | 2.698 | 0.858 | 0.261 | 0.293 → 0.256 (T=1.04) | 0.455 | 0.100 |
| banking77 (77) | nli-zeroshot | 40/40 | 0.750 [0.62, 0.88]† | 0.673 | 0.875 | 1.489 | 0.566 | 0.395 | 0.417 → 0.245 (T=0.41) | 0.256 | 0.025 |

## Score tasks

| task (levels) | model | n | exact acc | within-1 acc | MAE (E[score]) | Spearman ρ | QWK [95% CI] | ECE | log-loss |
|---|---|---|---|---|---|---|---|---|---|
| sst5 (5) | laya-english | 40/40 | 0.275 | 0.675 | 1.033 | 0.790 | 0.422 [0.28, 0.55] | 0.337 | 2.202 |
| sst5 (5) | laya-serve | 40/40 | 0.275 | 0.675 | 1.033 | 0.790 | 0.422 [0.28, 0.55] | 0.337 | 2.202 |
| sst5 (5) | nli-zeroshot | 40/40 | 0.575 | 0.950 | 0.521 | 0.841 | 0.757 [0.63, 0.85]‡ | 0.162 | 1.064 |

## Noul tasks

| task | model | n | AUROC [95% CI] | AUPRC | acc@0.5 | F1@0.5 | best F1 (oracle thr) | Brier | log-loss | ECE | ECE eval-fold → after TS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sst2 | laya-english | 40/40 | 0.955 [0.87, 1.00] | 0.966 | 0.900 | 0.889 | 0.923 @0.21 | 0.093 | 0.332 | 0.100 | 0.125 → 0.191 (T=1.65) |
| sst2 | laya-serve | 40/40 | 0.955 [0.87, 1.00] | 0.966 | 0.900 | 0.889 | 0.923 @0.21 | 0.093 | 0.332 | 0.100 | 0.125 → 0.191 (T=1.65) |
| sst2 | nli-zeroshot | 40/40 | 1.000 [1.00, 1.00]‡ | 1.000 | 0.950 | 0.944 | 1.000 @0.06 | 0.042 | 0.134 | 0.044 | 0.038 → 0.063 (T=1.78) |
| boolq | laya-english | 40/40 | 0.766 [0.58, 0.93]† | 0.755 | 0.775 | 0.816 | 0.889 @0.23 | 0.173 | 0.803 | 0.097 | 0.115 → 0.120 (T=0.92) |
| boolq | laya-serve | 40/40 | 0.766 [0.58, 0.93]† | 0.755 | 0.775 | 0.816 | 0.889 @0.23 | 0.173 | 0.803 | 0.097 | 0.115 → 0.120 (T=0.92) |
| boolq | nli-zeroshot | 40/40 | 0.909 [0.76, 1.00] | 0.903 | 0.775 | 0.791 | 0.923 @0.02 | 0.179 | 0.636 | 0.175 | 0.219 → 0.140 (T=1.68) |

## typed-decisions

All five questions of a case are asked in one request. Decision accuracy: the model's argmax (noul: p ≥ 0.5) equals the dataset's gold `label` (the argmax of its teacher-averaged gold distribution, ties already broken). The dataset card does not define its leaderboard accuracy, so these are not guaranteed to match its numbers. CI: bootstrap over cases. KL: KL(gold ‖ model) against the full gold distribution, in nats. QWK: mean over score questions.

| model | cases | decision acc [95% CI] | choice acc | score exact acc | score MAE | score QWK | noul acc | noul AUROC | noul Brier ↓ | KL from gold ↓ | ECE ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-english | 400/400 | 0.363 [0.34, 0.39] | 0.290 | 0.324 | 0.923 | 0.077 | 0.488 | 0.448 | 0.304 | 0.570 | 0.174 |
| laya-serve | 400/400 | 0.363 [0.34, 0.39] | 0.290 | 0.324 | 0.923 | 0.077 | 0.488 | 0.448 | 0.304 | 0.570 | 0.174 |
| laya-typed-decisions (specialist) | 400/400 | 0.767 [0.75, 0.79]† | 0.733 | 0.724 | 0.460 | 0.759 | 0.858 | 0.937 | 0.144 | 0.117 | 0.214 |
| nli-zeroshot | 400/400 | 0.531 [0.51, 0.55] | 0.542 | 0.425 | 0.743 | 0.275 | 0.663 | 0.749 | 0.242 | 0.398 | 0.085 |
| prior (train label frequencies, no input) | 2000 decisions | 0.483 | – | – | – | – | – | – | – | – | – |
| uniform (expected, no input) |  | 0.318 | – | – | – | – | – | – | – | – | – |

Decision accuracy per workflow (config):

| model | agent_trace_observability | customer_service | invoice_processing | security_incidents |
|---|---|---|---|---|
| laya-english | 0.388 | 0.384 | 0.362 | 0.318 |
| laya-serve | 0.388 | 0.384 | 0.362 | 0.318 |
| laya-typed-decisions | 0.730 | 0.766 | 0.806 | 0.766 |
| nli-zeroshot | 0.440 | 0.612 | 0.530 | 0.544 |
| prior | 0.362 | 0.452 | 0.518 | 0.602 |

## Robustness

Option-order sensitivity: each choice question re-asked with its options reversed (typed-decisions: all choice questions of a case reversed in one request; n counts decisions).

| task | model | n | prediction flip rate ↓ | mean TV distance ↓ | probe errors |
|---|---|---|---|---|---|
| agnews | laya-english | 40 | 0.050 | 0.033 | 0 |
| agnews | laya-serve | 40 | 0.050 | 0.033 | 0 |
| agnews | nli-zeroshot | 40 | 0.000 | 0.000 | 0 |
| emotion | laya-english | 40 | 0.050 | 0.098 | 0 |
| emotion | laya-serve | 40 | 0.050 | 0.098 | 0 |
| emotion | nli-zeroshot | 40 | 0.000 | 0.000 | 0 |
| banking77 | laya-english | 40 | 0.750 | 0.621 | 0 |
| banking77 | laya-serve | 40 | 0.750 | 0.621 | 0 |
| banking77 | nli-zeroshot | 40 | 0.000 | 0.000 | 0 |
| typed-decisions | laya-english | 600 | 0.357 | 0.132 | 0 |
| typed-decisions | laya-serve | 600 | 0.357 | 0.132 | 0 |
| typed-decisions | laya-typed-decisions | 600 | 0.068 | 0.040 | 0 |
| typed-decisions | nli-zeroshot | 600 | 0.000 | 0.000 | 0 |

Negation consistency: the statement and its negation asked separately.

| task | model | n | mean \|p_pos + p_neg − 1\| ↓ | contradiction rate ↓ | probe errors |
|---|---|---|---|---|---|
| sst2 | laya-english | 40 | 0.415 | 0.400 | 0 |
| sst2 | laya-serve | 40 | 0.415 | 0.400 | 0 |
| sst2 | nli-zeroshot | 40 | 0.103 | 0.100 | 0 |

## Systems

Latency is per request, sequential, warm (the first requests of each task are excluded). p50/p95/mean and req/s pool the one-question tasks; typed-decisions requests (all five questions of a case) are reported separately. Throughput = 1 / mean latency. Load time includes any first-time weight download and, for server-backed models, server start-up. Cost uses Jev's list price ($0.042 / 1M input tokens) spread over the decisions per request; local models cost $0 in API fees.

| model | device | answered by | load s | first request ms | p50 ms | p95 ms | mean ms | req/s | typed-dec. case p50 ms | typed-dec. case p95 ms | error rate | input tokens/req | $ / 1k decisions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-english | mps | – | 2.6 | 232 | 37.4 | 80.3 | 44.3 | 22.6 | 177.9 | 265.8 | 0.000 | 959 | $0 (local) |
| laya-serve | local server | laya-rl-agent | 5.1 | 1419 | 38.7 | 85.6 | 47.6 | 21.0 | 170.0 | 282.2 | 0.000 | 959 | $0 (local) |
| laya-typed-decisions | mps | – | 37.3 | 121 | – | – | – | – | 166.9 | 236.1 | 0.000 | 1456 | $0 (local) |
| nli-zeroshot | mps | – | 1.6 | 1248 | 39.8 | 331.4 | 93.2 | 10.7 | 1533.9 | 4190.0 | 0.000 | 3281 | $0 (local) |

Mean warm latency (ms) per request, per task:

| model | agnews | emotion | banking77 | sst5 | sst2 | boolq | typed-decisions |
|---|---|---|---|---|---|---|---|
| laya-english | 43.6 | 36.4 | 80.0 | 29.3 | 27.3 | 49.1 | 175.4 |
| laya-serve | 52.6 | 39.2 | 81.7 | 31.4 | 29.6 | 51.4 | 175.8 |
| laya-typed-decisions | – | – | – | – | – | – | 161.2 |
| nli-zeroshot | 57.8 | 43.0 | 360.1 | 34.4 | 24.1 | 39.7 | 1717.3 |

## Training overlap

† the model was trained on this task's dataset (a different split). ‡ the model was trained on a dataset built from the same source texts (SST-2 / SST-5 / rotten_tomatoes). Marked cells measure in-distribution performance, not zero-shot generalization.

- **laya-english**: agnews†, boolq† (source: https://github.com/NandhaKishorM/laya#english-tasks)
- **laya-serve**: agnews†, boolq† (source: https://github.com/NandhaKishorM/laya#english-tasks)
- **laya-typed-decisions**: typed-decisions† (source: https://huggingface.co/convaiinnovations/laya)
- **nli-zeroshot**: agnews†, emotion†, banking77†, sst5‡, sst2‡ (source: https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0#training-data)

## Skipped models

- **jev**: TYPESAFE_API_KEY is not set; export it to benchmark 'jev-1.13.0'
- **kev-0.8b**: kev-0.8b server is not installed (/Users/srallaba/projects/project_decisionmodels/third_party/kev/.venv/bin/python missing); set it up with: (cd third_party/kev && uv sync --extra serve), after cloning https://github.com/jaredpalmer/kev into third_party/kev if it is not there (not `pip install kev`: that PyPI name is an unrelated package)
- **kev-4b**: kev-4b server is not installed (/Users/srallaba/projects/project_decisionmodels/third_party/kev/.venv/bin/python missing); set it up with: (cd third_party/kev && uv sync --extra serve), after cloning https://github.com/jaredpalmer/kev into third_party/kev if it is not there (not `pip install kev`: that PyPI name is an unrelated package)
- **open-jev-deberta**: com-kotobalabs/open-jev-deberta-v3-large runs Python shipped in its Hub repo (typed_decisions/*.py @ 19bf9a6481); review it, then enable with --model-opts '{"open-jev-deberta": {"allow_hub_code": true}}'
