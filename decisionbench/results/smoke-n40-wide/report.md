# decisionbench report

Headline metric per task type (with 95% bootstrap CI, 1000 resamples): choice = accuracy, score = quadratic weighted kappa, noul = AUROC. `n` is valid/sampled examples; metrics use valid examples only (see error rates in the systems table). ECE → ECE after temperature scaling is on the 70% evaluation fold, T fitted on the other 30%.

## Choice tasks

| task (labels) | model | n | accuracy [95% CI] | macro-F1 | top-2 acc | log-loss | Brier | ECE | ECE eval-fold → after TS | AURC ↓ | cov@95%acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews (4) | laya-english-wide | 40/40 | 0.900 [0.80, 0.97] | 0.883 | 1.000 | 0.217 | 0.117 | 0.108 | 0.146 → 0.093 (T=0.69) | 0.010 | 0.900 |
| banking77 (77) | laya-english-wide | 40/40 | 0.325 [0.20, 0.47] | 0.216 | 0.575 | 2.648 | 0.836 | 0.287 | 0.274 → 0.158 (T=1.56) | 0.425 | 0.050 |

## Score tasks

| task (levels) | model | n | exact acc | within-1 acc | MAE (E[score]) | Spearman ρ | QWK [95% CI] | ECE | log-loss |
|---|---|---|---|---|---|---|---|---|---|

## Noul tasks

| task | model | n | AUROC [95% CI] | AUPRC | acc@0.5 | F1@0.5 | best F1 (oracle thr) | Brier | log-loss | ECE | ECE eval-fold → after TS |
|---|---|---|---|---|---|---|---|---|---|---|---|

## Robustness

Option-order sensitivity: each choice question re-asked with its options reversed.

| task | model | n | prediction flip rate ↓ | mean TV distance ↓ | probe errors |
|---|---|---|---|---|---|
| agnews | laya-english-wide | 40 | 0.050 | 0.033 | 0 |
| banking77 | laya-english-wide | 40 | 0.700 | 0.582 | 0 |

## Systems

Latency is per request (one question), sequential, warm (the first requests of each task are excluded); throughput = 1 / mean latency. Load time includes any first-time weight download. Cost uses Jev's list price ($0.042 / 1M input tokens); local models cost $0 in API fees.

| model | device | load s | first request ms | p50 ms | p95 ms | mean ms | req/s | error rate | input tokens/req | $ / 1k decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| laya-english-wide | mps | 4.5 | 887 | 183.1 | 225.9 | 170.2 | 5.9 | 0.000 | 249 | $0 (local) |

Mean warm latency (ms) per task:

| model | agnews | banking77 |
|---|---|---|
| laya-english-wide | 138.8 | 201.6 |

## Skipped models

None.
