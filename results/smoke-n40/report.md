# decisionbench report

Headline metric per task type (with 95% bootstrap CI, 1000 resamples): choice = accuracy, score = quadratic weighted kappa, noul = AUROC. `n` is valid/sampled examples; metrics use valid examples only (see error rates in the systems table). ECE → ECE after temperature scaling is on the 70% evaluation fold, T fitted on the other 30%.

## Choice tasks

| task (labels) | model | n | accuracy [95% CI] | macro-F1 | top-2 acc | log-loss | Brier | ECE | ECE eval-fold → after TS | AURC ↓ | cov@95%acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews (4) | laya-english | 40/40 | 0.900 [0.80, 0.97] | 0.883 | 1.000 | 0.217 | 0.117 | 0.108 | 0.146 → 0.093 (T=0.69) | 0.010 | 0.900 |
| agnews (4) | nli-zeroshot | 40/40 | 0.875 [0.78, 0.97] | 0.852 | 1.000 | 0.514 | 0.201 | 0.096 | 0.104 → 0.110 (T=2.10) | 0.121 | 0.775 |
| emotion (6) | laya-english | 40/40 | 0.575 [0.42, 0.70] | 0.463 | 0.850 | 1.416 | 0.627 | 0.297 | 0.276 → 0.235 (T=2.30) | 0.240 | 0.200 |
| emotion (6) | nli-zeroshot | 40/40 | 0.800 [0.68, 0.93] | 0.700 | 0.950 | 0.529 | 0.263 | 0.128 | 0.168 → 0.104 (T=1.14) | 0.045 | 0.675 |
| banking77 (77) | laya-english | 40/40 | 0.300 [0.17, 0.45] | 0.230 | 0.500 | 2.698 | 0.858 | 0.261 | 0.293 → 0.256 (T=1.04) | 0.455 | 0.100 |
| banking77 (77) | nli-zeroshot | 40/40 | 0.750 [0.62, 0.88] | 0.673 | 0.875 | 1.489 | 0.566 | 0.395 | 0.417 → 0.245 (T=0.41) | 0.256 | 0.025 |

## Score tasks

| task (levels) | model | n | exact acc | within-1 acc | MAE (E[score]) | Spearman ρ | QWK [95% CI] | ECE | log-loss |
|---|---|---|---|---|---|---|---|---|---|
| sst5 (5) | laya-english | 40/40 | 0.275 | 0.675 | 1.033 | 0.790 | 0.422 [0.28, 0.55] | 0.337 | 2.202 |
| sst5 (5) | nli-zeroshot | 40/40 | 0.575 | 0.950 | 0.521 | 0.841 | 0.757 [0.63, 0.85] | 0.162 | 1.064 |

## Noul tasks

| task | model | n | AUROC [95% CI] | AUPRC | acc@0.5 | F1@0.5 | best F1 (oracle thr) | Brier | log-loss | ECE | ECE eval-fold → after TS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sst2 | laya-english | 40/40 | 0.955 [0.87, 1.00] | 0.966 | 0.900 | 0.889 | 0.923 @0.21 | 0.093 | 0.332 | 0.100 | 0.125 → 0.191 (T=1.65) |
| sst2 | nli-zeroshot | 40/40 | 1.000 [1.00, 1.00] | 1.000 | 0.950 | 0.944 | 1.000 @0.06 | 0.042 | 0.134 | 0.044 | 0.038 → 0.063 (T=1.78) |
| boolq | laya-english | 40/40 | 0.766 [0.58, 0.93] | 0.755 | 0.775 | 0.816 | 0.889 @0.23 | 0.173 | 0.803 | 0.097 | 0.115 → 0.120 (T=0.92) |
| boolq | nli-zeroshot | 40/40 | 0.909 [0.76, 1.00] | 0.903 | 0.775 | 0.791 | 0.923 @0.02 | 0.179 | 0.636 | 0.175 | 0.219 → 0.140 (T=1.68) |

## Robustness

Option-order sensitivity: each choice question re-asked with its options reversed.

| task | model | n | prediction flip rate ↓ | mean TV distance ↓ | probe errors |
|---|---|---|---|---|---|
| agnews | laya-english | 40 | 0.050 | 0.033 | 0 |
| agnews | nli-zeroshot | 40 | 0.000 | 0.000 | 0 |
| emotion | laya-english | 40 | 0.050 | 0.098 | 0 |
| emotion | nli-zeroshot | 40 | 0.000 | 0.000 | 0 |
| banking77 | laya-english | 40 | 0.750 | 0.621 | 0 |
| banking77 | nli-zeroshot | 40 | 0.000 | 0.000 | 0 |

Negation consistency: the statement and its negation asked separately.

| task | model | n | mean \|p_pos + p_neg − 1\| ↓ | contradiction rate ↓ | probe errors |
|---|---|---|---|---|---|
| sst2 | laya-english | 40 | 0.415 | 0.400 | 0 |
| sst2 | nli-zeroshot | 40 | 0.103 | 0.100 | 0 |

## Systems

Latency is per request (one question), sequential, warm (the first requests of each task are excluded); throughput = 1 / mean latency. Load time includes any first-time weight download. Cost uses Jev's list price ($0.042 / 1M input tokens); local models cost $0 in API fees.

| model | device | load s | first request ms | p50 ms | p95 ms | mean ms | req/s | error rate | input tokens/req | $ / 1k decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| laya-english | mps | 2.5 | 443 | 38.1 | 86.7 | 46.9 | 21.3 | 0.000 | 138 | $0 (local) |
| nli-zeroshot | mps | 45.5 | 1466 | 42.0 | 384.5 | 117.5 | 8.5 | 0.000 | 577 | $0 (local) |

Mean warm latency (ms) per task:

| model | agnews | emotion | banking77 | sst5 | sst2 | boolq |
|---|---|---|---|---|---|---|
| laya-english | 51.9 | 38.3 | 79.0 | 29.8 | 31.6 | 51.0 |
| nli-zeroshot | 124.7 | 66.6 | 357.5 | 34.4 | 55.9 | 65.7 |

## Skipped models

- **jev**: TYPESAFE_API_KEY is not set; export it to benchmark Jev
