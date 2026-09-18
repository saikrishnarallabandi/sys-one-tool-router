# Sys One — Next Steps Roadmap

Experiments to strengthen and stress-test selective tool-call routing.
Work proceeds one item at a time; each gets a report before moving on.
Priority (per 2026-09-17): datasets with real training AND eval splits first;
eval-only benchmarks after.

## Datasets

- [x] **(1) BFCL — Berkeley Function-Calling Leaderboard** [DONE 2026-09-17]
      Frozen-router zero-shot transfer: 91.3% accuracy on 69 mappable single
      items, 98.4% selective @0.9. Key limitation found: fixed-catalog
      classifier cannot express "right tool, not in my catalog" and is
      overconfident OOD — escalate path needs an explicit out-of-catalog
      detector, not just the confidence threshold.
- [ ] **(2) Glaive function-calling v2** [IN PROGRESS] — has train AND test
      splits. Conservative function mapping + frozen-router baseline on the
      mapped test subset first; then decide: augment the 25-class head vs
      train for catalog expansion. Checks results are not a data artifact.
- [ ] (3) xLAM / APIGen (Salesforce) — large-scale synthetic data with
      training splits; tests catalog scaling to 100+ tools.
- [ ] (4) τ-bench — multi-turn agentic dialogues with user simulator
      (eval-only); tests whether routing errors compound across steps.
- [ ] (5) Multilingual / code-mixed queries (Hindi/Telugu-English);
      relevant to the Judith audience.

## Stress-test tasks

- [ ] (6) **Ontology ablation** — merge/split tool classes deliberately and
      measure the accuracy delta. Direct test of the paper's claim:
      "router quality is upper-bounded by tool ontology quality."
- [ ] (7) Catalog drift — add new tools at test time without retraining
      (prototype / nearest-class-mean head).
- [ ] (8) Adversarial routing — prompt-injection-style queries steering
      toward dangerous tools; abstention as a safety mechanism (Jataayu angle).
- [ ] (9) Calibration under shift — risk-coverage curves on shifted data
      (typos, paraphrases, new domains).
- [ ] (10) End-to-end pipeline — router picks tool, generator fills args;
      full task success + dollar/latency savings vs pure LLM-as-judge.

## Benchmark plan: classification vs generation (central thesis)

Routing is a discriminative task (pick one of K known tools); generation pays
autoregressive cost for a single discrete choice. The benchmark must establish
this with fair, same-model / same-hardware comparisons.

**Generative baselines — Sai:**
- Constrained decoding (grammar-restricted to valid tool names): kills the
  parse-failure strawman; the fair latency fight.
- Bigger generative models: does scale close the quality gap? (Latency gap
  only widens; do not dodge the quality question.)

**Sys One runs — Veronica:**
- [x] Same-model/same-hardware latency+quality vs greedy 0-shot/3-shot (Table 1).
- [x] Risk-coverage / selective accuracy; BFCL zero-shot transfer + OOD finding.
- [x] Calibration: ECE15/NLL/Brier before vs after temperature scaling
      (2026-09-17: ECE15 0.0067 -> 0.0077, NLL 0.0310 -> 0.0261, Brier
      0.0131 -> 0.0120 — already well-calibrated in-distribution).
- [ ] Reliability diagram + AUROC of confidence for in-catalog vs
      out-of-catalog (needs per-item inference pass; queued).
- [ ] Classifier-side latency breakdown: tokenize vs forward vs argmax
      (needs healthy GPU; queued).
- [ ] Catalog-size scaling 25 -> 100+ tools via xLAM/APIGen: does latency
      stay flat, how does accuracy degrade? (queued after Glaive v2).
