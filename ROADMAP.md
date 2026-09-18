# Sys One — Next Steps Roadmap

Experiments to strengthen and stress-test selective tool-call routing.
Work proceeds one item at a time; each gets a report before moving on.

## Datasets

- [ ] **(1) BFCL — Berkeley Function-Calling Leaderboard** [IN PROGRESS]
      Irrelevance split (= our `no_tool` with ground truth) + parallel
      (multi-label) split. The parallel split tests a limit of the current
      single-label head.
- [ ] (2) τ-bench — multi-turn agentic dialogues with user simulator;
      tests whether routing errors compound across steps.
- [ ] (3) xLAM / APIGen (Salesforce) — large-scale synthetic data;
      tests catalog scaling to 100+ tools.
- [ ] (4) Glaive function-calling v2 — second training distribution;
      checks results are not a data artifact.
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
