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
