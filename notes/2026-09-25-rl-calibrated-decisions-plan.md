## Goal

Decide whether and how to build a typed-decision model (choice / score / noul, one non-autoregressive pass over `state`) with reinforcement learning that is both accurate **and** calibrated, using this repo (`decisionbench`) as the evaluator.

## Success Criteria

- A training recipe is agreed: base starting point, what `q` (confidence) means for our outputs, exact reward formula, algorithm, data, and eval gates.
- The recipe preserves what `decisionbench` already measures: headline metrics (accuracy / QWK / AUROC / decision accuracy), ECE + ECE-after-temperature-scaling, Brier / log-loss, robustness probes, and contamination-aware held-out comparison.
- No implementation starts until this plan is approved.

## Context And Current Facts

**What "RL for calibrated decisions" is (from inspected sources):**

- RLCD (TypeSafe's term for Jev's post-training): instead of rewarding text people prefer (RLHF) or answers a checker can verify (RLVR), train the model to return decisions whose probabilities match how often those decisions turn out right. Calibration is a group property (of 1000 answers at 0.8, ~800 should be right), which makes thresholds meaningful for automation. TypeSafe argues RLHF rewards sounding right and narrows the output distribution, which is the wrong objective for unattended decisions. TypeSafe has published no paper, training details, or calibration curves for RLCD as of Sep 2026 — so RLCD is a goal statement, not a reproducible recipe.
- RLCR (Damani et al., peer-reviewed recipe): models output an answer `y` plus verbalized confidence `q`, optimized with `R_RLCR = 1_{y≡y*} − (q − 1_{y≡y*})²` (correctness + Brier). Proven to incentivize both accuracy and calibration when the calibration term is a **bounded** proper scoring rule. Log loss, though proper, is unbounded and does **not** incentivize correctness — so Brier, not log-likelihood, is the calibration term. Uses GRPO from a base model (Qwen2.5-7B in their setup), no KL regularization, plus a small format reward.
- Calibration-aware RL diagnosis (Yaldiz et al.): across decision benchmarks, RLVR gives higher accuracy but extreme overconfidence; SFT gives better calibration with smaller gains. Diagnosis: decision tokens act as extraction steps and inherit overconfidence from reasoning traces — there are no calibrated paths to reinforce from the base model. Fix: directly regulate decision-token probabilities; reduces ECE up to 9 points while keeping RLVR accuracy.

**Workspace facts (verified in this repo):**

- `decisionbench/types.py`: `QuestionSpec` + `Answer` (probs aligned to spec order); `choice/score/noul_answer()` already validate distributions — reusable as the reward computer's input contract.
- `decisionbench/tasks.py`: 7 tasks (agnews / emotion / banking77 / sst5 / sst2 / boolq / typed-decisions); typed-decisions asks 5 questions per case in one request with full gold distributions.
- `decisionbench/metrics.py` + `decisionbench/report.py`: ECE (15 bins), ECE-after-temperature-scaling (30% cal fold), Brier, log-loss, AUROC/QWK/accuracy, AURC, coverage@95%, robustness (option-order flip/TV, negation consistency), bootstrap CIs, contamination-marked held-out tables.
- `decisionbench/contamination.py`: overlap map (laya / kev / open-jev / nli-zeroshot trained on several of our tasks; jev undisclosed) — any training plan must respect this or its "gains" are in-distribution illusions.
- Adapters (`decisionbench/models`): Laya is the local System-One-style reference; NLI baseline is order-invariant by construction.

## Constraints And Non-goals

- Non-goal: no training code, no weight runs, no new dependencies in this planning turn.
- The model must stay a **single-pass typed-decision head** (no free-form reasoning chains at inference) — that is what `DecisionModel.predict(state, questions)` and the systems/latency accounting assume.
- Eval stays inside `decisionbench.run` + `report`; no new metric framework.

## Key Decisions

1. **Reward = correctness + Brier (bounded proper scoring rule).** Recommended: per-decision `r = 1_{hit} − (q − 1_{hit})²`, averaged over the decisions in the request (typed-decisions: 5/case). Rejected: pure binary RLVR reward (proven to drive overconfidence), RLHF preference reward (rewards confident-sounding mistakes), log-loss-as-reward (does not incentivize correctness per RLCR theorem).
2. **Confidence `q` = the model's own output distribution, not a separate verbalized head.** For choice: `q = p(argmax)`; score: full level distribution (Brier over levels); noul: `p(true)` / `[1−p, p]`. Why: our inference contract returns distributions in one pass; adding a verbalized-confidence channel would change the product surface and latency. Assumption: the Yaldiz et al. "regulate decision-token probabilities" finding transfers to our distribution head — flagged as the highest-risk assumption (see Validation).
3. **RL post-training only — no SFT warmup (user decision 2026-09-25).** Post-train an existing typed-decision model directly with proper-scoring rewards. Dropped: SFT warmup and SFT-only baseline arms.
4. **Train on typed-decisions train (+ optionally one synthetic held-out family); evaluate on the contamination-aware held-out set.** Prevents fooling ourselves with banking77/ag_news/boolq overlap.
5. **Judge calibration by distribution + reliability + ECE jointly**, not ECE alone (a collapsed 0.51-everywhere model can score low ECE while being useless — the inspected calibration paper stresses confidence histograms + reliability diagrams + ECE together).

## Recommended Approach

RL post-training on typed-decisions train with per-decision proper-scoring reward (correctness + Brier-style calibration term) → evaluate with `decisionbench` headless (ECE-after-TS, Brier, headline metrics, robustness probes, held-out mean) → only then decide on a full run. No SFT stage. Keep inference single-pass throughout.

## Work Plan

1. **SFT baseline arm.** Fine-tune the chosen base on typed-decisions train with cross-entropy; record `decisionbench` report as the calibration-friendly baseline. (Surfaces: new training script only; `decisionbench/` untouched.)
2. **Reward function unit.** Implement `reward(record, gold)` = mean over decisions of `hit − (q − hit)²`, with `q` per Decision 2; reuse `decision_hit`/`decision_distribution` semantics from `metrics.py`. Pure function, no model code.
3. **RL env + algorithm spike.** GRPO-style policy optimization on short rollouts, small format-validity reward, sampling temperature high enough to surface diverse (including uncertain) paths — directly addresses the "no calibrated paths" diagnosis. Start with noul + choice before score.
4. **Decisionbench eval gate.** Run `--tasks all --robustness` on train-adjacent and held-out splits; require: headline accuracy ≥ SFT arm, ECE-after-TS and Brier better than both SFT and pure-RLVR probe arms, no robustness-probe regression.
5. **Deferral check.** Convert calibrated probs into thresholds (coverage@95%acc, selective risk) to prove the calibration is operationally useful, not just a lower ECE number.
6. **Scale decision.** Only on passing (4)–(5): longer run + full report + leaderboard entry.

## Validation Plan

- Reward unit: `pytest tests/` equivalent for the new reward (hit/Brier edge cases: p=0/1, ties, noul threshold) — expected: all green.
- Short RL smoke: `.venv/bin/python -m decisionbench.run --models <candidate> --tasks typed-decisions,sst2,agnews --n 40 --robustness` then `.venv/bin/python -m decisionbench.report <out>` — expected: report renders, ECE-after-TS and Brier move before accuracy does.
- Highest-risk check: confidence-histogram spread + reliability diagram slope on held-out tasks (not just ECE scalar) — expected: spread-out confidences tracking the diagonal; a collapsed histogram fails the gate even with good ECE.
- Contamination audit: `contamination.py` overlap marks on every reported cell — expected: headline claims cite held-out-task means only.

## Risks / Rollback

- Decision-distribution `q` may not transfer the verbalized-confidence theorems cleanly → gate (4) catches it; rollback to SFT-only arm, which is kept throughout.
- Reward hacking (format compliance, probability mass gaming) → probe regressions + histogram check catch it; rollback = drop format reward weight, shorten horizon.
- Compute cost of RL vs. gain → time-box the spike (Phase 3); SFT arm always shippable.
- Training on contaminated tasks inflates scores → held-out-gate blocks promotion.

## Open Questions

- None blocking. Assumptions carried forward: (a) single-pass distribution head is the right `q` (vs. verbalized confidence); (b) base model choice is open — recommend starting from the strongest local SFT-able checkpoint, not necessarily Qwen2.5-7B; (c) score-level Brier vs. expected-score MSE weighting to be settled in the reward unit.

## Sources

- https://jevaiguide.com/concepts/rlcd/
- https://arxiv.org/abs/2507.16806v1
- https://arxiv.org/html/2507.16806v1
- https://arxiv.org/abs/2601.13284v1
- https://arxiv.org/html/2601.13284v1
