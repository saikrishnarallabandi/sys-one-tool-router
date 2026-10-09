# Raschka "From Bag-of-Words to Jev" — learnings for decisionbench / td-delta

Date: 2026-10-01
Source: https://magazine.sebastianraschka.com/p/classifier-history-and-jev (Sep 29, 2026)

## Article summary (what's new to us)

1. **Jev positioning, quantified.** Jev on IMDb-25k: 96.47% (Choice) / 96.20% (Noul),
   ~$0.65 total. On par with a fine-tuned ModernBERT (~95% + 1-2% tunable headroom),
   but zero per-task fine-tuning. Thesis: Jev is the "ChatGPT moment for
   classification" — not a new capability, a plug-and-play generality + latency/cost
   wrapper around known classifier tech. Narrow specialists can still win on one
   task; the bar to justify training one is now much higher.
2. **DIY Jev head (§4).** Single-scalar scoring head shared across candidates +
   softmax over scores; BERT uses [CLS], GPT uses last-non-padding-token state.
   Fine-tuned jointly with cross-entropy. This is the same family as our
   Kev/Laya adapters — no architectural surprise, and Raschka's own verdict is
   that the API is trivial but matching Jev's breadth is not (data + eval grind).
3. **RLCD vs RLCR (§5).** Still no public RLCD details; CEO claims 100% synthetic
   data and a small (ModernBERT-scale) backbone is Raschka's guess. RLCR
   (R = c − (q−c)²) presented as the closest public analog — same recipe our
   RL plan already adopts. Raschka floats two adaptations for typed decisions:
   (a) RL with calibration rewards applied directly to typed-decision heads
   (no reasoning text), or (b) supervised CE + Brier joint loss as the
   "DPO-is-to-RLHF" analog. His (b) gave only a *modest* gain over temperature
   scaling on ModernBERT.
4. **"Is calibration necessary?" (§5.3) — the key section for us.** CE is
   self-calibrating only at the true optimum; in practice networks "overfit to
   NLL without overfitting to 0/1 loss" (Guo et al. 2017), so probability
   quality degrades while accuracy improves. The Brier term's motivation is
   sharpest in RL (no CE gradient there); in supervised training it just
   re-weights errors and must prove itself on held-out data.
5. **Calibration how-to (§5.1).** Temperature scaling on a held-out fold as the
   default production step (T > 1 = was overconfident); class prediction
   unchanged under scaling. Matches our ECE-after-TS metric design.
6. **Ecosystem notes.** Jev clones trail badly on breadth (Laya 92.3% IMDb but
   "fails the Tetris test"; Contrastive-LM 82.9%). OpenAI announced a Jev-like
   Decision API the same day (DevDay, limited preview) — competition is now
   platform-level, which raises the value of an open, evaluable alternative
   but also the bar for claiming generality.
7. **Methodology caveats worth stealing.** Jev runs are mildly non-deterministic
   (batch-dependent GPU kernels); Choice-vs-Noul disagree slightly (1.0 vs 0.98)
   on identical inputs; IMDb-may-be-in-training caveat stated explicitly.

## What this confirms in our project

- **Reward = correctness + Brier** (RL plan Decision 1): independently converged —
  Raschka lands on the same RLCR formula as the best public recipe.
- **q = output distribution, not a verbalized head** (RL plan Decision 2): his
  adaptation (a) is exactly this — RL calibration rewards on typed-decision
  heads without reasoning text.
- **ECE-after-TS + reliability-shape gating** (RL plan Decisions 5, Validation):
  temperature scaling as first-line calibration and "check held-out, not just
  the scalar" are both endorsed.
- **Contamination discipline**: his explicit "we don't know if IMDb was in
  training" caveat is our contamination.py practice; keep it.
- **v2 direction**: his "more/better data beats tuning" anecdote (§5, learning
  curves) supports the v2 dataset expansion over further hyperparameter grind
  on v1.

## What this challenges / refines

- **CE + Brier joint loss is weak medicine.** His ModernBERT result (modest gain
  over temp scaling) plus our own td-delta shape (Brier/NLL/AURC way better,
  ECE worse) jointly suggest: proper-score training improves *distribution
  quality* (KL 0.079 vs 0.226) but does NOT reliably fix *reliability-diagram
  slope*. Post-hoc temperature remains the load-bearing calibration step.
  Implication: stop expecting brier_w/ord_w tuning to close ECE; budget for
  per-primitive temperature (already validated: all-T=0.45 → ECE 0.020 on 9B).
- **Our failure mode is the unusual one.** Literature + article frame the enemy
  as overconfidence (RLVR/RLHF sharpening). Ours is UNDERconfidence from soft
  gold targets spreading mass (conf~0.6 bin → acc 0.91). This is good news:
  sharpening via T < 1 is safer than trying to de-overconfident an RLVR model,
  and accuracy@0.9 = 1.000 shows selective prediction already works.
- **Generality claim needs a Tetris-like test.** Raschka's breadth criterion
  (IMDb + something totally off-distribution) is a gap in our eval: v2-eval
  external choice at 0.80 is strong, but all our evals are still
  text-classification-shaped. A deliberately alien probe (structured-state
  decisions unlike training workflows, e.g. game/synthetic-logic states)
  would test the "Jev-like generality" claim the way his Tetris test does.
- **Non-determinism + Choice/Noul disagreement**: our reports assume stable
  point estimates with bootstrap CIs over examples. Server-side run variance
  (re-running the same model gives slightly different answers) is a second
  variance source we don't measure. Cheap add: re-run one (model, task) cell
  twice on a small n and report the disagreement rate.

## Concrete follow-ups (ordered)

1. **Ship per-primitive temperature for td-delta-v2-9b** (fit on held-out, report
   ECE before/after) — closes the one open gate; no retraining.
2. **Add an alien-probe eval** (one off-distribution structured-state task) to
   test breadth, not just in-family accuracy.
3. **Measure run variance**: duplicate one small cell, report flip rate; fold
   into report caveats if material.
4. **Track OpenAI Decision API** as a future decisionbench model (systemone-custom
   or new adapter) once broadly available — keeps the benchmark relevant.
5. **No change** to reward formula, q-definition, or contamination gating —
   article independently validates all three.
