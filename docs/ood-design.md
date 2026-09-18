# OOD Challenge Set — Design

## Goal

Answer the reviewer's objection: *"You've shown Qwen3-0.6B classifies intents on a
templated dataset. Why should I believe this works on real tool-routing traffic?"*

The existing router (trained only on the Glaive-derived 10,936-example set, 25 classes)
is evaluated **without retraining** on a challenge set built to be distributionally
far from Glaive's templates.

## Categories and quotas (n = 800)

| # | Category | Per tool | Total | Construction |
|---|----------|----------|-------|--------------|
| 1 | Independent paraphrases | 10 | 250 | LLM-drafted, anti-template prompt |
| 2 | Indirect requests | 5 | 125 | LLM-drafted |
| 3 | Terse queries (2–6 words) | 4 | 100 | LLM-drafted |
| 4 | Misspellings / typos | 3 | 75 | Programmatic typo noise on cat. 1 |
| 5 | Conversational follow-ups | 2 | 50 | LLM-drafted (context + follow-up) |
| 6 | Implicit intent | 3 | 75 | LLM-drafted |
| 7 | Distractors (decoy keywords) | 2 | 50 | LLM-drafted |
| 8 | Ambiguous tool boundaries | 2 | 50 | LLM-drafted, dual-labeled |
| 9 | Genuine no_tool | — | 75 | LLM-drafted |
| **Total** | | | **800** | |

## Definitions

- **Independent paraphrase:** same intent, phrased nothing like Glaive. Banned openers
  ("Hi, I saw…", "Can you tell me…", "I need to…"), varied registers (texting,
  voice-assistant, formal), varied lengths.
- **Indirect request:** states a situation or need; never names the action or the tool.
  ("My lease renews at $1,850 with a 3% bump — new rent?" → calculate_discount is
  wrong; that one is percentage increase → no_tool. Kept only when the mapping is
  unambiguous to a rater.)
- **Terse:** keyword-style, as typed into a search box or barked at a voice assistant.
- **Misspellings:** 1–2 realistic corruptions per utterance (adjacent-key swaps,
  dropped/doubled letters), applied to category-1 items. Label unchanged.
- **Follow-up:** 1–2 turns of prior context are prepended to the input, because a
  production router sees conversation history. Stored as `context` + `text`;
  evaluation input is the concatenation.
- **Implicit:** intent is implied by the situation, never requested.
- **Distractor:** contains keywords of a *different* tool; the correct label requires
  reading past them. May resolve to no_tool.
- **Ambiguous:** genuinely borderline between two tools (mortgage vs loan payment,
  search_movies vs get_movie_details). Labeled with `label` (primary) and
  `acceptable` (list); either counts as correct.
- **no_tool:** chit-chat, opinions, feelings, advice, or questions none of the 24
  tools can answer.

## Generation protocol

- Drafted by phi4 via local ollama (a different model family/style than whatever
  produced Glaive), one prompt per (category), with per-tool quotas and 2–3
  hand-written few-shot examples per category.
- Anti-template constraints in every prompt: no Glaive-style openers, vary
  person/register/length, no two examples sharing a sentence skeleton.
- Output: JSON lines, `{"text", "tool"}` (ambiguous: `{"text", "primary", "acceptable"}`).

## Quality control

1. Schema check: every label ∈ the 25 canonical classes; ambiguous items carry
   `acceptable ⊇ {primary}`, |acceptable| = 2.
2. Exact-dedup against train/val/test (normalized: lowercase, strip punctuation).
   Target: 0 exact matches.
3. Template-overlap metric: share of 5-grams also present in train. Reported, not
   filtered — it quantifies "independent of Glaive's templates."
4. Human audit: 100% of ambiguous (50) and no_tool (75) items, plus a random 10%
   sample of the rest, read by the author. Items with debatable labels are fixed
   or dropped; the audit log is kept with the set.

## Evaluation

The frozen router (temperature T=1.791) is evaluated on the challenge set:
accuracy per category, accuracy with `acceptable` credit on ambiguous items,
and the same selective-routing analysis (coverage vs. accuracy, abstention rate).
No training on challenge data, ever — it is a test-only artifact.
