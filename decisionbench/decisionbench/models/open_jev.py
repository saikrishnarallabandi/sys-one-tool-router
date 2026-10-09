"""open-jev-deberta (com-kotobalabs) adapter.

The model's inference code (`typed_decisions/open_jev.py`) ships inside the Hub repo rather than as a
package, so loading it imports Python downloaded from the Hub. Set `allow_hub_code=True` (e.g. via
`--model-opts '{"open-jev-deberta": {"allow_hub_code": true}}'`) only after reviewing that code; the
adapter refuses otherwise.
"""

from __future__ import annotations

import gc
import importlib
import json
import sys

from decisionbench.models.base import DecisionModel
from decisionbench.types import Answer, InvalidOutput, ModelUnavailable, QuestionSpec, choice_answer, noul_answer, score_answer

OPEN_JEV_REPO = "com-kotobalabs/open-jev-deberta-v3-large"
# Pinned so the reviewed code is the code that runs.
OPEN_JEV_REVISION = "19bf9a64815add579fbf6c907bef584d9277a8e4"


def option_text(label: str, desc: str | None) -> str:
    # open-jev takes bare option strings; descriptions are part of the task input, so they are inlined.
    return f"{label}: {desc}" if desc else label


def to_open_jev(spec: QuestionSpec) -> dict:
    if spec.type == "choice":
        return {"type": "choice", "instructions": spec.instructions,
                "options": [option_text(lab, desc) for lab, desc in spec.criteria]}
    if spec.type == "score":
        return {"type": "score", "instructions": spec.instructions, "options": list(spec.criteria)}
    # open-jev's noul has fixed (no, yes) options and no outcome descriptions.
    return {"type": "noul", "instructions": spec.instructions}


def from_open_jev(spec: QuestionSpec, q: dict, a: dict, tokens: int | None) -> Answer:
    if spec.type == "noul":
        return noul_answer(a.get("noul"), a, tokens)
    probs = a.get("probabilities") or {}
    options = q["options"]
    if len(set(options)) != len(options):
        raise InvalidOutput("open-jev keys probabilities by option text; option texts are not unique")
    if spec.type == "choice":
        return choice_answer(spec, {lab: probs.get(opt) for lab, opt in zip(spec.labels, options)}, a, tokens)
    return score_answer(spec, {i: probs.get(opt) for i, opt in enumerate(options)}, a, tokens)


class OpenJevModel(DecisionModel):
    """All of a request's questions go into one forward pass (the model's native packing). The model has
    a hard 512-token sequence limit; when a packed request exceeds it, each question is answered in its
    own pass instead and the answer is marked `packing: per-question`."""

    deployment = "local"
    price_per_1m_input_tokens = None

    def __init__(self, repo: str = OPEN_JEV_REPO, revision: str = OPEN_JEV_REVISION, device: str | None = None,
                 allow_hub_code: bool = False, module=None):
        self.repo, self.revision = repo, revision
        if module is None:
            if not allow_hub_code:
                raise ModelUnavailable(
                    f"{repo} runs Python shipped in its Hub repo (typed_decisions/*.py @ {revision[:10]}); "
                    "review it, then enable with --model-opts '{\"open-jev-deberta\": {\"allow_hub_code\": true}}'")
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo, revision=revision)
            sys.path.insert(0, path)
            module = importlib.import_module("typed_decisions.open_jev")
        self._m = module.OpenJev.from_pretrained(repo, device=device, revision=revision)

    @property
    def device(self) -> str:
        return str(getattr(self._m, "device", "unknown"))

    def info(self) -> dict:
        cfg = getattr(self._m, "config", {}) or {}
        return {"repo": self.repo, "revision": self.revision, "temperature": cfg.get("temperature"),
                "max_len": cfg.get("max_len"), "max_state_tokens": cfg.get("max_state_tokens")}

    def predict(self, state, questions: dict[str, QuestionSpec]) -> dict[str, Answer]:
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        qids = list(questions)
        qs = [to_open_jev(questions[qid]) for qid in qids]
        tokens = None  # the model does not report token usage
        packing = "packed"
        try:
            answers = self._m.decide(text, qs)
        except ValueError as e:
            if "> max_len" not in str(e) or len(qs) == 1:
                raise
            answers = [self._m.decide(text, [q])[0] for q in qs]
            packing = "per-question"
        if len(answers) != len(qs):
            raise InvalidOutput(f"open-jev returned {len(answers)} answers for {len(qs)} questions")
        out = {}
        for qid, q, a in zip(qids, qs, answers):
            out[qid] = from_open_jev(questions[qid], q, a, tokens)
            out[qid].extra["packing"] = packing
        return out

    def close(self) -> None:
        import torch

        self._m = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


VARIANTS = {
    "open-jev-deberta": (OpenJevModel, {}),
}
