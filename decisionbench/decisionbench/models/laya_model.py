"""Convai Laya (local, torch) adapter."""

from __future__ import annotations

import gc
from typing import Any

from decisionbench.models.base import DecisionModel
from decisionbench.types import Answer, InvalidOutput, QuestionSpec, choice_answer, noul_answer, score_answer

LAYA_REPO = "convaiinnovations/laya"


def to_laya_question(spec: QuestionSpec) -> dict[str, Any]:
    q: dict[str, Any] = {"type": spec.type, "instructions": spec.instructions}
    if spec.type == "choice":
        q["criteria"] = {label: desc for label, desc in spec.criteria}
    elif spec.type == "score":
        q["criteria"] = list(spec.criteria)
    elif spec.criteria:
        q["criteria"] = spec.noul_criteria
    return q


def from_laya_answers(questions: dict[str, QuestionSpec], result: dict[str, Any]) -> dict[str, Answer]:
    answers = result.get("answers") or {}
    tokens = (result.get("usage") or {}).get("input_tokens")
    out: dict[str, Answer] = {}
    for qid, spec in questions.items():
        raw = answers.get(qid)
        if not isinstance(raw, dict) or raw.get("type") != spec.type:
            raise InvalidOutput(f"question {qid!r}: missing or mistyped answer {raw!r}")
        if spec.type == "choice":
            out[qid] = choice_answer(spec, raw.get("probabilities") or {}, raw, tokens)
        elif spec.type == "score":
            out[qid] = score_answer(spec, raw.get("probabilities") or {}, raw, tokens)
        else:
            out[qid] = noul_answer(raw.get("noul"), raw, tokens)
    return out


class LayaModel(DecisionModel):
    """One Laya checkpoint (`subfolder` picks it from the bundle repo), or the auto-routing Router.

    `max_len` / `head_max_len` override the checkpoint's token budget per call. `head_max_len`
    bounds the instructions + option texts; with many options (banking77) the default 192 truncates
    every option to a few tokens, so raise it for high-cardinality questions.
    """

    deployment = "local"
    price_per_1m_input_tokens = None

    def __init__(self, repo: str = LAYA_REPO, subfolder: str | None = None, router: bool = False,
                 device: str | None = None, max_len: int | None = None, head_max_len: int | None = None):
        import laya

        self.repo, self.subfolder, self.router = repo, subfolder, router
        self.max_len, self.head_max_len = max_len, head_max_len
        if router:
            self._router = laya.Router(device=device)
            # Load the checkpoint the router picks for English text now, so load time is
            # measured at construction rather than inside the first timed request.
            self._agent = self._router.load(self._router.default)
        else:
            self._router = None
            self._agent = laya.load(repo, device=device, subfolder=subfolder)

    @property
    def device(self) -> str:
        return str(self._agent.device)

    def info(self) -> dict:
        cfg = self._agent.cfg
        return {
            "repo": self.repo, "subfolder": self.subfolder, "router": self.router,
            "max_len": self.max_len or cfg.get("max_len"),
            "head_max_len": self.head_max_len or cfg.get("head_max_len"),
        }

    def predict(self, state, questions: dict[str, QuestionSpec]) -> dict[str, Answer]:
        qs = {qid: to_laya_question(spec) for qid, spec in questions.items()}
        kw = {"max_len": self.max_len, "head_max_len": self.head_max_len}
        if self._router is not None:
            result = self._router.predict(state, qs, **kw)
        else:
            result = self._agent.predict(state, qs, **kw)
        answers = from_laya_answers(questions, result)
        if self._router is not None:
            routed = (result.get("routing") or {}).get("model")
            for a in answers.values():
                a.extra["routed_to"] = routed
        return answers

    def close(self) -> None:
        import torch

        if self._router is not None:
            self._router.unload()
        self._agent = self._router = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


VARIANTS = {
    "laya-english": (LayaModel, {"subfolder": None}),
    "laya-multilingual": (LayaModel, {"subfolder": "multilingual"}),
    "laya-typed-decisions": (LayaModel, {"subfolder": "typed-decisions"}),
    "laya-router": (LayaModel, {"router": True}),
    # Same English checkpoint with room for long option lists (see class docstring).
    "laya-english-wide": (LayaModel, {"subfolder": None, "head_max_len": 384}),
}
