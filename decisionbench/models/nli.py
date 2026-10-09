"""Zero-shot NLI baseline (DeBERTa-v3 entailment) adapter."""

from __future__ import annotations

import gc
import json

import numpy as np

from decisionbench.models.base import DecisionModel
from decisionbench.types import Answer, QuestionSpec, choice_answer, noul_answer, score_answer

NLI_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"


def hypotheses(spec: QuestionSpec) -> list[str]:
    """One hypothesis per option (choice/score); for noul, [false, true] outcome descriptions when both
    are given, otherwise the statement itself."""
    if spec.type == "choice":
        return [f"{spec.instructions} Answer: {label} ({desc})." if desc else f"{spec.instructions} Answer: {label}."
                for label, desc in spec.criteria]
    if spec.type == "score":
        return [f"{spec.instructions} Answer: {level}." for level in spec.criteria]
    crit = spec.noul_criteria
    if crit.get("true") and crit.get("false"):
        return [crit["false"], crit["true"]]
    return [spec.instructions]


def premise(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class NLIZeroShotModel(DecisionModel):
    """choice/score: softmax over the entailment logits of the per-option hypotheses, scored in one
    batched forward pass. noul: P(entailment) of the statement against not_entailment, or, when both
    outcomes are described, softmax over the entailment logits of the [false, true] descriptions."""

    deployment = "local"
    price_per_1m_input_tokens = None

    def __init__(self, model_id: str = NLI_MODEL, device: str | None = None, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id, self.max_length = model_id, max_length
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self._device).eval()
        label2id = {k.lower(): v for k, v in self._model.config.label2id.items()}
        if "entailment" not in label2id:
            raise ValueError(f"{model_id} has no 'entailment' label: {self._model.config.label2id}")
        self._entail = label2id["entailment"]
        if len(label2id) != 2:
            raise ValueError(f"{model_id}: expected a 2-way entailment/not_entailment head, got {label2id}")

    @property
    def device(self) -> str:
        return str(self._device)

    def info(self) -> dict:
        return {"model": self.model_id, "max_length": self.max_length}

    def _logits(self, premise: str, hyps: list[str]) -> tuple[np.ndarray, int]:
        import torch

        enc = self._tok([premise] * len(hyps), hyps, truncation="only_first", max_length=self.max_length,
                        padding=True, return_tensors="pt")
        n_tokens = int(enc["attention_mask"].sum())
        with torch.inference_mode():
            logits = self._model(**{k: v.to(self._device) for k, v in enc.items()}).logits
        return logits.float().cpu().numpy(), n_tokens

    def predict(self, state, questions: dict[str, QuestionSpec]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        per_q: dict[str, tuple[np.ndarray, int]] = {}
        total_tokens = 0
        text = premise(state)
        for qid, spec in questions.items():
            logits, n = self._logits(text, hypotheses(spec))
            per_q[qid] = (logits, n)
            total_tokens += n
        for qid, spec in questions.items():
            logits = per_q[qid][0]
            raw = {"logits": logits.round(4).tolist()}
            if spec.type == "noul" and len(logits) == 2:
                out[qid] = noul_answer(float(_softmax(logits[:, self._entail])[1]), raw, total_tokens)
            elif spec.type == "noul":
                out[qid] = noul_answer(float(_softmax(logits[0])[self._entail]), raw, total_tokens)
            else:
                p = _softmax(logits[:, self._entail])
                if spec.type == "choice":
                    out[qid] = choice_answer(spec, dict(zip(spec.labels, p)), raw, total_tokens)
                else:
                    out[qid] = score_answer(spec, dict(enumerate(p)), raw, total_tokens)
        return out

    def close(self) -> None:
        import torch

        self._model = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


VARIANTS = {
    "nli-zeroshot": (NLIZeroShotModel, {}),
}
