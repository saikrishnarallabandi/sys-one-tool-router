"""Model-agnostic question and answer types shared by adapters, tasks, runner and metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

QuestionType = Literal["choice", "score", "noul"]

# Adapters return probabilities rounded by the upstream model (Laya rounds to 4 decimals), so the
# sum can drift by a few 1e-4. Anything further off than this is treated as invalid output.
PROB_SUM_TOLERANCE = 0.02


class ModelUnavailable(RuntimeError):
    """The model cannot be used in this environment (missing credentials, bad auth, ...)."""


class InvalidOutput(ValueError):
    """The model answered, but the answer cannot be mapped onto the question spec."""


@dataclass(frozen=True)
class QuestionSpec:
    """A typed question.

    choice: `criteria` is an ordered tuple of (label, description | None).
    score:  `criteria` is an ordered tuple of level descriptions, level 0 first.
    noul:   `instructions` is the statement whose truth is estimated; `criteria` is empty or a tuple
            of ("true" | "false", description) pairs describing the outcomes.
    """

    type: QuestionType
    instructions: str
    criteria: tuple = ()

    def __post_init__(self):
        if self.type not in ("choice", "score", "noul"):
            raise ValueError(f"unknown question type {self.type!r}")
        crit = tuple(tuple(c) if self.type in ("choice", "noul") else c for c in self.criteria)
        object.__setattr__(self, "criteria", crit)
        if self.type == "choice":
            if not crit:
                raise ValueError("a choice question needs at least one label")
            for c in crit:
                if len(c) != 2 or not isinstance(c[0], str):
                    raise ValueError(f"choice criteria must be (label, description|None) pairs, got {c!r}")
            if len(set(self.labels)) != len(crit):
                raise ValueError("choice labels must be unique")
        elif self.type == "score":
            if not crit or not all(isinstance(c, str) for c in crit):
                raise ValueError("a score question needs a nonempty list of level descriptions")
        else:
            keys = [c[0] if len(c) == 2 else None for c in crit]
            if any(k not in ("true", "false") for k in keys) or len(set(keys)) != len(keys):
                raise ValueError(f"noul criteria must be distinct ('true'|'false', description) pairs, got {crit!r}")

    @property
    def labels(self) -> list[str]:
        """Choice labels in spec order (choice only)."""
        if self.type != "choice":
            raise AttributeError("labels is only defined for choice questions")
        return [label for label, _ in self.criteria]

    @property
    def noul_criteria(self) -> dict[str, str | None]:
        """Outcome descriptions keyed 'true'/'false' (noul only; empty when undescribed)."""
        if self.type != "noul":
            raise AttributeError("noul_criteria is only defined for noul questions")
        return dict(self.criteria)

    @property
    def n_options(self) -> int:
        return 2 if self.type == "noul" else len(self.criteria)

    def reversed(self) -> "QuestionSpec":
        """Same question with the option order reversed (choice only)."""
        if self.type != "choice":
            raise ValueError("only choice questions can be reordered")
        return QuestionSpec(self.type, self.instructions, tuple(reversed(self.criteria)))


@dataclass
class Answer:
    """A normalized answer.

    probs: choice/score -> np.ndarray aligned with the spec's label/level order;
           noul -> float p_true.
    pred:  choice -> argmax label index; score -> expected score sum(i * p_i); noul -> p_true.
    input_tokens: tokens reported for the whole request this answer came from (None if unknown).
    """

    type: QuestionType
    probs: Any
    pred: float | int
    raw: Any = None
    input_tokens: int | None = None
    extra: dict = field(default_factory=dict)

    def probs_list(self) -> list[float]:
        return [float(self.probs)] if self.type == "noul" else [float(x) for x in self.probs]


def _check_distribution(values: Sequence[float], what: str) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise InvalidOutput(f"{what}: empty probability vector")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise InvalidOutput(f"{what}: probabilities must be finite and nonnegative, got {p.tolist()}")
    s = p.sum()
    if abs(s - 1.0) > PROB_SUM_TOLERANCE:
        raise InvalidOutput(f"{what}: probabilities sum to {s:.4f}, not 1")
    return p / s


def choice_answer(spec: QuestionSpec, probabilities: Mapping[str, float], raw: Any = None,
                  input_tokens: int | None = None) -> Answer:
    """Align a label->probability mapping with the spec's label order."""
    labels = spec.labels
    missing = [lab for lab in labels if lab not in probabilities]
    extra = [lab for lab in probabilities if lab not in set(labels)]
    if missing or extra:
        raise InvalidOutput(f"choice labels mismatch: missing={missing[:5]} unexpected={extra[:5]}")
    p = _check_distribution([probabilities[lab] for lab in labels], "choice")
    return Answer("choice", p, int(np.argmax(p)), raw, input_tokens)


def score_answer(spec: QuestionSpec, probabilities: Mapping[int, float], raw: Any = None,
                 input_tokens: int | None = None) -> Answer:
    """Align a level->probability mapping (int or numeric-string keys) with levels 0..K-1."""
    k = len(spec.criteria)
    by_level: dict[int, float] = {}
    for key, value in probabilities.items():
        if isinstance(key, int) and not isinstance(key, bool):
            level = key
        elif isinstance(key, str) and key.strip().isdigit():
            level = int(key)
        else:
            raise InvalidOutput(f"score level key {key!r} is not an integer")
        if level in by_level:
            raise InvalidOutput(f"score level {level} appears twice")
        by_level[level] = value
    if sorted(by_level) != list(range(k)):
        raise InvalidOutput(f"score levels {sorted(by_level)} do not match 0..{k - 1}")
    p = _check_distribution([by_level[i] for i in range(k)], "score")
    return Answer("score", p, float(np.dot(np.arange(k), p)), raw, input_tokens)


def noul_answer(p_true: float, raw: Any = None, input_tokens: int | None = None) -> Answer:
    try:
        p = float(p_true)
    except (TypeError, ValueError):
        raise InvalidOutput(f"noul probability {p_true!r} is not a number") from None
    if not math.isfinite(p) or p < 0 or p > 1:
        raise InvalidOutput(f"noul probability {p!r} outside [0, 1]")
    return Answer("noul", p, p, raw, input_tokens)
