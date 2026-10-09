"""The adapter interface every benchmarked model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from decisionbench.types import Answer, QuestionSpec


class DecisionModel(ABC):
    """A typed-decision model.

    Construction loads the model (weights, clients). Raise `ModelUnavailable` from `__init__` or
    `predict` when the model cannot run in this environment; the runner then skips it and records
    the reason. Raise `InvalidOutput` (or any other exception) from `predict` for a per-request
    failure; the runner records it against that example and counts it in the error rate.
    """

    name: str = "unnamed"
    # "local" models run on this machine and cost $0; "api" models are billed per input token.
    deployment: str = "local"
    # USD per 1M input tokens; None when not billed.
    price_per_1m_input_tokens: float | None = None

    @property
    def device(self) -> str:
        return "unknown"

    @abstractmethod
    def predict(self, state: str | dict | list, questions: dict[str, QuestionSpec]) -> dict[str, Answer]:
        """Answer every question about `state` (text, or a JSON object/array) in one request.
        Each Answer's probs follow its spec's order."""

    def info(self) -> dict:
        """Static facts recorded next to the results (resolved model id, config overrides, ...)."""
        return {}

    def close(self) -> None:
        """Release weights/connections. Called once the runner is done with the model."""
