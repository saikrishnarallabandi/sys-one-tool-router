"""Any server speaking TypeSafe's `POST /v1/systemone` protocol, driven through `typesafe-sdk`.

TypeSafe Jev (hosted), Kev (local), laya-serve (local) and any other compatible endpoint share this
one code path; they differ only in base_url, model name, credentials, pricing and whether the
harness launches the server.
"""

from __future__ import annotations

import os
from typing import Any

from decisionbench.models.base import DecisionModel
from decisionbench.types import (
    Answer, InvalidOutput, ModelUnavailable, QuestionSpec, choice_answer, noul_answer, score_answer,
)

JEV_PRICE_PER_1M_INPUT_TOKENS = 0.042


def to_sdk_question(spec: QuestionSpec) -> Any:
    from typesafe_sdk import Choice, Noul, Score

    if spec.type == "choice":
        return Choice(instructions=spec.instructions, criteria={label: desc for label, desc in spec.criteria})
    if spec.type == "score":
        return Score(instructions=spec.instructions, criteria=list(spec.criteria))
    return Noul(instructions=spec.instructions, criteria=spec.noul_criteria or None)


def _server_latency_ms(response: Any) -> float | None:
    """The `latency_ms` field some servers (Kev) add to the body; the SDK model drops unknown fields."""
    raw = getattr(response, "__dict__", {}).get("_raw")
    if raw is None:
        return None
    try:
        value = raw.json().get("latency_ms")
    except ValueError:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def from_sdk_response(questions: dict[str, QuestionSpec], response: Any) -> dict[str, Answer]:
    """Map a typesafe_sdk SystemOneResponse onto Answers (score probability keys are ints)."""
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "input_tokens", None)
    by_type = {"choice": response.choices, "score": response.scores, "noul": response.nouls}
    extra = {"served_model": getattr(response, "model", None), "server_latency_ms": _server_latency_ms(response)}
    out: dict[str, Answer] = {}
    for qid, spec in questions.items():
        ans = by_type[spec.type].get(qid)
        if ans is None:
            raise InvalidOutput(f"question {qid!r}: no {spec.type} answer in response")
        raw = ans.model_dump(mode="json")
        if spec.type == "choice":
            out[qid] = choice_answer(spec, dict(ans.probabilities), raw, tokens)
        elif spec.type == "score":
            out[qid] = score_answer(spec, dict(ans.probabilities), raw, tokens)
        else:
            out[qid] = noul_answer(ans.noul, raw, tokens)
        out[qid].extra.update(extra)
    return out


class SystemOneModel(DecisionModel):
    """A /v1/systemone endpoint.

    base_url: None uses the SDK default (TypeSafe's API, or TYPESAFE_BASE_URL).
    api_key: explicit key; otherwise read from `api_key_env`. Local servers accept any key.
    server: name of a `decisionbench.serve.SERVERS` spec to launch (or reuse) at load time.
    expect_run: for Kev, the checkpoint `/v1/models` must report, so a stale server on the port
        (or Kev's silent fallback checkpoint) is refused rather than benchmarked.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 api_key_env: str = "TYPESAFE_API_KEY", deployment: str = "api",
                 price_per_1m_input_tokens: float | None = None, server: str | None = None,
                 expect_run: str | None = None, max_retries: int = 3, timeout: float = 30.0, client: Any = None):
        self.model, self.deployment = model, deployment
        self.price_per_1m_input_tokens = price_per_1m_input_tokens
        self._server = None
        self._server_info: dict = {}
        if client is not None:
            self._client, self.base_url = client, base_url
            return
        if server is not None:
            from decisionbench.serve import SERVERS, ManagedServer

            if server not in SERVERS:
                raise ValueError(f"unknown server {server!r}; known: {', '.join(SERVERS)}")
            self._server = ManagedServer(SERVERS[server], api_key or "local").start()
            base_url = self._server.base_url
        self.base_url = base_url
        key = api_key if api_key is not None else os.environ.get(api_key_env, "").strip()
        if not key:
            if deployment == "local":
                key = "local"
            else:
                raise ModelUnavailable(f"{api_key_env} is not set; export it to benchmark {model!r}")
        from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

        try:
            self._client = TypeSafeClient(api_key=key, model=model, base_url=base_url, timeout=timeout,
                                          retry=RetryPolicy(max_retries=max_retries,
                                                            timeout=timeout * (max_retries + 1)))
        except TypeSafeError as e:
            self._stop_server()
            raise ModelUnavailable(f"TypeSafe client could not be created: {e}") from e
        self._server_info = self._describe_server(key)
        if expect_run is not None:
            runs = {m.get("run") for m in self._server_info.get("models", []) if isinstance(m, dict)}
            if expect_run not in runs:
                self._stop_server()
                raise ModelUnavailable(f"server at {self.base_url} serves {sorted(r for r in runs if r) or 'an unknown run'}, "
                                       f"not {expect_run!r}; stop it or pick another port")

    def _describe_server(self, key: str) -> dict:
        """What the server says about itself (GET /v1/models), recorded with the results."""
        from decisionbench.serve import get_json

        if self._server is not None and self._server.spec.health_path != "/v1/models":
            return {"health": get_json(self._server.base_url + self._server.spec.health_path, key)}
        if self.base_url is None:
            return {}  # hosted API: the answering model is recorded per request instead
        doc = get_json(self.base_url.rstrip("/") + "/v1/models", key)
        if not isinstance(doc, dict):
            return {"models_endpoint": "unavailable"}
        keep = ("name", "run", "base", "lora", "device", "backend", "dtype", "temperature", "release_date")
        return {"models": [{k: m.get(k) for k in keep if k in m} for m in doc.get("models", []) if isinstance(m, dict)]}

    @property
    def device(self) -> str:
        if self._server is not None or (self.base_url or "").startswith(("http://127.0.0.1", "http://localhost")):
            models = self._server_info.get("models") or [{}]
            backend = models[0].get("backend")
            return f"local server ({backend})" if backend else "local server"
        return "remote API"

    def info(self) -> dict:
        return {"model": self.model, "base_url": self.base_url or "SDK default",
                "server_launched_by_harness": bool(self._server and self._server.owned), **self._server_info}

    def predict(self, state, questions: dict[str, QuestionSpec]) -> dict[str, Answer]:
        from typesafe_sdk import (
            TypeSafeAPIConnectionError, TypeSafeAuthenticationError, TypeSafeNotFoundError, TypeSafePermissionDeniedError,
        )

        try:
            response = self._client.system_one(state, {qid: to_sdk_question(s) for qid, s in questions.items()})
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as e:
            raise ModelUnavailable(f"server rejected the credentials: {e}") from e
        except TypeSafeNotFoundError as e:
            raise ModelUnavailable(f"model {self.model!r} not available at {self.base_url or 'the TypeSafe API'}: {e}") from e
        except TypeSafeAPIConnectionError as e:
            if self._server is not None and self._server.proc is not None and self._server.proc.poll() is not None:
                raise ModelUnavailable(f"server process died: {self._server.log_tail()}") from e
            raise
        return from_sdk_response(questions, response)

    def _stop_server(self) -> None:
        if self._server is not None:
            self._server.stop()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()
        self._stop_server()


def _custom(**kw) -> SystemOneModel:
    if not kw.get("base_url") or not kw.get("model"):
        raise ModelUnavailable("systemone-custom needs --model-opts "
                               '\'{"systemone-custom": {"base_url": "...", "model": "...", "api_key_env": "..."}}\'')
    return SystemOneModel(**kw)


def _kev(size: str) -> tuple:
    return (SystemOneModel, {"model": "kev-latest", "server": f"kev-{size}", "deployment": "local",
                             "expect_run": f"jaredpalmer/kev-{size}"})


VARIANTS = {
    "jev": (SystemOneModel, {"model": "jev-1.13.0", "price_per_1m_input_tokens": JEV_PRICE_PER_1M_INPUT_TOKENS}),
    "jev-latest": (SystemOneModel, {"model": "jev-latest", "price_per_1m_input_tokens": JEV_PRICE_PER_1M_INPUT_TOKENS}),
    "kev-0.8b": _kev("0.8b"),
    "kev-4b": _kev("4b"),
    "kev-9b": _kev("9b"),
    # Laya's own HTTP server; "english" pins the checkpoint so it matches laya-english.
    "laya-serve": (SystemOneModel, {"model": "english", "server": "laya-serve", "deployment": "local"}),
    "systemone-custom": (_custom, {"model": None, "base_url": None}),
}
