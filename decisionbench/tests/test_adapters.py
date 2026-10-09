import json
from types import SimpleNamespace

import httpx2
import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, TypeSafeAuthenticationError

from decisionbench import report, run, tasks
from decisionbench.models import registry
from decisionbench.models.base import DecisionModel
from decisionbench.models.systemone import SystemOneModel, from_sdk_response
from decisionbench.models.laya_model import from_laya_answers, to_laya_question
from decisionbench.models.nli import hypotheses
from decisionbench.types import (
    InvalidOutput, ModelUnavailable, QuestionSpec, choice_answer, noul_answer, score_answer,
)

CHOICE = QuestionSpec("choice", "Which department?", (("billing", "invoices"), ("technical", None), ("sales", None)))
SCORE = QuestionSpec("score", "How urgent?", ("not urgent", "soon", "critical"))
NOUL = QuestionSpec("noul", "The customer threatens to leave.")


# ---------------------------------------------------------------- normalization
def test_choice_answer_aligns_to_spec_order():
    a = choice_answer(CHOICE, {"sales": 0.1, "billing": 0.2, "technical": 0.7})
    assert a.probs.tolist() == pytest.approx([0.2, 0.7, 0.1])
    assert a.pred == 1


def test_choice_answer_renormalizes_rounding_drift():
    a = choice_answer(CHOICE, {"billing": 0.3334, "technical": 0.3334, "sales": 0.3334})
    assert a.probs.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("probs", [
    {"billing": 0.5, "technical": 0.5},                                  # missing label
    {"billing": 0.5, "technical": 0.3, "sales": 0.1, "other": 0.1},      # extra label
    {"billing": 0.9, "technical": 0.9, "sales": 0.1},                    # does not sum to 1
    {"billing": float("nan"), "technical": 0.5, "sales": 0.5},
    {"billing": -0.1, "technical": 0.6, "sales": 0.5},
])
def test_choice_answer_rejects_invalid(probs):
    with pytest.raises(InvalidOutput):
        choice_answer(CHOICE, probs)


def test_score_answer_accepts_int_and_str_keys_and_computes_expectation():
    for probs in ({0: 0.1, 1: 0.2, 2: 0.7}, {"0": 0.1, "1": 0.2, "2": 0.7}):
        a = score_answer(SCORE, probs)
        assert a.probs.tolist() == pytest.approx([0.1, 0.2, 0.7])
        assert a.pred == pytest.approx(1.6)


@pytest.mark.parametrize("probs", [{0: 0.5, 1: 0.5}, {"0": 0.2, "1": 0.3, "x": 0.5}, {0: 0.2, 1: 0.3, 3: 0.5},
                                   {0: 0.2, "0": 0.3, 1: 0.5}])
def test_score_answer_rejects_invalid(probs):
    with pytest.raises(InvalidOutput):
        score_answer(SCORE, probs)


def test_noul_answer_bounds():
    assert noul_answer(0.7).pred == pytest.approx(0.7)
    for bad in (1.2, -0.1, float("nan"), "x", None):
        with pytest.raises(InvalidOutput):
            noul_answer(bad)


def test_question_spec_validation_and_reverse():
    assert CHOICE.reversed().labels == ["sales", "technical", "billing"]
    with pytest.raises(ValueError):
        QuestionSpec("choice", "q", (("a", None), ("a", None)))
    with pytest.raises(ValueError):
        QuestionSpec("noul", "q", ("x",))
    with pytest.raises(ValueError):
        QuestionSpec("score", "q", ())


# ---------------------------------------------------------------- laya mapping
def test_laya_question_and_answer_mapping():
    assert to_laya_question(CHOICE)["criteria"] == {"billing": "invoices", "technical": None, "sales": None}
    assert to_laya_question(SCORE)["criteria"] == ["not urgent", "soon", "critical"]
    assert "criteria" not in to_laya_question(NOUL)
    result = {"answers": {
        "d": {"type": "choice", "choice": "technical",
              "probabilities": {"billing": 0.2, "technical": 0.7, "sales": 0.1}},
        "u": {"type": "score", "score": 1.6, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
        "c": {"type": "noul", "noul": 0.3},
    }, "usage": {"input_tokens": 140, "output_tokens": 0}}
    out = from_laya_answers({"d": CHOICE, "u": SCORE, "c": NOUL}, result)
    assert out["d"].pred == 1 and out["u"].pred == pytest.approx(1.6) and out["c"].probs == pytest.approx(0.3)
    assert out["d"].input_tokens == 140
    with pytest.raises(InvalidOutput):
        from_laya_answers({"d": CHOICE}, {"answers": {"d": {"type": "noul", "noul": 0.5}}})


# ---------------------------------------------------------------- systemone (Jev) mapping
def _jev_response():
    answers = {
        "d": ChoiceAnswer.model_validate_json(json.dumps(
            {"type": "choice", "choice": "billing", "confidence": 0.9,
             "probabilities": {"billing": 0.8, "technical": 0.15, "sales": 0.05}})),
        "u": ScoreAnswer.model_validate_json(json.dumps(
            {"type": "score", "score": 1.0, "confidence": 0.5, "legend": {"0": "a", "1": "b", "2": "c"},
             "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25}})),
        "c": NoulAnswer.model_validate_json(json.dumps({"type": "noul", "noul": 0.66})),
    }
    return SimpleNamespace(model="jev-1.13.0", usage=SimpleNamespace(input_tokens=55, output_tokens=0),
                           choices={"d": answers["d"]}, scores={"u": answers["u"]}, nouls={"c": answers["c"]})


def test_jev_response_mapping():
    out = from_sdk_response({"d": CHOICE, "u": SCORE, "c": NOUL}, _jev_response())
    assert out["d"].probs.tolist() == pytest.approx([0.8, 0.15, 0.05])
    assert out["u"].pred == pytest.approx(1.0)
    assert out["c"].probs == pytest.approx(0.66)
    assert out["d"].input_tokens == 55 and out["d"].extra["served_model"] == "jev-1.13.0"


def test_jev_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ModelUnavailable, match="TYPESAFE_API_KEY"):
        SystemOneModel("jev-1.13.0")


def test_jev_auth_failure_is_unavailable():
    class Client:
        def system_one(self, state, questions):
            raise TypeSafeAuthenticationError(401, {"error": "bad key"}, httpx2.Headers())

    with pytest.raises(ModelUnavailable, match="credentials"):
        SystemOneModel("jev-1.13.0", client=Client()).predict("x", {"c": NOUL})


def test_nli_hypotheses():
    h = hypotheses(CHOICE)
    assert h[0] == "Which department? Answer: billing (invoices)." and h[1] == "Which department? Answer: technical."
    assert hypotheses(NOUL) == ["The customer threatens to leave."]
    assert len(hypotheses(SCORE)) == 3


def test_registry_has_required_variants():
    reg = registry()
    for name in ("laya-english", "laya-multilingual", "laya-typed-decisions", "laya-router", "jev", "jev-latest",
                 "nli-zeroshot", "kev-0.8b", "kev-4b", "kev-9b", "laya-serve", "systemone-custom", "open-jev-deberta"):
        assert name in reg


def test_boolq_statement():
    assert tasks.boolq_statement("is the sky blue") == "The answer to the question 'Is the sky blue?' is yes."


# ---------------------------------------------------------------- runner end-to-end with a fake model
class FakeModel(DecisionModel):
    """Choice: puts 0.8 on the gold label of 'label-<k>' states; noul: returns a fixed p."""

    deployment = "local"

    def __init__(self, fail_every: int = 0):
        self.calls = 0
        self.fail_every = fail_every

    @property
    def device(self):
        return "cpu"

    def predict(self, state, questions):
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("boom")
        out = {}
        for qid, spec in questions.items():
            if spec.type == "choice":
                gold = spec.labels.index(state.split(":")[0])
                probs = {lab: (0.8 if j == gold else 0.2 / (len(spec.labels) - 1)) for j, lab in enumerate(spec.labels)}
                out[qid] = choice_answer(spec, probs, input_tokens=10)
            else:
                out[qid] = noul_answer(0.8 if "pos" in state else 0.3, input_tokens=10)
        return out


def _fake_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(tasks, "CACHE_DIR", tmp_path / "cache")
    q = QuestionSpec("choice", "Which?", (("a", None), ("b", None), ("c", None)))
    labels = ["a", "b", "c"]
    choice_rows = lambda: [{"id": f"r{i}", "gold": i % 3, "text": f"{labels[i % 3]}: item {i}"} for i in range(30)]
    noul_rows = lambda: [{"id": f"s{i}", "gold": i % 2, "text": ("pos" if i % 2 else "neg") + f" {i}"} for i in range(30)]
    neg = QuestionSpec("noul", "negative")
    t1 = tasks.Task("fchoice", "choice", "fake", "test", "q", choice_rows,
                    lambda r: tasks.single(r["id"], r["text"], "q", q, r["gold"], {"reversed": q.reversed()}))
    t2 = tasks.Task("fnoul", "noul", "fake", "test", "q", noul_rows,
                    lambda r: tasks.single(r["id"], r["text"], "q", QuestionSpec("noul", "positive"), r["gold"],
                                           {"negation": neg}))
    return [t1, t2]


def test_runner_end_to_end_resume_and_report(monkeypatch, tmp_path):
    ts = _fake_tasks(monkeypatch, tmp_path)
    made = []

    def fake_create(name, **kw):
        if name == "jev":
            raise ModelUnavailable("TYPESAFE_API_KEY is not set")
        m = FakeModel(**kw)
        m.name = name
        made.append(m)
        return m

    monkeypatch.setattr(run, "create", fake_create)
    out = tmp_path / "res"
    run.run(["fake", "jev"], ts, n=20, seed=0, robustness=True, out=out, warmup=2, model_opts={}, retry_errors=False)
    res = report.build(out)
    m = res["tasks"]["fchoice"]["models"]["fake"]
    assert m["n_valid"] == 20 and m["complete"]
    assert m["metrics"]["accuracy"] == 1.0
    assert m["robustness"]["option_order"]["flip_rate"] == 0.0
    assert m["robustness"]["option_order"]["mean_tv"] == pytest.approx(0.0)
    assert res["tasks"]["fnoul"]["models"]["fake"]["metrics"]["auroc"] == 1.0
    assert res["tasks"]["fnoul"]["models"]["fake"]["robustness"]["negation"]["contradiction_rate"] == 1.0
    assert res["skipped"] == {"jev": "TYPESAFE_API_KEY is not set"}
    assert res["systems"]["fake"]["usd_per_1k_decisions"] == 0.0
    assert res["systems"]["fake"]["latency"]["n"] == 2 * (20 - 2)
    md = (out / "report.md").read_text()
    assert "## Choice tasks" in md and "jev" in md and "TYPESAFE_API_KEY" in md

    # Resume: nothing to do, so no model is even constructed.
    made.clear()
    run.run(["fake"], ts, n=20, seed=0, robustness=True, out=out, warmup=2, model_opts={}, retry_errors=False)
    assert made == []

    # Growing n keeps the first 20 and only runs the new 5 (+ probes and the cold first request).
    run.run(["fake"], ts, n=25, seed=0, robustness=True, out=out, warmup=2, model_opts={}, retry_errors=False)
    assert len(made) == 1
    assert made[0].calls == 1 + 5 * 2 + 5 * 2
    res = report.build(out)
    assert res["tasks"]["fchoice"]["models"]["fake"]["n_valid"] == 25


def test_runner_records_errors_and_aborts_on_systematic_failure(monkeypatch, tmp_path):
    ts = _fake_tasks(monkeypatch, tmp_path)

    def flaky(name, **kw):
        m = FakeModel(fail_every=kw.get("fail_every", 4))
        m.name = name
        return m

    monkeypatch.setattr(run, "create", flaky)
    out = tmp_path / "res"
    run.run(["flaky"], ts[:1], n=20, seed=0, robustness=False, out=out, warmup=0, model_opts={}, retry_errors=False)
    res = report.build(out)
    m = res["tasks"]["fchoice"]["models"]["flaky"]
    assert m["complete"] and 0 < m["error_rate"] < 0.5 and m["n_valid"] < 20

    class Broken(FakeModel):
        def predict(self, state, questions):
            raise InvalidOutput("garbage")

    def broken(name, **kw):
        m = Broken()
        m.name = name
        return m

    monkeypatch.setattr(run, "create", broken)
    out2 = tmp_path / "res2"
    run.run(["broken"], ts[:1], n=20, seed=0, robustness=False, out=out2, warmup=0, model_opts={}, retry_errors=False)
    skipped = json.loads((out2 / "skipped.json").read_text())
    assert "first 5 requests" in skipped["broken"] and "garbage" in skipped["broken"]
    # The aborted session's failures are not left behind as reusable records.
    assert run.read_records(out2 / "preds" / "broken__fchoice.jsonl") == []
