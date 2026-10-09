import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from decisionbench import contamination as C
from decisionbench import metrics as M
from decisionbench import report, run, serve, tasks
from decisionbench.models import create
from decisionbench.models.base import DecisionModel
from decisionbench.models.open_jev import OpenJevModel, to_open_jev
from decisionbench.models.systemone import SystemOneModel
from decisionbench.types import ModelUnavailable, QuestionSpec, choice_answer, noul_answer, score_answer

CHOICE = QuestionSpec("choice", "Which department?", (("billing", "invoices"), ("technical", None), ("sales", None)))
SCORE = QuestionSpec("score", "How urgent?", ("not urgent", "soon", "critical"))
NOUL_DESC = QuestionSpec("noul", "Needs a human.", (("false", "No human needed."), ("true", "A human should look.")))


# ---------------------------------------------------------------- a fake /v1/systemone server
class FakeSystemOne(BaseHTTPRequestHandler):
    run_id = "jaredpalmer/kev-4b"
    require_key: str | None = None
    requests: list = []

    def log_message(self, *a):
        pass

    def _auth_ok(self):
        return self.require_key is None or self.headers.get("Authorization") == f"Bearer {self.require_key}"

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("x-typesafe-request-id", "req-1")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"models": [{"name": "kev-latest", "description": "d", "release_date": "2026-09-01",
                                         "run": self.run_id, "backend": "mlx", "dtype": "bfloat16"}]})
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        if not self._auth_ok():
            return self._send(401, {"detail": "missing or invalid API key"})
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(body)
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                p = [0.7] + [0.3 / (len(keys) - 1)] * (len(keys) - 1)
                answers[qid] = {"type": "choice", "choice": keys[0], "confidence": 0.5,
                                "probabilities": dict(zip(keys, p))}
            elif q["type"] == "score":
                k = len(q["criteria"])
                answers[qid] = {"type": "score", "score": float(k - 1), "confidence": 0.9,
                                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                "probabilities": {str(i): (1.0 if i == k - 1 else 0.0) for i in range(k)}}
            else:
                answers[qid] = {"type": "noul", "noul": 0.25}
        self._send(200, {"model": body.get("model"), "answers": answers,
                         "usage": {"input_tokens": 42, "output_tokens": 7}, "latency_ms": 12.5})


@pytest.fixture
def fake_server():
    handler = type("H", (FakeSystemOne,), {"requests": [], "require_key": None, "run_id": "jaredpalmer/kev-4b"})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    httpd.shutdown()
    httpd.server_close()


def test_systemone_against_local_http_server(fake_server):
    url, handler = fake_server
    m = SystemOneModel("kev-latest", base_url=url, deployment="local", expect_run="jaredpalmer/kev-4b", max_retries=0)
    out = m.predict({"ticket": "charged twice"}, {"d": CHOICE.reversed(), "u": SCORE, "n": NOUL_DESC})
    # Wire format: criteria order preserved, noul outcome descriptions sent, JSON state passed through.
    body = handler.requests[-1]
    assert list(body["questions"]["d"]["criteria"]) == ["sales", "technical", "billing"]
    assert body["questions"]["n"]["criteria"] == {"false": "No human needed.", "true": "A human should look."}
    assert body["state"] == {"ticket": "charged twice"} and body["model"] == "kev-latest"
    # Answers aligned to the spec order, with provenance.
    assert out["d"].probs.tolist() == pytest.approx([0.7, 0.15, 0.15])
    assert out["u"].pred == pytest.approx(2.0)
    assert out["n"].probs == pytest.approx(0.25)
    assert out["d"].input_tokens == 42
    assert out["d"].extra == {"served_model": "kev-latest", "server_latency_ms": 12.5}
    info = m.info()
    assert info["models"][0]["run"] == "jaredpalmer/kev-4b" and info["models"][0]["backend"] == "mlx"
    assert m.device == "local server (mlx)"
    m.close()


def test_systemone_refuses_wrong_checkpoint(fake_server):
    url, handler = fake_server
    handler.run_id = "runs/smoke"  # Kev's silent fallback when --run is not found
    with pytest.raises(ModelUnavailable, match="runs/smoke"):
        SystemOneModel("kev-latest", base_url=url, deployment="local", expect_run="jaredpalmer/kev-4b")


def test_systemone_bad_key_is_unavailable(fake_server):
    url, handler = fake_server
    handler.require_key = "secret"
    m = SystemOneModel("kev-latest", base_url=url, api_key="wrong", deployment="local", max_retries=0)
    with pytest.raises(ModelUnavailable, match="credentials"):
        m.predict("x", {"n": NOUL_DESC})


def test_systemone_custom_needs_endpoint():
    with pytest.raises(ModelUnavailable, match="systemone-custom needs"):
        create("systemone-custom")


def test_systemone_custom_via_opts(fake_server, monkeypatch):
    url, _ = fake_server
    monkeypatch.setenv("MY_KEY", "k")
    m = create("systemone-custom", base_url=url, model="sd-1", api_key_env="MY_KEY", price_per_1m_input_tokens=0.03)
    assert m.predict("x", {"n": NOUL_DESC})["n"].extra["served_model"] == "sd-1"
    assert m.deployment == "api" and m.price_per_1m_input_tokens == 0.03


# ---------------------------------------------------------------- server lifecycle
FAKE_SERVER_SCRIPT = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        data = json.dumps({"status": "ok"}).encode()
        self.send_response(200); self.send_header("content-length", str(len(data))); self.end_headers()
        self.wfile.write(data)
print("fake server up", flush=True)
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_managed_server_starts_waits_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "LOG_DIR", tmp_path)
    script = tmp_path / "srv.py"
    script.write_text(FAKE_SERVER_SCRIPT)
    port = _free_port()
    spec = serve.ServerSpec("fake", (sys.executable, str(script), str(port)), port, "/health", ready_timeout_s=30)
    s = serve.ManagedServer(spec).start()
    assert s.owned and s.healthy()
    # A second manager finds it healthy and reuses it without owning it.
    reuse = serve.ManagedServer(spec).start()
    assert not reuse.owned
    reuse.stop()
    assert s.healthy()
    s.stop()
    assert not s.healthy()


def test_managed_server_reports_missing_install_and_early_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "LOG_DIR", tmp_path)
    port = _free_port()
    missing = serve.ServerSpec("kev-x", ("nope",), port, "/v1/models", requires=(tmp_path / "no-venv",), setup="do X")
    with pytest.raises(ModelUnavailable, match="(?s)not installed.*do X"):
        serve.ManagedServer(missing).start()
    crash = serve.ServerSpec("crash", (sys.executable, "-c", "print('boom'); raise SystemExit(3)"), port, "/health")
    with pytest.raises(ModelUnavailable, match="(?s)exited with code 3.*boom"):
        serve.ManagedServer(crash).start()


def test_kev_specs_never_install():
    for name in ("kev-0.8b", "kev-4b", "kev-9b"):
        spec = serve.SERVERS[name]
        assert spec.cmd[0] == str(serve.KEV_PYTHON) and "uv" not in spec.cmd
        assert spec.cmd[spec.cmd.index("--run") + 1] == f"jaredpalmer/{name}"
    ports = [s.port for s in serve.SERVERS.values()]
    assert len(ports) == len(set(ports))


# ---------------------------------------------------------------- open-jev
class FakeOpenJev:
    def __init__(self, max_questions=10):
        self.calls, self.max_questions, self.device, self.config = [], max_questions, "cpu", {"temperature": 1.05}

    def decide(self, state, questions):
        self.calls.append(len(questions))
        if len(questions) > self.max_questions:
            raise ValueError("sequence 900 > max_len 512: state + 3 questions")
        out = []
        for q in questions:
            if q["type"] == "noul":
                out.append({"noul": 0.8})
            else:
                n = len(q["options"])
                p = [1.0 / n] * n
                out.append({"probabilities": dict(zip(q["options"], p)), "confidence": 1.0 / n})
        return out


def _fake_module(model):
    return SimpleNamespace(OpenJev=SimpleNamespace(from_pretrained=lambda repo, device=None, revision=None: model))


def test_open_jev_refuses_hub_code_by_default():
    with pytest.raises(ModelUnavailable, match="allow_hub_code"):
        OpenJevModel()


def test_open_jev_mapping_and_packing_fallback():
    assert to_open_jev(CHOICE)["options"] == ["billing: invoices", "technical", "sales"]
    fake = FakeOpenJev()
    m = OpenJevModel(module=_fake_module(fake))
    out = m.predict({"a": 1}, {"d": CHOICE, "u": SCORE, "n": NOUL_DESC})
    assert fake.calls == [3]
    assert out["d"].probs.tolist() == pytest.approx([1 / 3] * 3)
    assert out["n"].probs == pytest.approx(0.8) and out["d"].extra["packing"] == "packed"
    fake2 = FakeOpenJev(max_questions=1)
    out2 = OpenJevModel(module=_fake_module(fake2)).predict("s", {"d": CHOICE, "u": SCORE, "n": NOUL_DESC})
    assert fake2.calls == [3, 1, 1, 1] and out2["u"].extra["packing"] == "per-question"


# ---------------------------------------------------------------- typed-decisions
TD_ROW = {
    "id": "c1", "workflow": "customer_service", "state": {"thread": "hi"},
    "questions": {
        "action": {"type": "choice", "instructions": "Action?", "criteria": {"reply": "Answer it.", "refund": "Refund."}},
        "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
        "needs_human": {"type": "noul", "instructions": "Needs a human.",
                        "criteria": {"false": "No.", "true": "Yes."}},
    },
    "gold": {
        "action": {"type": "choice", "label": "refund", "probabilities": {"reply": 0.5, "refund": 0.5}},
        "urgency": {"type": "score", "label": "2", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
        "needs_human": {"type": "noul", "label": "true", "noul": 0.6, "probabilities": {"false": 0.4, "true": 0.6}},
    },
}


def test_typed_decisions_build_aligns_gold_and_probes():
    ex = tasks._typed_decisions_build(TD_ROW)
    assert ex.group == "customer_service" and ex.state == {"thread": "hi"}
    assert ex.questions["action"].labels == ["reply", "refund"]
    assert ex.questions["needs_human"].noul_criteria == {"false": "No.", "true": "Yes."}
    # Ties are broken by the dataset's label, not by position.
    assert ex.gold == {"action": 1, "urgency": 2, "needs_human": 1}
    assert ex.gold_probs["needs_human"] == [0.4, 0.6]
    assert list(ex.probes) == ["reversed"] and list(ex.probes["reversed"]) == ["action"]
    assert ex.probes["reversed"]["action"].labels == ["refund", "reply"]


def test_multi_question_metrics_hand_checked():
    items = [
        {"case": "a", "group": "g1", "question": "g1/c", "type": "choice", "probs": [0.2, 0.8], "gold": 1,
         "gold_probs": [0.5, 0.5]},
        {"case": "a", "group": "g1", "question": "g1/s", "type": "score", "probs": [0.0, 0.5, 0.5], "gold": 2},
        {"case": "b", "group": "g2", "question": "g2/n", "type": "noul", "probs": [0.7], "gold": 1},
        {"case": "b", "group": "g2", "question": "g2/n2", "type": "noul", "probs": [0.4], "gold": 1},
    ]
    m = M.multi_question_metrics(items)
    # hits: choice yes; score argmax=1 (first of tie) != 2 no; noul 0.7 yes; noul 0.4 no.
    assert m["decision_accuracy"] == pytest.approx(0.5)
    assert m["accuracy_by_group"] == {"g1": 0.5, "g2": 0.5}
    assert m["score_mae_expected"] == pytest.approx(0.5)
    assert m["noul_brier"] == pytest.approx((0.09 + 0.36) / 2)
    assert m["kl_from_gold"] == pytest.approx(0.5 * np.log(0.5 / 0.2) + 0.5 * np.log(0.5 / 0.8))
    hits, counts = M.case_accuracy_arrays(items)
    assert hits.tolist() == [1, 1] and counts.tolist() == [2, 2]


def test_typed_decisions_reference_on_cached_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "CACHE_DIR", tmp_path)
    train = [dict(TD_ROW, id=f"t{i}", gold={**TD_ROW["gold"], "action": {**TD_ROW["gold"]["action"], "label": "reply"}})
             for i in range(3)]
    (tmp_path / "typed-decisions.jsonl").write_text(json.dumps(TD_ROW) + "\n")
    (tmp_path / "typed-decisions-train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    ref = tasks.typed_decisions_reference(["c1"])
    # prior answers reply (wrong), urgency 2 (right), needs_human true (right)
    assert ref["prior"] == pytest.approx(2 / 3)
    assert ref["uniform"] == pytest.approx((1 / 2 + 1 / 3 + 1 / 2) / 3)


# ---------------------------------------------------------------- contamination
def test_contamination_map():
    assert C.overlap("kev-4b", "agnews") == "direct"
    assert C.overlap("kev-4b", "sst2") == "corpus"
    assert C.overlap("kev-4b", "emotion") is None
    assert C.overlap("nli-zeroshot", "emotion") == "direct"
    assert C.overlap("nli-zeroshot", "sst2") == "corpus"
    assert C.overlap("laya-english", "boolq") == "direct" and C.overlap("laya-english", "emotion") is None
    assert C.overlap("laya-typed-decisions", "typed-decisions") == "direct"
    assert C.is_specialist("laya-typed-decisions", "typed-decisions") and not C.is_specialist("laya-english", "typed-decisions")
    assert C.overlap("jev", "agnews") == "unknown" and not C.disclosed("jev")
    assert C.overlap("open-jev-deberta", "banking77") == "direct"
    assert C.overlap("kev-4b", "some-new-task") == "unknown"


def _res(models_by_task):
    return {"tasks": {t: {"type": "choice", "models": {m: {"headline": "accuracy", "metrics": {"accuracy": v}}
                                                         for m, v in ms.items()}}
                      for t, ms in models_by_task.items()},
            "systems": {m: {} for ms in models_by_task.values() for m in ms}}


def test_heldout_tasks_derived_from_map():
    res = _res({"emotion": {"laya-english": 0.6, "kev-4b": 0.7, "laya-typed-decisions": 0.5},
                "agnews": {"laya-english": 0.9, "kev-4b": 0.95},
                "typed-decisions": {"laya-english": 0.4, "kev-4b": 0.6, "laya-typed-decisions": 0.77}})
    s = report.summarize(res)
    # agnews is dirty for both; the specialist does not remove typed-decisions from the held-out set.
    assert s["heldout"]["tasks"] == ["emotion", "typed-decisions"]
    assert s["heldout"]["specialists"] == ["laya-typed-decisions"]
    assert s["heldout"]["rows"]["kev-4b"]["mean"] == pytest.approx(0.65)
    # ranks: emotion 1, agnews 1, typed-decisions 2 (behind the specialist)
    assert s["models"]["kev-4b"]["mean_rank"] == pytest.approx(4 / 3)
    assert s["models"]["laya-english"]["heldout_mean_headline"] == pytest.approx(0.5)
    res2 = _res({"emotion": {"nli-zeroshot": 0.8, "kev-4b": 0.7}})
    assert report.summarize(res2)["heldout"]["excluded"] == {"emotion": ["nli-zeroshot"]}


# ---------------------------------------------------------------- runner: multi-question task, old-format files
class FakeMulti(DecisionModel):
    deployment = "local"

    def __init__(self):
        self.calls = 0

    @property
    def device(self):
        return "cpu"

    def predict(self, state, questions):
        self.calls += 1
        out = {}
        for qid, s in questions.items():
            if s.type == "choice":
                out[qid] = choice_answer(s, {lab: (0.9 if lab == "refund" else 0.1 / (len(s.labels) - 1)) for lab in s.labels},
                                         input_tokens=100)
            elif s.type == "score":
                out[qid] = score_answer(s, {i: 1.0 / len(s.criteria) for i in range(len(s.criteria))}, input_tokens=100)
            else:
                out[qid] = noul_answer(0.9, input_tokens=100)
        return out


def test_runner_multi_question_task(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "CACHE_DIR", tmp_path / "cache")
    rows = [dict(TD_ROW, id=f"c{i}") for i in range(12)]
    task = tasks.Task("typed-decisions", "multi", "fake", "test", None, lambda: rows, tasks._typed_decisions_build,
                      default_n=None)
    made = []

    def fake_create(name, **kw):
        m = FakeMulti()
        m.name = name
        made.append(m)
        return m

    monkeypatch.setattr(run, "create", fake_create)
    import decisionbench.tasks as T
    monkeypatch.setattr(T, "typed_decisions_reference", lambda ids: {"prior": 0.5, "uniform": 0.4, "n_decisions": 3 * len(ids),
                                                                    "prior_by_group": {"customer_service": 0.5}})
    out = tmp_path / "res"
    run.run(["fake"], [task], n=None, seed=0, robustness=True, out=out, warmup=0, model_opts={}, retry_errors=False)
    assert made[0].calls == 1 + 12 * 2  # cold request + (case + reversed-choice probe) per case
    res = report.build(out)
    m = res["tasks"]["typed-decisions"]["models"]["fake"]
    x = m["metrics"]
    # action: refund wins (gold refund) -> hit; urgency uniform -> argmax 0 != 2; needs_human 0.9 -> hit.
    assert x["decision_accuracy"] == pytest.approx(2 / 3) and x["n_decisions"] == 36 and x["n_cases"] == 12
    assert m["decisions_per_request"] == 3
    assert m["robustness"]["option_order"]["n"] == 12
    md = (out / "report.md").read_text()
    assert "## typed-decisions" in md and "prior (train label frequencies" in md and "customer_service" in md
    # --n smaller than the split caps it; default takes the whole split.
    assert len(task.load(5)) == 5 and len(task.load()) == 12


def test_round1_records_are_upgraded_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "CACHE_DIR", tmp_path / "cache")
    q = QuestionSpec("choice", "Which?", (("a", None), ("b", None)))
    task = tasks.Task("fchoice", "choice", "fake", "test", "q",
                      lambda: [{"id": f"r{i}", "gold": i % 2, "text": "a"} for i in range(4)],
                      lambda r: tasks.single(r["id"], r["text"], "q", q, r["gold"], {"reversed": q.reversed()}))
    exs = task.load(4)
    out = tmp_path / "res"
    (out / "preds").mkdir(parents=True)
    meta = {"model": "old", "task": "fchoice", "task_type": "choice", "question_id": "q", "source": "fake",
            "split": "test", "n_requested": 4, "n_examples": 4, "seed": 0, "robustness": True, "model_opts": {},
            "example_ids": [e.id for e in exs], "n_options": 2}
    (out / "preds" / "old__fchoice.meta.json").write_text(json.dumps(meta))
    lines = [{"i": i, "id": e.id, "gold": e.gold["q"], "error": None, "latency_s": 0.01, "probs": [0.9, 0.1], "pred": 0,
              "input_tokens": 5, "raw": {}, "extra": {}, "probes": {"reversed": {"probs_aligned": [0.2, 0.8]}},
              "warm": True} for i, e in enumerate(exs)]
    (out / "preds" / "old__fchoice.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    keep, todo = run.plan_pair(out, "old", task, exs, 0, True, {}, False)
    assert todo == [] and keep[0]["answers"]["q"]["probs"] == [0.9, 0.1]
    res = report.build(out)
    m = res["tasks"]["fchoice"]["models"]["old"]
    assert m["n_valid"] == 4 and m["robustness"]["option_order"]["flip_rate"] == 1.0
