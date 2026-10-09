"""Task registry: each task is a fixed, seeded sample of a labelled evaluation split."""

from __future__ import annotations

import csv
import functools
import io
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from decisionbench.types import QuestionSpec, QuestionType

CACHE_DIR = Path(os.environ.get("DECISIONBENCH_CACHE", Path(__file__).resolve().parents[1] / ".cache" / "decisionbench"))


State = str | dict | list


@dataclass
class Example:
    id: str
    state: State
    questions: dict[str, QuestionSpec]  # all asked in one request
    gold: dict[str, int]  # per question: choice label index; score level; noul 1 = true, 0 = false
    # Full gold distributions aligned with each spec (noul: [P(false), P(true)]), when the dataset has them.
    gold_probs: dict[str, list[float]] = field(default_factory=dict)
    # Robustness probes: check name -> questions asked in one extra request (a subset, possibly altered).
    probes: dict[str, dict[str, QuestionSpec]] = field(default_factory=dict)
    group: str | None = None  # e.g. the typed-decisions workflow


def single(id: str, state: State, qid: str, spec: QuestionSpec, gold: int,
           probes: dict[str, QuestionSpec] | None = None) -> Example:
    """An example with one question; `probes` maps check name -> the altered question."""
    return Example(id, state, {qid: spec}, {qid: gold}, probes={k: {qid: v} for k, v in (probes or {}).items()})


@dataclass
class Task:
    name: str
    type: QuestionType | Literal["multi"]  # "multi": several typed questions per example
    source: str
    split: str
    question_id: str | None  # the single question's id; None for multi-question tasks
    # Returns every row of the split as dicts with keys id, gold and whatever build() needs.
    load_rows: Callable[[], list[dict]]
    build: Callable[[dict], Example]
    description: str = ""
    # Examples used when --n is not given; None means the whole split.
    default_n: int | None = 300

    def load(self, n: int | None = None, seed: int = 0) -> list[Example]:
        """First `n` rows (default: `default_n`) of a seeded permutation of the split, so the sample
        for a smaller n is a prefix of the sample for a larger one."""
        rows = _cached_rows(self.name, self.load_rows)
        n = self.default_n if n is None else n
        order = np.random.default_rng(seed).permutation(len(rows))
        return [self.build(rows[i]) for i in order[: len(rows) if n is None else min(n, len(rows))]]


def _cached_rows(name: str, loader: Callable[[], list[dict]]) -> list[dict]:
    path = CACHE_DIR / f"{name}.jsonl"
    if path.exists():
        with path.open() as f:
            return [json.loads(line) for line in f]
    rows = loader()
    if not rows:
        raise RuntimeError(f"task {name!r}: loader returned no rows")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    return rows


def _hf_rows(repo: str, split: str, fields: dict[str, str], label: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    return [{"id": f"{split}-{i}", "gold": int(r[label]), **{k: r[src] for k, src in fields.items()}}
            for i, r in enumerate(ds)]


def _hf_label_names(repo: str, label: str = "label") -> list[str]:
    from datasets import load_dataset_builder

    return list(load_dataset_builder(repo).info.features[label].names)


def humanize(label: str) -> str:
    return label.replace("_", " ").strip().lower()


# ---------------------------------------------------------------- agnews
AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]  # ClassLabel order 0..3 in fancyzhx/ag_news
AGNEWS_DESC = {
    "World": "world news, international affairs, politics, conflicts",
    "Sports": "sports, games, athletes, teams, tournaments",
    "Business": "business, economy, companies, markets, finance",
    "Sci/Tech": "science and technology, computing, internet, space, research",
}
AGNEWS_Q = QuestionSpec("choice", "What is the topic of this news article?",
                        tuple((lab, AGNEWS_DESC[lab]) for lab in AGNEWS_LABELS))


def _agnews_rows():
    names = _hf_label_names("fancyzhx/ag_news")
    if names != AGNEWS_LABELS:
        raise RuntimeError(f"ag_news label names changed: {names}")
    return _hf_rows("fancyzhx/ag_news", "test", {"text": "text"}, "label")


# ---------------------------------------------------------------- emotion
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # dair-ai/emotion ClassLabel order
EMOTION_DESC = {
    "sadness": "feeling sad, down, hurt or lonely",
    "joy": "feeling happy, glad, content or excited",
    "love": "feeling love, affection, tenderness or longing",
    "anger": "feeling angry, irritated, annoyed or resentful",
    "fear": "feeling afraid, scared, anxious or nervous",
    "surprise": "feeling surprised, amazed, shocked or curious",
}
EMOTION_Q = QuestionSpec("choice", "Which emotion does the author express?",
                         tuple((lab, EMOTION_DESC[lab]) for lab in EMOTION_LABELS))


def _emotion_rows():
    names = _hf_label_names("dair-ai/emotion")
    if names != EMOTION_LABELS:
        raise RuntimeError(f"emotion label names changed: {names}")
    return _hf_rows("dair-ai/emotion", "test", {"text": "text"}, "label")


# ---------------------------------------------------------------- banking77
BANKING77_REPO = "PolyAI/banking77"
# The hub repo is script-only, which `datasets` >= 4 refuses to run. Label names come from the
# repo's dataset_infos.json (the ClassLabel features the script declares); rows come from the
# test.csv the script itself downloads.
BANKING77_TEST_CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"


@functools.lru_cache(maxsize=1)
def _banking77_names() -> tuple[str, ...]:
    from huggingface_hub import hf_hub_download

    info = json.loads(Path(hf_hub_download(BANKING77_REPO, "dataset_infos.json", repo_type="dataset")).read_text())
    return tuple(info["default"]["features"]["label"]["names"])


def _banking77_rows():
    names = _banking77_names()
    try:
        return _hf_rows(BANKING77_REPO, "test", {"text": "text"}, "label")
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" not in str(e):
            raise
    with urllib.request.urlopen(BANKING77_TEST_CSV, timeout=60) as resp:
        reader = csv.DictReader(io.StringIO(resp.read().decode("utf-8")))
        index = {n: i for i, n in enumerate(names)}
        rows = []
        for i, r in enumerate(reader):
            if r["category"] not in index:
                raise RuntimeError(f"banking77 row {i}: unknown category {r['category']!r}")
            rows.append({"id": f"test-{i}", "gold": index[r["category"]], "text": r["text"]})
    return rows


@functools.lru_cache(maxsize=1)
def _banking77_question() -> QuestionSpec:
    labels = [humanize(n) for n in _banking77_names()]
    return QuestionSpec("choice", "What is the customer asking the bank about?", tuple((lab, None) for lab in labels))


def _banking77_build(r):
    q = _banking77_question()
    return single(r["id"], r["text"], "intent", q, r["gold"], {"reversed": q.reversed()})


# ---------------------------------------------------------------- sst5
SST5_LEVELS = ("very negative", "negative", "neutral", "positive", "very positive")  # SetFit/sst5 label 0..4
SST5_Q = QuestionSpec("score", "How positive is the sentiment of this movie review?", SST5_LEVELS)


def _sst5_rows():
    rows = _hf_rows("SetFit/sst5", "test", {"text": "text", "label_text": "label_text"}, "label")
    bad = [r for r in rows if SST5_LEVELS[r["gold"]] != r["label_text"]]
    if bad:
        raise RuntimeError(f"sst5 label/label_text mismatch, e.g. {bad[0]}")
    return rows


# ---------------------------------------------------------------- sst2
SST2_POS = QuestionSpec("noul", "The text expresses positive sentiment.")
SST2_NEG = QuestionSpec("noul", "The text expresses negative sentiment.")


def _sst2_rows():
    names = _hf_label_names("stanfordnlp/sst2")
    if names != ["negative", "positive"]:
        raise RuntimeError(f"sst2 label names changed: {names}")
    return _hf_rows("stanfordnlp/sst2", "validation", {"text": "sentence"}, "label")


# ---------------------------------------------------------------- boolq
def _boolq_rows():
    return _hf_rows("google/boolq", "validation", {"passage": "passage", "question": "question"}, "answer")


def boolq_statement(question: str) -> str:
    q = question.strip().rstrip("?").strip()
    return f"The answer to the question '{q[:1].upper() + q[1:]}?' is yes."


def _boolq_build(r):
    return single(r["id"], r["passage"], "answer_yes", QuestionSpec("noul", boolq_statement(r["question"])), r["gold"])


# ---------------------------------------------------------------- typed-decisions
TYPED_DECISIONS_REPO = "LocalLLaMA/typed-decisions"
TYPED_DECISIONS_CONFIGS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")


def _typed_decisions_rows(split: str = "test") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(TYPED_DECISIONS_REPO, "all", split=split)
    rows = []
    for r in ds:
        if r["workflow"] not in TYPED_DECISIONS_CONFIGS:
            raise RuntimeError(f"typed-decisions: unknown workflow {r['workflow']!r}")
        rows.append({"id": r["id"], "workflow": r["workflow"], "state": json.loads(r["state"]),
                     "questions": json.loads(r["questions"]), "gold": json.loads(r["gold"])})
    return rows


def _td_spec(q: dict) -> QuestionSpec:
    crit = q.get("criteria")
    if q["type"] == "choice":
        return QuestionSpec("choice", q["instructions"], tuple(crit.items()))
    if q["type"] == "score":
        return QuestionSpec("score", q["instructions"], tuple(crit))
    return QuestionSpec("noul", q["instructions"], tuple((crit or {}).items()))


def _td_gold(spec: QuestionSpec, g: dict) -> tuple[int, list[float]]:
    """Gold index and gold distribution aligned with the spec. The dataset's `label` is the argmax of
    its gold distribution with ties already broken, so it is used as the discrete gold."""
    p = g["probabilities"]
    if spec.type == "choice":
        return spec.labels.index(g["label"]), [p[lab] for lab in spec.labels]
    if spec.type == "score":
        return int(g["label"]), [p[str(i)] for i in range(len(spec.criteria))]
    if g["label"] not in ("true", "false"):
        raise RuntimeError(f"typed-decisions noul label {g['label']!r}")
    return int(g["label"] == "true"), [p["false"], p["true"]]


def _typed_decisions_build(r: dict) -> Example:
    specs = {qid: _td_spec(q) for qid, q in r["questions"].items()}
    gold, gold_probs = {}, {}
    for qid, spec in specs.items():
        gold[qid], gold_probs[qid] = _td_gold(spec, r["gold"][qid])
    reversed_choices = {qid: s.reversed() for qid, s in specs.items() if s.type == "choice"}
    return Example(r["id"], r["state"], specs, gold, gold_probs,
                   probes={"reversed": reversed_choices} if reversed_choices else {}, group=r["workflow"])


def typed_decisions_reference(example_ids: list[str]) -> dict:
    """Input-free reference points on the given test cases, using the same discrete rule as models
    (argmax; noul p >= 0.5): 'prior' answers each question's label frequencies on the train split;
    'uniform' is the expected accuracy of a uniform guess (mean 1/K)."""
    test = {r["id"]: r for r in _cached_rows("typed-decisions", _typed_decisions_rows)}
    train = _cached_rows("typed-decisions-train", lambda: _typed_decisions_rows("train"))
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for r in train:
        for qid, g in r["gold"].items():
            c = counts.setdefault((r["workflow"], qid), {})
            c[g["label"]] = c.get(g["label"], 0) + 1
    prior_hits, uniform, by_group = [], [], {}
    for cid in example_ids:
        ex = _typed_decisions_build(test[cid])
        for qid, spec in ex.questions.items():
            c = counts[(ex.group, qid)]
            if spec.type == "choice":
                keys = spec.labels
            elif spec.type == "score":
                keys = [str(i) for i in range(len(spec.criteria))]
            else:
                keys = ["false", "true"]
            freq = np.array([c.get(k, 0) for k in keys], dtype=float)
            pred = int(freq[1] / freq.sum() >= 0.5) if spec.type == "noul" else int(np.argmax(freq))
            hit = pred == ex.gold[qid]
            prior_hits.append(hit)
            by_group.setdefault(ex.group, []).append(hit)
            uniform.append(1.0 / spec.n_options)
    return {"prior": float(np.mean(prior_hits)), "uniform": float(np.mean(uniform)),
            "prior_by_group": {g: float(np.mean(v)) for g, v in sorted(by_group.items())},
            "n_decisions": len(prior_hits)}


def _choice_build(qid: str, q: QuestionSpec):
    return lambda r: single(r["id"], r["text"], qid, q, r["gold"], {"reversed": q.reversed()})


TASKS: dict[str, Task] = {
    "agnews": Task("agnews", "choice", "fancyzhx/ag_news", "test", "topic", _agnews_rows, _choice_build("topic", AGNEWS_Q),
                   "news topic, 4 labels"),
    "emotion": Task("emotion", "choice", "dair-ai/emotion", "test", "emotion", _emotion_rows,
                    _choice_build("emotion", EMOTION_Q), "tweet emotion, 6 labels"),
    "banking77": Task("banking77", "choice", BANKING77_REPO, "test", "intent", _banking77_rows, _banking77_build,
                      "banking intent, 77 labels (high-cardinality)"),
    "sst5": Task("sst5", "score", "SetFit/sst5", "test", "sentiment", _sst5_rows,
                 lambda r: single(r["id"], r["text"], "sentiment", SST5_Q, r["gold"]), "5-level ordinal sentiment"),
    "sst2": Task("sst2", "noul", "stanfordnlp/sst2", "validation", "positive", _sst2_rows,
                 lambda r: single(r["id"], r["text"], "positive", SST2_POS, r["gold"], {"negation": SST2_NEG}),
                 "binary sentiment as a noul statement"),
    "boolq": Task("boolq", "noul", "google/boolq", "validation", "answer_yes", _boolq_rows, _boolq_build,
                  "passage yes/no question as a noul statement"),
    "typed-decisions": Task("typed-decisions", "multi", f"{TYPED_DECISIONS_REPO} (all)", "test", None,
                            _typed_decisions_rows, _typed_decisions_build,
                            "4 workflows x 5 typed questions per case, asked in one request", default_n=None),
}


def get_tasks(spec: str) -> list[Task]:
    names = list(TASKS) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise KeyError(f"unknown task(s) {unknown}; available: {', '.join(TASKS)}")
    return [TASKS[n] for n in names]
