"""Which models were trained on which benchmark datasets.

Two kinds of overlap are tagged:
  direct  (†) the model was trained on the task's own dataset (a different split of it).
  corpus  (‡) the model was trained on a different dataset built from the same source texts
          (SST-2 and SST-5 are both the Stanford Sentiment Treebank, whose sentences come from the
          Pang & Lee `rotten_tomatoes` sentence-polarity set).
A family mapped to None has undisclosed training data; its cells are marked "?" and not tagged.
"""

from __future__ import annotations

# Canonical dataset keys. Hub ids that count as each key: ag_news = fancyzhx/ag_news; banking77 =
# PolyAI/banking77, legacy-datasets/banking77, mteb/banking77; emotion = dair-ai/emotion; sst5 =
# SetFit/sst5; sst2 = stanfordnlp/sst2; boolq = google/boolq; typed-decisions = LocalLLaMA/typed-decisions.
CONTAMINATION: dict[str, dict | None] = {
    # https://github.com/NandhaKishorM/laya README, "English tasks" table: AG News and BoolQ
    # "in training mix"; DAIR Emotion and SST-5 "held out". Banking77 / SST-2 are not mentioned.
    "laya": {"datasets": {"ag_news", "boolq"},
             "source": "https://github.com/NandhaKishorM/laya#english-tasks"},
    # https://huggingface.co/convaiinnovations/laya: laya-typed-decisions is "the checkpoint fine-tuned
    # on that benchmark's own training split", on top of the English checkpoint.
    "laya-typed-decisions": {"datasets": {"ag_news", "boolq", "typed-decisions"}, "specialist": {"typed-decisions"},
                             "source": "https://huggingface.co/convaiinnovations/laya"},
    # `datasets:` field of https://huggingface.co/jaredpalmer/kev-0.8b, -4b, -9b (identical lists).
    "kev": {"datasets": {"banking77", "boolq", "ag_news", "multi_nli", "sst5", "yelp_review_full", "trec",
                         "dbpedia_14", "amazon_reviews_multi_en", "imdb"},
            "source": "https://huggingface.co/jaredpalmer/kev-4b"},
    # `datasets:` field and "Training data" table of the model card.
    "open-jev": {"datasets": {"banking77", "sst5", "boolq"},
                 "source": "https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large"},
    # The non "-c" zeroshot-v2.0 models add every dataset with used_in_v1.1 == TRUE in
    # https://github.com/MoritzLaurer/zeroshot-classifier/blob/7f82e4ab88d7aa82a4776f161b368cc9fa778001/v1_human_data/datasets_overview.csv
    # (sst2 is listed there as excluded; rotten_tomatoes is included).
    "nli-zeroshot": {"datasets": {"banking77", "emotion", "ag_news", "rotten_tomatoes", "imdb", "yelp_review_full",
                                  "amazon_polarity", "multi_nli", "anli", "wanli", "fever_nli", "massive",
                                  "yahoo_answers_topics", "financial_phrasebank", "app_reviews"},
                     "source": "https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0#training-data"},
    # TypeSafe does not publish Jev's training data.
    "jev": None,
}

# Model-name prefix -> family, first match wins.
FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("laya-typed-decisions", "laya-typed-decisions"),
    ("laya", "laya"),
    ("kev-", "kev"),
    ("open-jev", "open-jev"),
    ("nli-zeroshot", "nli-zeroshot"),
    ("jev", "jev"),
)

# Task -> the dataset keys that make up the task (direct) and those sharing its source texts (corpus).
TASK_SOURCES: dict[str, dict[str, set[str]]] = {
    "agnews": {"direct": {"ag_news"}},
    "emotion": {"direct": {"emotion"}},
    "banking77": {"direct": {"banking77"}},
    "sst5": {"direct": {"sst5"}, "corpus": {"sst2", "rotten_tomatoes"}},
    "sst2": {"direct": {"sst2"}, "corpus": {"sst5", "rotten_tomatoes"}},
    "boolq": {"direct": {"boolq"}},
    "typed-decisions": {"direct": {"typed-decisions"}},
}

MARKS = {"direct": "†", "corpus": "‡", "unknown": "?"}


def family(model: str) -> str | None:
    for prefix, fam in FAMILY_PREFIXES:
        if model.startswith(prefix):
            return fam
    return None


def overlap(model: str, task: str) -> str | None:
    """'direct', 'corpus', 'unknown' (training data undisclosed or model not in the map), or None."""
    fam = family(model)
    if fam is None or CONTAMINATION.get(fam) is None or task not in TASK_SOURCES:
        return "unknown"
    seen = CONTAMINATION[fam]["datasets"]
    src = TASK_SOURCES[task]
    if seen & src.get("direct", set()):
        return "direct"
    if seen & src.get("corpus", set()):
        return "corpus"
    return None


def disclosed(model: str) -> bool:
    fam = family(model)
    return fam is not None and CONTAMINATION.get(fam) is not None


def is_specialist(model: str, task: str) -> bool:
    """Fine-tuned on the benchmark's own training split (not comparable with zero-shot models)."""
    fam = family(model)
    entry = CONTAMINATION.get(fam) if fam else None
    return bool(entry and task in entry.get("specialist", set()))


def mark(model: str, task: str) -> str:
    kind = overlap(model, task)
    return MARKS.get(kind, "") if kind else ""


def source(model: str) -> str | None:
    fam = family(model)
    entry = CONTAMINATION.get(fam) if fam else None
    return entry["source"] if entry else None
