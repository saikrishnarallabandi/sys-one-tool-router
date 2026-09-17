#!/usr/bin/env python3
"""Build the tool-routing dataset from glaiveai/glaive-function-calling-v2 (apache-2.0).

Pipeline:
  1. Stream all ~113k rows; split each chat into turns.
  2. Walk turns: every ASSISTANT turn containing <functioncall> yields
     (the immediately preceding USER utterance, tool name).
     Rows with no functioncall yield (last USER utterance, "no_tool").
  3. Normalize tool names via SYNONYMS, keep the top TOP_K canonical tools.
  4. Dedupe exact (text, label) pairs; cap per-class counts; stratified 70/15/15 split.

Output: data_train.jsonl, data_val.jsonl, data_test.jsonl, label_map.json,
        data_stats.json
"""
import json, re, collections, random
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET = "glaiveai/glaive-function-calling-v2"
TOP_K = 24
PER_TOOL_CAP = 700
NOTOOL_CAP = 2000
SEED = 42
OUT_DIR = "."

# variant -> canonical tool name (same intent, different name in the wild)
SYNONYMS = {
    "calculate_mortgage": "calculate_mortgage_payment",
    "generate_random_password": "generate_password",
    "search_recipe": "search_recipes",
    "create_event": "create_calendar_event",
    "schedule_meeting": "create_calendar_event",
    "get_news_headlines": "get_news",
    "search_news": "get_news",
    "search_movie": "search_movies",
    "search_book": "search_books",
    "get_random_quote": "get_random_quote",
    "generate_random_quote": "get_random_quote",
    "generate_quote": "get_random_quote",
    "get_quote": "get_random_quote",
    "get_quote_of_the_day": "get_random_quote",
    "get_random_quote_of_the_day": "get_random_quote",
    "get_daily_quote": "get_random_quote",
    "generate_random_joke": "get_random_joke",
    "get_joke": "get_random_joke",
    "generate_random_fact": "get_random_fact",
    "generate_random_username": "generate_username",
    "get_time": "get_current_time",
    "calculate_discounted_price": "calculate_discount",
    "create_invoice": "generate_invoice",
    "create_user_account": "create_user",
}
NO_TOOL = "no_tool"

name_pat = re.compile(r'"name"\s*:\s*"([^"]+)"')
turn_pat = re.compile(r'^(USER|ASSISTANT):\s*(.*?)(?=^USER:|^ASSISTANT:|\Z)',
                      re.M | re.S)

def split_turns(chat):
    turns = []
    for m in turn_pat.finditer(chat):
        role, text = m.group(1), m.group(2)
        text = text.replace("<|endoftext|>", "").strip()
        # collapse the blank-line padding between turns
        text = re.sub(r"\n{2,}", "\n", text).strip()
        turns.append((role, text))
    return turns

def extract_rows():
    ds = load_dataset(DATASET, split="train", streaming=True)
    call_rows, nocall_rows, skipped = 0, 0, 0
    examples = []  # (text, raw_tool)
    n = 0
    for row in ds:
        n += 1
        chat = row.get("chat") or ""
        turns = split_turns(chat)
        if not turns:
            skipped += 1
            continue
        last_user = None
        emitted_call = False
        for role, text in turns:
            if role == "USER":
                last_user = text
            elif "<functioncall>" in text:
                m = name_pat.search(text)
                if m and last_user:
                    examples.append((last_user, m.group(1).strip().lower()))
                    emitted_call = True
                    call_rows += 1
        if not emitted_call:
            # natural no-call: last user utterance the assistant answered w/o a tool
            users = [t for r, t in turns if r == "USER"]
            if users and users[-1]:
                examples.append((users[-1], NO_TOOL))
                nocall_rows += 1
            else:
                skipped += 1
        if n % 20000 == 0:
            print(f"... {n} rows, {len(examples)} examples", flush=True)
    print(f"rows: {n}, call examples: {call_rows}, no_tool: {nocall_rows}, skipped: {skipped}")
    return examples

def normalize(name):
    return SYNONYMS.get(name, name)

def main():
    random.seed(SEED)
    examples = extract_rows()

    # normalize + count
    normed = [(t, normalize(nm)) for t, nm in examples]
    counts = collections.Counter(nm for _, nm in normed if nm != NO_TOOL)
    print(f"distinct canonical tools: {len(counts)}")
    top_tools = [nm for nm, _ in counts.most_common(TOP_K)]
    print("top 24 after normalization:")
    for nm in top_tools:
        print(f"  {nm}: {counts[nm]}")
    keep = set(top_tools) | {NO_TOOL}

    # filter, dedupe (text, label), resolve text->label conflicts by majority
    seen = {}
    conflicts = 0
    for text, nm in normed:
        if nm not in keep:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text or len(text) > 600:
            continue
        key = text.lower()
        if key not in seen:
            seen[key] = [text, collections.Counter([nm])]
        else:
            seen[key][1][nm] += 1
            if nm != seen[key][1].most_common(1)[0][0]:
                conflicts += 1
    print(f"unique texts: {len(seen)}, label conflicts on same text: {conflicts}")

    by_class = collections.defaultdict(list)
    for text, counter in seen.values():
        nm = counter.most_common(1)[0][0]
        by_class[nm].append(text)

    # cap per class (deterministic shuffle first)
    final = []
    for nm, texts in by_class.items():
        random.shuffle(texts)
        cap = NOTOOL_CAP if nm == NO_TOOL else PER_TOOL_CAP
        for t in texts[:cap]:
            final.append((t, nm))
    random.shuffle(final)
    print(f"total after capping: {len(final)}")
    for nm in sorted(by_class):
        kept = min(len(by_class[nm]), NOTOOL_CAP if nm == NO_TOOL else PER_TOOL_CAP)
        print(f"  {nm}: raw {len(by_class[nm])} -> kept {kept}")

    # label ids: alphabetical, deterministic
    labels = sorted(set(nm for _, nm in final))
    label2id = {nm: i for i, nm in enumerate(labels)}
    id2label = {i: nm for nm, i in label2id.items()}
    with open(f"{OUT_DIR}/label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()},
                   "num_labels": len(labels)}, f, indent=2)
    print(f"classes ({len(labels)}): {labels}")

    X = [t for t, _ in final]
    y = [label2id[nm] for _, nm in final]
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=SEED, stratify=ytmp)

    stats = {}
    for name, Xs, ys in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        with open(f"{OUT_DIR}/data_{name}.jsonl", "w") as f:
            for t, l in zip(Xs, ys):
                f.write(json.dumps({"text": t, "label": l, "tool": id2label[l]}) + "\n")
        c = collections.Counter(id2label[l] for l in ys)
        stats[name] = {"n": len(ys), "per_class": dict(sorted(c.items()))}
        print(f"{name}: {len(ys)}")
    with open(f"{OUT_DIR}/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
