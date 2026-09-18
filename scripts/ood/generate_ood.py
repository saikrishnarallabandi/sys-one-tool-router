"""Generate the Sys One OOD challenge set with phi4 via local ollama.

One prompt per (category, tool-batch); strict JSON output; validation and
retry. Misspellings are applied programmatically to category-1 items.
Writes ood/ood_challenge.jsonl and ood/ood_generation_log.json.
"""
import json, os, random, re, urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "phi4:latest"
OUT = "/home2/srallaba/projects/system-one-tool-router/exp/ood"

TOOLS = {
    "analyze_sentiment": "detect positive / negative / neutral sentiment of text",
    "calculate_age": "compute age from a birthdate",
    "calculate_area": "area of geometric shapes",
    "calculate_bmi": "body-mass index from height and weight",
    "calculate_discount": "sale price after a discount",
    "calculate_distance": "distance between two places",
    "calculate_loan_payment": "monthly payment for a loan",
    "calculate_mortgage_payment": "monthly mortgage payment",
    "calculate_tip": "restaurant tip amount",
    "convert_currency": "convert money between currencies",
    "create_calendar_event": "add an event to the calendar",
    "create_todo": "add a task to a to-do list",
    "generate_password": "create a random secure password",
    "generate_qr_code": "make a QR code",
    "generate_random_number": "pick a random number",
    "get_definition": "define a word or term",
    "get_movie_details": "details about a specific movie",
    "get_news": "latest news headlines",
    "get_stock_price": "current price of a stock",
    "no_tool": "no tool applies (chit-chat, opinions, unanswerable)",
    "search_books": "find books",
    "search_movies": "discover/search movies",
    "search_recipes": "find recipes",
    "send_email": "compose and send an email",
    "translate_text": "translate text between languages",
}
ROUTING_TOOLS = [t for t in TOOLS if t != "no_tool"]

ANTI_TEMPLATE = (
    "Rules (follow all):\n"
    "- Sound like real humans: texts, voice-assistant commands, search boxes. "
    "Vary register, length, and person.\n"
    "- NEVER use template openers like 'Hi, I saw', 'Can you tell me', "
    "'I need to', 'Could you please'.\n"
    "- No two examples may share a sentence skeleton; vary verbs and structure.\n"
    "- Keep each example to one message (one or two sentences).\n"
    "- Use the EXACT tool names given."
)

FEWSHOTS = {
    "paraphrase": [
        ("calculate_tip", "What's 20% on top of $86?"),
        ("get_definition", "my kid asked what 'photosynthesis' means and I blanked"),
        ("convert_currency", "is 50 euros a lot for dinner in Lisbon"),
    ],
    "indirect": [
        ("calculate_age", "born March 1991 -- so how old am I exactly?"),
        ("create_todo", "gotta remember to pick up the dry cleaning tomorrow"),
        ("get_stock_price", "wondering if I should finally sell my Tesla shares"),
    ],
    "terse": [
        ("calculate_bmi", "bmi 70kg 178cm"),
        ("translate_text", "'gracias' in english"),
        ("search_recipes", "easy weeknight pasta"),
    ],
    "followup": [
        ("get_news",
         "User: who won the world series last year\\nAssistant: The Dodgers won.",
         "and who was the mvp"),
        ("search_books",
         "User: find me a good sci-fi book\\nAssistant: Try Project Hail Mary.",
         "anything by the same author"),
    ],
    "implicit": [
        ("get_definition", "reading this contract and 'indemnification' keeps coming up"),
        ("create_calendar_event", "dentist said come back in six months and I'll definitely forget"),
        ("send_email", "need to get my resume in front of the hiring manager by Friday"),
    ],
    "distractor": [
        ("calculate_distance", "no need to book anything -- just how far is the airport from downtown"),
        ("no_tool", "don't give me a recipe, I just want to know if avocados are healthy"),
        ("get_movie_details", "not searching for anything new, just the runtime of Dune 2"),
    ],
    "ambiguous": [
        ("calculate_mortgage_payment",
         ["calculate_mortgage_payment", "calculate_loan_payment"],
         "what's my monthly payment on a $400k home loan at 6.5%"),
        ("search_movies",
         ["search_movies", "get_movie_details"],
         "find me that new Scorsese movie"),
    ],
    "no_tool": [
        "do you think pineapple belongs on pizza",
        "i'm feeling kind of down today, everything's grey",
        "tell me something interesting about octopuses",
    ],
}

CATEGORIES = {
    "paraphrase": {"per_tool": 10, "desc": "natural, independent paraphrases of the intent"},
    "indirect": {"per_tool": 5, "desc": "indirect requests: state a situation/need, never name the action or tool"},
    "terse": {"per_tool": 4, "desc": "terse 2-6 word queries, like a search box or voice command"},
    "followup": {"per_tool": 2, "desc": "conversational follow-ups that depend on 1-2 turns of prior context"},
    "implicit": {"per_tool": 3, "desc": "implicit intent: the need is implied by the situation, never requested"},
    "distractor": {"per_tool": 2, "desc": "distractors: contain keywords of a DIFFERENT tool; read past them to the true intent (which may be no_tool)"},
    "ambiguous": {"per_tool": 2, "desc": "genuinely borderline between two tools; give primary and acceptable labels"},
}

def call_ollama(prompt, num_predict=3000):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "format": "json", "options": {"temperature": 0.85,
                       "num_predict": num_predict}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["response"]


def extract_array(text):
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.S)
        obj = json.loads(m.group(0))
    if isinstance(obj, dict):
        for k in ("examples", "items", "data"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        raise ValueError("no list in: " + text[:200])
    return obj


def tool_block(tools):
    return "\n".join("- %s: %s" % (t, TOOLS[t]) for t in tools)


def fewshot_block(cat):
    if cat == "ambiguous":
        return "\n".join(
            '  {"text": "%s", "primary": "%s", "acceptable": %s}' % (t, p, json.dumps(a))
            for p, a, t in FEWSHOTS[cat])
    if cat == "followup":
        return "\n".join(
            '  {"context": "%s", "text": "%s", "tool": "%s"}' % (c, t, tool)
            for tool, c, t in FEWSHOTS[cat])
    if cat == "no_tool":
        return "\n".join('  {"text": "%s", "tool": "no_tool"}' % t for t in FEWSHOTS[cat])
    return "\n".join(
        '  {"text": "%s", "tool": "%s"}' % (t, tool) for tool, t in FEWSHOTS[cat])


def build_prompt(cat, tools, per_tool):
    n = per_tool * len(tools)
    schema = '{"text": "<utterance>", "tool": "<exact_tool_name>"}'
    extra = ""
    if cat == "followup":
        schema = ('{"context": "User: ...\\\\nAssistant: ...", '
                  '"text": "<follow-up>", "tool": "<exact_tool_name>"}')
        extra = "- The follow-up alone should be ambiguous; with the context the intent is clear.\n"
    if cat == "ambiguous":
        schema = '{"text": "<utterance>", "primary": "<best tool>", "acceptable": ["<tool1>", "<tool2>"]}'
        extra = "- Pick pairs that are genuinely confusable.\n"
    if cat == "distractor":
        extra = "- About half should resolve to no_tool (the decoy wins: nothing applies).\n"
    return (
        "You are building a test set for a tool-routing classifier. The classifier routes "
        "a user message to exactly one of these tools:\n" + tool_block(tools) + "\n\n"
        "Write %d %s -- %d for EACH of these tools: " % (n, CATEGORIES[cat]["desc"], per_tool)
        + ", ".join(tools) + ".\n\n" + ANTI_TEMPLATE + "\n" + extra + "\n"
        "Few-shot examples (match this style, not the content):\n" + fewshot_block(cat) + "\n\n"
        "Output ONLY a JSON array of %d objects like %s." % (n, schema)
    )


def validate(cat, items, tools, per_tool):
    problems, good, counts = [], [], {}
    for it in items:
        if cat == "ambiguous":
            t, p, a = it.get("text"), it.get("primary"), it.get("acceptable")
            ok = (t and p in TOOLS and isinstance(a, list) and len(a) == 2
                  and all(x in TOOLS for x in a) and p in a)
            key = p
        else:
            t, tool = it.get("text"), it.get("tool")
            ctx = it.get("context", "")
            ok = (t and tool in TOOLS and (cat != "followup" or ctx))
            key = tool
        if not ok:
            problems.append("bad item: %s" % json.dumps(it)[:150])
            continue
        counts[key] = counts.get(key, 0) + 1
        good.append(it)
    for t in tools:
        if counts.get(t, 0) < per_tool:
            problems.append("short: %s got %d/%d" % (t, counts.get(t, 0), per_tool))
    return good, problems


def gen_category(cat, tools, per_tool, log):
    prompt = build_prompt(cat, tools, per_tool)
    for attempt in range(3):
        try:
            raw = call_ollama(prompt)
            items = extract_array(raw)
            good, problems = validate(cat, items, tools, per_tool)
            if not problems:
                log.append({"category": cat, "tools": tools, "attempt": attempt,
                            "n": len(good), "status": "ok"})
                return good
            log.append({"category": cat, "tools": tools, "attempt": attempt,
                        "status": "retry", "problems": problems[:5]})
        except Exception as e:
            log.append({"category": cat, "tools": tools, "attempt": attempt,
                        "status": "error", "error": str(e)[:200]})
    log.append({"category": cat, "tools": tools, "status": "FAILED"})
    return []


KEY_NEIGHBORS = {"a": "sq", "e": "wr", "i": "uo", "o": "ip", "t": "ry", "n": "bm"}


def typo(word):
    if len(word) < 4:
        return word
    op = random.choice(["swap", "drop", "double", "neighbor"])
    i = random.randrange(1, len(word) - 1)
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    c = word[i].lower()
    rep = random.choice(KEY_NEIGHBORS.get(c, "e"))
    return word[:i] + (rep.upper() if word[i].isupper() else rep) + word[i + 1:]


def add_typos(text, rng):
    words = text.split()
    idx = [i for i, w in enumerate(words)
           if len(w) >= 4 and w.isalpha() and not w[0].isupper()]
    if not idx:
        idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return text
    for i in rng.sample(idx, k=min(len(idx), rng.choice([1, 2]))):
        words[i] = typo(words[i])
    return " ".join(words)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(20260917)
    log, challenge = [], []
    batches = [ROUTING_TOOLS[i:i + 5] for i in range(0, len(ROUTING_TOOLS), 5)]
    for cat, spec in CATEGORIES.items():
        for b in batches:
            good = gen_category(cat, b, spec["per_tool"], log)
            for it in good:
                if cat == "ambiguous":
                    challenge.append({"text": it["text"], "label": it["primary"],
                                      "acceptable": it["acceptable"],
                                      "category": "ambiguous", "context": None,
                                      "source": "phi4"})
                else:
                    challenge.append({"text": it["text"], "label": it["tool"],
                                      "acceptable": [it["tool"]],
                                      "category": cat,
                                      "context": it.get("context"),
                                      "source": "phi4"})
            print("cat=%s batch=%d n=%d" % (cat, len(challenge), len(good)), flush=True)
    # no_tool standalone
    prompt = (
        "Write 75 short, varied human messages that NO tool can handle: chit-chat, "
        "opinions, feelings, jokes, personal advice, trivia questions. "
        "Tool list for reference (none of these apply):\n" + tool_block(ROUTING_TOOLS)
        + "\n\n" + ANTI_TEMPLATE + "\nFew-shot examples:\n" + fewshot_block("no_tool")
        + '\n\nOutput ONLY a JSON array of 75 objects like {"text": "<message>", "tool": "no_tool"}.'
    )
    for attempt in range(3):
        try:
            items = extract_array(call_ollama(prompt, num_predict=6000))
            good = [it for it in items
                    if it.get("text") and it.get("tool") == "no_tool"][:75]
            if len(good) >= 70:
                for it in good:
                    challenge.append({"text": it["text"], "label": "no_tool",
                                      "acceptable": ["no_tool"],
                                      "category": "no_tool", "context": None,
                                      "source": "phi4"})
                log.append({"category": "no_tool", "n": len(good), "status": "ok"})
                break
        except Exception as e:
            log.append({"category": "no_tool", "attempt": attempt, "status": "error",
                        "error": str(e)[:200]})
    # misspellings: programmatic typo noise on paraphrases
    paras = [c for c in challenge if c["category"] == "paraphrase"]
    for c in rng.sample(paras, k=min(75, len(paras))):
        challenge.append({"text": add_typos(c["text"], rng), "label": c["label"],
                          "acceptable": [c["label"]], "category": "misspelled",
                          "context": None, "source": "typo-noise"})
    with open(os.path.join(OUT, "ood_challenge.jsonl"), "w") as f:
        for c in challenge:
            f.write(json.dumps(c) + "\n")
    json.dump(log, open(os.path.join(OUT, "ood_generation_log.json"), "w"), indent=2)
    print("TOTAL %d examples" % len(challenge))


if __name__ == "__main__":
    main()
