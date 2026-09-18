"""Follow-up audit: why so few function calls? Inspect samples."""
import json, re, random

BASE = "/home2/srallaba/projects/system-one-tool-router"
items = json.load(open(f"{BASE}/data/glaive/raw/glaive-function-calling-v2.json"))
random.seed(7)

# 1. raw count of <functioncall> in the whole file text
raw = open(f"{BASE}/data/glaive/raw/glaive-function-calling-v2.json").read()
print("raw '<functioncall>' occurrences:", raw.count("<functioncall>"))
print("raw '<|endoftext|>' occurrences:", raw.count("<|endoftext|>"))

# 2. items containing <functioncall> anywhere in chat
FNCALL = re.compile(r"<functioncall>")
with_call = [it for it in items if FNCALL.search(it["chat"])]
print(f"items with <functioncall> in chat: {len(with_call)}")

# 3. sample 4 zero-call items: show first user turn + first assistant turn
zero = [it for it in items if not FNCALL.search(it["chat"])]
for i, it in enumerate(random.sample(zero, 4)):
    chat = it["chat"]
    u = chat.split("USER:")[1].split("ASSISTANT:")[0].strip()[:150]
    a = chat.split("ASSISTANT:")[1].split("<|endoftext|>")[0].strip()[:300]
    sysnames = re.findall(r'"name"\s*:\s*"([^"]+)"', it["system"][:2000])
    print(f"--- zero-call sample {i} | spec names in system head: {sysnames[:3]}")
    print("USER:", u.replace("\n", " "))
    print("ASST:", a.replace("\n", " ")[:300])

# 4. sample 2 with-call items: show user turn + the functioncall
for i, it in enumerate(random.sample(with_call, 2)):
    chat = it["chat"]
    u = chat.split("USER:")[1].split("ASSISTANT:")[0].strip()[:150]
    fc = FNCALL.search(chat)
    print(f"--- with-call sample {i}")
    print("USER:", u.replace("\n", " "))
    print("CALL:", chat[fc.start():fc.start()+220].replace("\n", " "))

# 5. item with >1 fn name in spec: inspect system prompt structure
NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
for it in items:
    m = it["system"].find("{")
    names = NAME_RE.findall(it["system"][m:])
    if len(set(names)) > 1:
        print("--- multi-name spec sample; distinct names:", sorted(set(names))[:6])
        print(it["system"][:600].replace("\n", " "))
        break
