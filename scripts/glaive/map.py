"""Map Glaive v2 items to the Sys One 25-class catalog.

Reads glaive_items.jsonl (built by scripts/glaive/build.py), applies an explicit audited
mapping: lowercase Glaive function name -> our tool class, and writes back
gt_classes + mappable per item. Seeds glaive_report.json with coverage stats.

Mapping bar (same as BFCL): only clear semantic overlaps. Synthetic-but-nearby
functions (tax, shipping, invoices, music, restaurants, flights, time/date,
random quotes/facts/jokes, image search, etc.) are left unmapped.

Label rules:
- item calls >=1 function: mappable iff EVERY called name maps; gt = sorted set
  of mapped classes (nearly always exactly one).
- item has a function spec but no call (assistant declined / answered directly):
  gt = ["no_tool"], mappable = True.
- item has no function spec at all (plain chit-chat, no routing decision):
  mappable = False (excluded from routing eval, counted in audit).
"""
import json
from collections import Counter

BASE = "/home2/srallaba/projects/system-one-tool-router"

# Explicit audited mapping: Glaive function name -> our class.
MAPPING = {
    # calculate_age
    "calculate_age": "calculate_age",
    # calculate_area (geometric area only)
    "calculate_area": "calculate_area",
    "calculate_rectangle_area": "calculate_area",
    # calculate_bmi
    "calculate_bmi": "calculate_bmi",
    # calculate_discount
    "calculate_discount": "calculate_discount",
    "calculate_discounted_price": "calculate_discount",
    # calculate_distance (point-to-point / route distance; routing-as-navigation excluded)
    "calculate_distance": "calculate_distance",
    "calculate_route_distance": "calculate_distance",
    # calculate_loan_payment
    "calculate_loan_payment": "calculate_loan_payment",
    "calculate_loan_repayment": "calculate_loan_payment",
    # calculate_mortgage_payment
    "calculate_mortgage_payment": "calculate_mortgage_payment",
    "calculate_mortgage": "calculate_mortgage_payment",
    # calculate_tip (tip computation incl. splitting)
    "calculate_tip": "calculate_tip",
    "calculate_tip_split": "calculate_tip",
    # convert_currency
    "convert_currency": "convert_currency",
    "get_exchange_rate": "convert_currency",
    # create_calendar_event
    "create_calendar_event": "create_calendar_event",
    "schedule_meeting": "create_calendar_event",
    "create_event": "create_calendar_event",
    # create_todo
    "create_todo": "create_todo",
    "create_task": "create_todo",
    # generate_password (generate_random_password is the same function, renamed)
    "generate_password": "generate_password",
    "generate_random_password": "generate_password",
    # generate_qr_code
    "generate_qr_code": "generate_qr_code",
    "generate_qrcode": "generate_qr_code",
    # generate_random_number
    "generate_random_number": "generate_random_number",
    "get_random_number": "generate_random_number",
    # get_definition
    "get_definition": "get_definition",
    # get_movie_details
    "get_movie_details": "get_movie_details",
    # get_news (headlines/articles/search all news retrieval)
    "get_news": "get_news",
    "get_news_headlines": "get_news",
    "get_news_articles": "get_news",
    "search_news": "get_news",
    # get_stock_price
    "get_stock_price": "get_stock_price",
    "check_stock_price": "get_stock_price",
    # search_books
    "search_books": "search_books",
    "search_book": "search_books",
    # search_movies
    "search_movies": "search_movies",
    "search_movie": "search_movies",
    # search_recipes
    "search_recipes": "search_recipes",
    "search_recipe": "search_recipes",
    "get_recipe": "search_recipes",
    # send_email
    "send_email": "send_email",
    # analyze_sentiment
    "analyze_sentiment": "analyze_sentiment",
    # translate_text
    "translate_text": "translate_text",
    "get_translation": "translate_text",
}

rows = json.load(open(f"{BASE}/data/glaive/items.jsonl"))
mapped_fn = Counter()
unmapped_fn = Counter()
n_notestdata = 0
n_unmapped_call = 0
for r in rows:
    called = [c.lower() for c in r["called_names"]]
    if called:
        if all(c in MAPPING for c in called):
            r["gt_classes"] = sorted({MAPPING[c] for c in called})
            r["mappable"] = True
            for c in called:
                mapped_fn[c] += 1
        else:
            r["gt_classes"] = []
            r["mappable"] = False
            n_unmapped_call += 1
            for c in called:
                (mapped_fn if c in MAPPING else unmapped_fn)[c] += 1
    elif r["has_spec"]:
        r["gt_classes"] = ["no_tool"]
        r["mappable"] = True
    else:
        r["gt_classes"] = []
        r["mappable"] = False
        n_notestdata += 1

json.dump(rows, open(f"{BASE}/data/glaive/items.jsonl", "w"))

n_map = sum(1 for r in rows if r["mappable"])
n_test_map = sum(1 for r in rows if r["mappable"] and r["split"] == "test")
n_train_map = sum(1 for r in rows if r["mappable"] and r["split"] == "train")
per_class = Counter()
for r in rows:
    if r["mappable"]:
        for c in r["gt_classes"]:
            per_class[c] += 1

report = {
    "dataset": "glaiveai/glaive-function-calling-v2",
    "n_items": len(rows),
    "note": ("No official train/test split exists on the hub; split here is a "
             "hash-based 5% held-out test constructed for this eval. "
             "Items with no function spec are plain chit-chat, excluded from routing eval."),
    "coverage": {
        "n_mappable": n_map,
        "n_test_mappable": n_test_map,
        "n_train_mappable": n_train_map,
        "n_excluded_no_spec": n_notestdata,
        "n_excluded_unmapped_call": n_unmapped_call,
        "mapped_distinct_functions": len(mapped_fn),
        "unmapped_distinct_functions": len(unmapped_fn),
        "top_unmapped_functions": [n for n, _ in unmapped_fn.most_common(20)],
        "per_class_mappable": dict(per_class.most_common()),
    },
}
json.dump(report, open(f"{BASE}/exp/glaive/report.json", "w"), indent=2)
print(json.dumps(report["coverage"], indent=2))
