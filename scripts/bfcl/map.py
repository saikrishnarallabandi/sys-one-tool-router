"""Map BFCL v3 eval items to the Sys One 25-class catalog.

Writes bfcl_items.jsonl (one row per item: split, id, text, gt_classes, mappable)
and seeds bfcl_report.json with mapping/coverage stats.
Mapping is an explicit audited dict: BFCL function name (lowercased) -> our class.
"""
import json
from collections import Counter

BASE = "/home2/srallaba/projects/system-one-tool-router"
DATA = f"{BASE}/data/bfcl/raw"

# Explicit audited mapping: lowercase BFCL function name -> our tool class.
# Built by reviewing all 716 distinct BFCL function names across the 5 splits.
# Deliberately conservative: circumference, routing/directions, weather,
# flight/hotel booking, etc. are left unmapped (no corresponding class).
MAPPING = {
    # calculate_bmi
    "calculate_bmi": "calculate_bmi",
    # calculate_area (geometric area only; circumference excluded)
    "calculate_triangle_area": "calculate_area",
    "geometry.area_triangle": "calculate_area",
    "calc_area_triangle": "calculate_area",
    "calculate_area": "calculate_area",
    "geometry.calculate_area_circle": "calculate_area",
    "area_circle.calculate": "calculate_area",
    "circle.calculate_area": "calculate_area",
    "math.circle_area": "calculate_area",
    "math.triangle_area_base_height": "calculate_area",
    "math.triangle_area_heron": "calculate_area",
    "rectangle.area": "calculate_area",
    "area_rectangle.calculate": "calculate_area",
    "geometry_rectangle.calculate": "calculate_area",
    "geometry_circle.calculate": "calculate_area",
    "geometry_square.calculate": "calculate_area",
    # calculate_distance (point-to-point distance; routing/directions excluded)
    "calculate_distance": "calculate_distance",
    "geo_distance.calculate": "calculate_distance",
    "calculate_shortest_distance": "calculate_distance",
    "get_shortest_driving_distance": "calculate_distance",
    "geodistance.find": "calculate_distance",
    "distance_calculator.calculate": "calculate_distance",
    "calc_distance": "calculate_distance",
    "euclideandistance.calculate": "calculate_distance",
    "city_distance.find_shortest": "calculate_distance",
    "maps.get_distance_duration": "calculate_distance",
    "kinematics.calculate_displacement": "calculate_distance",
    "kinematics.distance": "calculate_distance",
    "kinematics.distance_traveled": "calculate_distance",
    # convert_currency
    "currency_exchange.convert": "convert_currency",
    "currency_conversion": "convert_currency",
    "currency_conversion.convert": "convert_currency",
    "convert_currency": "convert_currency",
    "currency_converter": "convert_currency",
    "get_exchange_rate": "convert_currency",
    "latest_exchange_rate": "convert_currency",
    "get_exchange_rate_with_fee": "convert_currency",
    "currency_conversion.get_rate": "convert_currency",
    # send_email
    "send_email": "send_email",
    # get_news
    "get_news": "get_news",
    "news": "get_news",
    # get_stock_price
    "get_stock_price": "get_stock_price",
    "stock_price": "get_stock_price",
    "get_stock_info": "get_stock_price",
    "get_stock_prices": "get_stock_price",
    "get_stock_data": "get_stock_price",
    "avg_closing_price": "get_stock_price",
    "stock_market_data": "get_stock_price",
    # translate_text
    "translate_text": "translate_text",
    "translate": "translate_text",
    # search_recipes
    "recipe_search": "search_recipes",
    "find_recipe": "search_recipes",
    "recipe.find": "search_recipes",
    "recipe_search.find": "search_recipes",
    "get_recipe": "search_recipes",
    "recipe_finder.find": "search_recipes",
    "find_recipes": "search_recipes",
    "get_vegan_recipe": "search_recipes",
    # search_books
    "library.search_book": "search_books",
    "library.search_books": "search_books",
    "google.books_search": "search_books",
    "openlibrary.books_search": "search_books",
    # search_movies / get_movie_details
    "imdb.find_movies_by_actor": "search_movies",
    "movie_details.brief": "get_movie_details",
    # mortgage / loan
    "calculate_mortgage_payment": "calculate_mortgage_payment",
    "finance.loan_repayment": "calculate_loan_payment",
    # analyze_sentiment
    "sentiment_analysis": "analyze_sentiment",
    "text_analysis.sentiment_analysis": "analyze_sentiment",
}

SPLITS = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]

def user_text(question):
    # question: list of conversations; each conversation is a list of messages
    parts = []
    for conv in question:
        for msg in conv:
            if msg.get("role") == "user":
                parts.append(msg.get("content", ""))
    return "\n".join(parts).strip()

def main():
    items = []
    gt_count = Counter()
    mapped_fn = Counter()
    unmapped_fn = Counter()
    for split in SPLITS:
        qrows = [json.loads(l) for l in open(f"{DATA}/BFCL_v3_{split}.json")]
        pa = {}
        if split != "irrelevance":
            for l in open(f"{DATA}/pa_BFCL_v3_{split}.json"):
                r = json.loads(l)
                pa[r["id"]] = r["ground_truth"]
        for q in qrows:
            qid = q["id"]
            text = user_text(q["question"])
            if split == "irrelevance":
                gt_names, gt_classes = [], ["no_tool"]
            else:
                gt_names = []
                for call in pa.get(qid, []):
                    gt_names.extend(list(call.keys()))
                gt_classes = []
                ok = True
                for n in gt_names:
                    m = MAPPING.get(n.lower())
                    if m is None:
                        ok = False
                    else:
                        gt_classes.append(m)
                gt_count.update(gt_names)
                for n in gt_names:
                    (mapped_fn if n.lower() in MAPPING else unmapped_fn)[n] += 1
                if not gt_names:
                    ok = False
            mappable = (split == "irrelevance") or (ok and len(gt_names) > 0)
            items.append({"split": split, "id": qid, "text": text,
                          "gt_names": gt_names, "gt_classes": sorted(set(gt_classes)),
                          "mappable": mappable})
    json.dump(items, open(f"{BASE}/data/bfcl/items.jsonl", "w"))
    # coverage stats
    by_split = Counter()
    map_by_split = Counter()
    for it in items:
        by_split[it["split"]] += 1
        if it["mappable"]:
            map_by_split[it["split"]] += 1
    report = {
        "mapping": {
            "n_bfcl_functions_mapped": len(set(MAPPING)),
            "n_bfcl_functions_total": len(set(list(mapped_fn) + list(unmapped_fn))),
            "mapped_function_counts": dict(mapped_fn.most_common()),
            "top_unmapped_functions": [n for n, _ in unmapped_fn.most_common(25)],
            "n_unmapped_distinct": len(unmapped_fn),
            "note": "Conservative explicit mapping; circumference, routing/directions, "
                    "weather, booking, and 700+ domain-specific functions left unmapped.",
        },
        "coverage": {
            split: {"n_total": by_split[split], "n_mappable": map_by_split[split],
                    "frac": round(map_by_split[split] / max(1, by_split[split]), 4)}
            for split in SPLITS
        },
    }
    json.dump(report, open(f"{BASE}/exp/bfcl/report.json", "w"), indent=2)
    print(json.dumps(report["coverage"], indent=2))
    print("mapped distinct:", len(set(MAPPING)), "| unmapped distinct:", len(unmapped_fn))

if __name__ == "__main__":
    main()
