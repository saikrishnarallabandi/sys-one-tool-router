# Sys One 101-class scale run — RUNBOOK

All commands run from the repo root: `/home2/srallaba/projects/system-one-tool-router`

Dataset (BUILT, CPU-only): `data/scale100/`
- 101 classes (100 canonical tools + `no_tool`), 17,041 items
- train 11,928 / val 2,556 / test 2,557, stratified 70/15/15, seed 42
- Built by `scripts/make_data_scale100.py` (same pipeline as `make_data.py`;
  top-100 canonical tools by frequency, min 74 raw examples at rank 100)

## GPU pipeline (run in this order, only when GPU is healthy)

```bash
cd /home2/srallaba/projects/system-one-tool-router

# 0. sanity: GPU must respond and stay < 90C
nvidia-smi --query-gpu=temperature.gpu --format=csv

# training venv (same one the 25-class run used)
VENV=/home2/srallaba/projects/system-one-poc/venv/bin/python

# 1. fine-tune (same config as the 25-class run: Qwen3-0.6B non-thinking,
#    fp32 frozen backbone + LoRA r=16 on q/k/v/o/gate/up/down + score head,
#    class-weighted CE, 3 epochs, lr 2e-4, batch 16, macro-F1 model selection)
$VENV scripts/train_scale100.py

# 2. eval + temperature scaling (fit on val only) -> exp/scale100/eval_report.json
$VENV scripts/eval_scale100.py

# 3. selective prediction / risk-coverage -> exp/scale100/selective_report.json
$VENV scripts/selective_analysis_scale100.py

# 4. latency (single-sample incl. tokenization + batch throughput)
$VENV scripts/bench_latency_scale100.py
```

## Generative baselines — SAI RUNS THESE HIMSELF

Staged scripts (DO NOT run as part of the Sys One pipeline):

```bash
cd /home2/srallaba/projects/system-one-tool-router

# zero-shot, full 101-tool list in prompt
$VENV scripts/gen_baseline_scale100.py        # -> exp/scale100/baseline_gen_report.json

# 3-shot (same 3 examples as the 25-class run), full 101-tool list
$VENV scripts/gen_baseline_fewshot_scale100.py  # -> exp/scale100/baseline_gen_fewshot_report.json
```

Prompt template (identical wording to the 25-class run, tool list swapped):

```
You are a tool router. Given the user message, output EXACTLY ONE tool name
from the list below, and nothing else.

Tools: <101 tool names, comma-separated, alphabetical>

User message: <text>
Tool:
```

3-shot prepends (unchanged):
```
Example 1:
User message: how much should I tip on a $45 bill
Tool: calculate_tip

Example 2:
User message: what is the capital of France
Tool: no_tool

Example 3:
User message: latest headlines on the election
Tool: get_news

```

Full 101-tool list (alphabetical, = label ids 0..100):
add_numbers, analyze_image, analyze_sentiment, calculate_age,
calculate_age_difference, calculate_area, calculate_average, calculate_bmi,
calculate_discount, calculate_distance, calculate_factorial, calculate_fibonacci,
calculate_fibonacci_sequence, calculate_gcd, calculate_gpa, calculate_gross_salary,
calculate_interest, calculate_loan_interest, calculate_loan_payment,
calculate_loan_repayment, calculate_mortgage_payment, calculate_percentage,
calculate_profit, calculate_rectangle_area, calculate_route, calculate_route_distance,
calculate_sales_tax, calculate_shipping_cost, calculate_square_root, calculate_tax,
calculate_tip, check_email_availability, check_flight_status, check_palindrome,
check_prime_number, check_spelling, check_word_count, convert_currency,
convert_temperature, count_words, create_account, create_calendar_event,
create_contact, create_note, create_playlist, create_reminder, create_task,
create_todo, create_todo_list, create_user, create_user_profile, encrypt_text,
execute_program, find_movie, find_nearby_places, find_nearby_restaurants,
find_nearest_gas_station, find_restaurants, find_shortest_path, generate_barcode,
generate_invoice, generate_password, generate_qr_code, generate_random_color,
generate_random_name, generate_random_number, generate_username, generate_uuid,
get_current_time, get_definition, get_lyrics, get_movie_details,
get_movie_recommendations, get_news, get_news_articles, get_random_fact,
get_random_joke, get_random_number, get_random_quote, get_recipe, get_song_lyrics,
get_stock_price, get_translation, no_tool, play_music, search_books, search_hotels,
search_images, search_jobs, search_movies, search_music, search_product,
search_recipes, search_restaurant, search_restaurants, search_song, send_email,
track_calories, track_expenses, track_package, translate_text

## Notes / known caveats

- Tail is thin by construction: 26 of 101 classes have <40 post-dedupe examples;
  smallest test supports are 1-3 (calculate_rectangle_area: 1). Per-class
  precision/recall/F1 for tail classes (test n<10) is noisy — read as indicative
  only. Aggregate metrics over n=2,557 test items are the Table 1 numbers.
- 1,133 same-text label conflicts resolved by majority vote (same as 25-class run).
- Label ids are alphabetical and DIFFER from the 25-class run (new head trained
  from scratch; no weight reuse).
- The 25-class dataset, checkpoints, and reports under data/, exp/ are untouched.
