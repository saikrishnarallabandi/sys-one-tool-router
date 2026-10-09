"""Render a self-contained HTML leaderboard from a decisionbench results directory.

    .venv/bin/python scripts/leaderboard.py results/smoke2-n40 [-o leaderboard.html]
"""
import argparse
import html
import json
import sys
from pathlib import Path

HEADLINE_LABEL = {
    "accuracy": "accuracy",
    "qwk": "QWK",
    "auroc": "AUROC",
    "decision_accuracy": "decision acc.",
}
TASK_ORDER = ["typed-decisions", "agnews", "emotion", "banking77", "sst5", "sst2", "boolq"]
# Same weights served over HTTP; kept out of the ranking, shown as a parity check.
PARITY_ALIASES = {"laya-serve": "laya-english"}
FAMILY = {"laya": "Laya", "kev": "Kev", "jev": "Jev", "nli": "Baseline", "open-jev": "Open-Jev"}


def family(name):
    for prefix, fam in sorted(FAMILY.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(prefix):
            return fam
    return "Other"


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def esc(s):
    return html.escape(str(s))


def load(results_dir):
    m = json.loads((results_dir / "metrics.json").read_text())
    tasks = [t for t in TASK_ORDER if t in m["tasks"]] + [t for t in m["tasks"] if t not in TASK_ORDER]
    return m, tasks


def contamination_marks(results_dir):
    """Cells the report marks with †/‡, recovered from report.md's headline cells."""
    marks = {}
    report = results_dir / "report.md"
    if not report.exists():
        return marks
    for line in report.read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[0] or cells[0].startswith(("task", "---", "rank", "model")):
            continue
        task = cells[0].split(" (")[0]
        model = cells[1]
        for c in cells[2:5]:
            if c.endswith(("†", "‡")) and "[" in c:
                marks[(task, model)] = c[-1]
    return marks


def build(results_dir):
    m, tasks = load(results_dir)
    summary = m.get("summary", {}).get("models", {})
    systems = m.get("systems", {})
    marks = contamination_marks(results_dir)
    models = [n for n in summary if n not in PARITY_ALIASES] or [
        n for n in systems if n not in PARITY_ALIASES
    ]
    # Models that ran only a few tasks (e.g. a checkpoint fine-tuned on one benchmark)
    # would top a mean-rank sort on a single task; rank them after full-coverage models.
    full = max((summary.get(n, {}).get("n_ranked") or 0 for n in models), default=0)

    def partial(n):
        return (summary.get(n, {}).get("n_ranked") or 0) < full / 2

    models.sort(key=lambda n: (partial(n), summary.get(n, {}).get("mean_rank") or 99))

    def cell(task, model):
        r = m["tasks"][task]["models"].get(model)
        if not r or r.get("headline") is None:
            return None
        v = r["metrics"].get(r["headline"])
        ci = r.get("headline_ci") or {}
        return {"v": v, "lo": ci.get("low"), "hi": ci.get("high"), "mark": marks.get((task, model), "")}

    # --- overall leaderboard rows
    rows = []
    for i, name in enumerate(models, 1):
        s = summary.get(name, {})
        sysm = systems.get(name, {})
        lat = (sysm.get("latency") or {}).get("p50_ms")
        rows.append(f"""
        <tr>
          <td class="rank">{'·' if partial(name) else i}</td>
          <td><div class="model">{esc(name)}</div><div class="sub">{esc(family(name))} · {esc(sysm.get('deployment', 'local'))} · {esc(sysm.get('device', ''))}{' · <b>specialist, ' + str(s.get('n_ranked')) + ' task</b>' if partial(name) else ''}</div></td>
          <td class="num strong">{fmt(s.get('mean_rank'), 2)}</td>
          <td class="num">{fmt(s.get('heldout_mean_headline'))}<span class="sub"> ({s.get('n_heldout', 0)})</span></td>
          <td class="num">{fmt(s.get('mean_ece_after_ts'))}</td>
          <td class="num">{fmt(lat, 1)}</td>
          <td class="num">$0 <span class="sub">local</span></td>
        </tr>""")
    for name, reason in (m.get("skipped") or {}).items():
        short = reason.split(";")[0].split("(")[0].strip()
        rows.append(f"""
        <tr class="pending">
          <td class="rank">–</td>
          <td><div class="model">{esc(name)}</div><div class="sub">{esc(family(name))} · not run</div></td>
          <td colspan="5" class="sub">{esc(short)}</td>
        </tr>""")

    # --- per-task matrix with bars + CI whiskers
    head = "".join(
        f'<th class="num">{esc(t)}<div class="sub">{esc(HEADLINE_LABEL.get(next((r.get("headline") for r in m["tasks"][t]["models"].values()), ""), ""))}</div></th>'
        for t in tasks
    )
    body = []
    for name in models:
        tds = []
        for t in tasks:
            c = cell(t, name)
            if c is None or c["v"] is None:
                tds.append('<td class="num sub">—</td>')
                continue
            best = max(
                (cell(t, o) or {}).get("v") or -1 for o in models
            )
            lo, hi = c["lo"] if c["lo"] is not None else c["v"], c["hi"] if c["hi"] is not None else c["v"]
            cls = "bar best" if abs(c["v"] - best) < 1e-9 else "bar"
            title = f"{name} on {t}: {c['v']:.3f}  95% CI [{lo:.2f}, {hi:.2f}]"
            if c["mark"]:
                title += "  — model trained on this dataset"
            tds.append(f"""<td class="num" title="{esc(title)}">
              <div class="val">{c['v']:.3f}{'<sup>' + c['mark'] + '</sup>' if c['mark'] else ''}</div>
              <div class="track"><div class="{cls}" style="width:{max(c['v'],0)*100:.1f}%"></div>
              <div class="ci" style="left:{lo*100:.1f}%;width:{max(hi-lo,0)*100:.1f}%"></div></div></td>""")
        body.append(f'<tr><td><div class="model">{esc(name)}</div></td>{"".join(tds)}</tr>')

    # --- robustness + systems
    rob = []
    for name in models:
        flips = [
            m["tasks"][t]["models"][name]["robustness"]["option_order"]["flip_rate"]
            for t in tasks
            if name in m["tasks"][t]["models"]
            and "option_order" in (m["tasks"][t]["models"][name].get("robustness") or {})
        ]
        neg = next(
            (
                m["tasks"][t]["models"][name]["robustness"]["negation"]
                for t in tasks
                if name in m["tasks"][t]["models"]
                and "negation" in (m["tasks"][t]["models"][name].get("robustness") or {})
            ),
            None,
        )
        b77 = (m["tasks"].get("banking77", {}).get("models", {}).get(name, {}).get("robustness") or {}).get("option_order", {})
        sysm = systems.get(name, {})
        lat = sysm.get("latency") or {}
        rob.append(f"""<tr><td><div class="model">{esc(name)}</div></td>
          <td class="num">{fmt(sum(flips)/len(flips) if flips else None)}</td>
          <td class="num">{fmt(b77.get('flip_rate'))}</td>
          <td class="num">{fmt(neg and neg['contradiction_rate'])}</td>
          <td class="num">{fmt(sysm.get('load_time_s'), 1)}</td>
          <td class="num">{fmt(lat.get('p50_ms'), 1)}</td>
          <td class="num">{fmt(lat.get('p95_ms'), 1)}</td>
          <td class="num">{fmt(lat.get('throughput_rps'), 1)}</td></tr>""")

    parity = []
    for alias, orig in PARITY_ALIASES.items():
        if alias in summary and orig in summary:
            same = all(
                (cell(t, alias) or {}).get("v") == (cell(t, orig) or {}).get("v") for t in tasks
            )
            parity.append(
                f"<li><b>{esc(alias)}</b> (HTTP /v1/systemone) vs <b>{esc(orig)}</b> (in-process): "
                f"{'identical headline metrics on every task ✓' if same else 'metrics DIFFER — check the protocol adapter'}; excluded from ranking.</li>"
            )

    n_ex = max((r.get("n_examples", 0) for t in tasks for r in m["tasks"][t]["models"].values()), default=0)
    top = models[0] if models else "—"
    return TEMPLATE.format(
        run=esc(results_dir.name),
        n=n_ex,
        n_models=len(models),
        n_pending=len(m.get("skipped") or {}),
        n_tasks=len(tasks),
        top=esc(top),
        rows="".join(rows),
        head=head,
        body="".join(body),
        rob="".join(rob),
        parity="".join(parity),
    )


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decision Model Leaderboard</title>
<style>
:root {{
  --bg:#fbfaf8; --panel:#ffffff; --ink:#1d1d1b; --muted:#6b6a66; --line:#e7e4de;
  --bar:#7c8db5; --best:#2f6f4f; --ci:#1d1d1b; --pending:#f3f1ec; --accent:#b4552d;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#161615; --panel:#1f1f1d; --ink:#ecebe7; --muted:#9b9993; --line:#33322f;
  --bar:#6f82b0; --best:#5fae84; --ci:#ecebe7; --pending:#252523; --accent:#e08a61; }} }}
:root[data-theme="dark"] {{ --bg:#161615; --panel:#1f1f1d; --ink:#ecebe7; --muted:#9b9993; --line:#33322f;
  --bar:#6f82b0; --best:#5fae84; --ci:#ecebe7; --pending:#252523; --accent:#e08a61; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.45 -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }}
h2 {{ font-size:15px; margin:32px 0 8px; }}
p.lede {{ color:var(--muted); margin:0 0 20px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.tile .k {{ color:var(--muted); font-size:12px; }} .tile .v {{ font-size:20px; font-weight:600; margin-top:2px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; }}
th, td {{ padding:9px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }}
tr:last-child td {{ border-bottom:0; }}
th {{ font-size:12px; font-weight:600; color:var(--muted); white-space:nowrap; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.rank {{ width:32px; color:var(--muted); font-weight:600; }}
.model {{ font-weight:600; white-space:nowrap; }} .sub {{ color:var(--muted); font-size:12px; font-weight:400; }}
.strong {{ font-weight:600; }}
tr.pending td {{ background:var(--pending); }}
.val {{ margin-bottom:4px; }} sup {{ color:var(--accent); }}
.track {{ position:relative; height:6px; background:var(--line); border-radius:3px; min-width:72px; }}
.bar {{ position:absolute; left:0; top:0; bottom:0; background:var(--bar); border-radius:3px; }}
.bar.best {{ background:var(--best); }}
.ci {{ position:absolute; top:-2px; height:10px; border-left:1.5px solid var(--ci); border-right:1.5px solid var(--ci); opacity:.55; }}
td:first-child, th:first-child {{ position:sticky; left:0; background:var(--panel); z-index:1; }}
tr.pending td:first-child {{ background:var(--pending); }}
.card table.lb td:nth-child(2), .card table.lb th:nth-child(2) {{ position:sticky; left:32px; background:var(--panel); z-index:1; }}
.card table.lb tr.pending td:nth-child(2) {{ background:var(--pending); }}
@media (max-width:640px) {{
  body {{ font-size:13px; }}
  main {{ padding:16px 12px 40px; }}
  th, td {{ padding:7px 8px; }}
  .tiles {{ grid-template-columns:1fr 1fr; }}
  .tile .v {{ font-size:17px; }}
  .model {{ font-size:12.5px; }}
  .track {{ min-width:56px; }}
}}
.notes {{ color:var(--muted); font-size:12.5px; margin-top:8px; }}
.notes li {{ margin:3px 0; }}
</style></head>
<body><main>
<h1>Decision model leaderboard</h1>
<p class="lede">System One typed-decision models on shared items and prompts · run <code>{run}</code> · up to {n} examples per task</p>

<div class="tiles">
  <div class="tile"><div class="k">Top model (mean rank)</div><div class="v">{top}</div></div>
  <div class="tile"><div class="k">Models ranked</div><div class="v">{n_models}</div></div>
  <div class="tile"><div class="k">Awaiting run</div><div class="v">{n_pending}</div></div>
  <div class="tile"><div class="k">Tasks</div><div class="v">{n_tasks}</div></div>
</div>

<h2>Overall</h2>
<div class="card"><table class="lb">
  <tr><th></th><th>model</th><th class="num">mean rank ↓</th><th class="num">held-out headline ↑ (tasks)</th>
      <th class="num">ECE after TS ↓</th><th class="num">p50 ms</th><th class="num">$ / 1k decisions</th></tr>
  {rows}
</table></div>

<h2>Per task · headline metric with 95% bootstrap CI</h2>
<div class="card"><table>
  <tr><th>model</th>{head}</tr>
  {body}
</table></div>
<ul class="notes">
  <li><sup>†</sup>/<sup>‡</sup> the model trained on this dataset, so the cell is not a held-out result. Hover a cell for its CI.</li>
  <li>Green bar = best on that task. Whiskers = 95% bootstrap CI (1000 resamples).</li>
  {parity}
</ul>

<h2>Robustness &amp; systems</h2>
<div class="card"><table>
  <tr><th>model</th><th class="num">option-order flip rate ↓</th><th class="num">flip rate, 77 options ↓</th>
      <th class="num">negation contradiction ↓</th><th class="num">load s</th><th class="num">p50 ms</th>
      <th class="num">p95 ms</th><th class="num">req/s</th></tr>
  {rob}
</table></div>
<ul class="notes">
  <li>The NLI baseline scores each option independently, so its 0 flip rate is by construction, not a measured strength.</li>
  <li>Latency is warm, sequential, one request at a time on this Mac.</li>
</ul>
</main></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    if not (a.results_dir / "metrics.json").exists():
        sys.exit(f"no metrics.json in {a.results_dir}")
    out = a.out or a.results_dir / "leaderboard.html"
    out.write_text(build(a.results_dir))
    print(out)


if __name__ == "__main__":
    main()
