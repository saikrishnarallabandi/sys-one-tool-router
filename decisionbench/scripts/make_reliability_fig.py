"""Reliability diagrams for the td-delta 4B and 9B runs (paper Figure 2).

    uv run --no-project --with matplotlib --with numpy python scripts/make_reliability_fig.py

Reads the saved sweep rows (results/sweep-8way/sweep/<run>/rows.json), applies kev's group-disjoint 5-fold
cross-fitted temperature (kev.metrics.out_of_fold_rows, TEMPERATURE_FIT), and writes
paper/latex/fig_reliability_scale.pdf.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "kev"))
from kev import metrics as M  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SWEEP = ROOT / "results" / "sweep-8way" / "sweep"
BINS = 10


def curve(rows):
    conf = np.array([max(r["p"]) for r in rows])
    hit = np.array([np.argmax(r["p"]) == r["label"] for r in rows], dtype=float)
    edges = np.linspace(0, 1, BINS + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.sum() >= 10:
            xs.append(conf[m].mean())
            ys.append(hit[m].mean())
    return xs, ys


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, (run, title) in zip(axes, [("td-delta-4b", "4B delta"), ("td-delta-9b", "9B delta (cold start)")]):
        rows = M.scored_rows(json.loads((SWEEP / run / "rows.json").read_text()))
        oof, temps = M.out_of_fold_rows(rows, folds=5, seed=0, **M.TEMPERATURE_FIT)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for data, label, style in [(rows, "T = 1", "s-"), (oof, f"cross-fit T ≈ {np.mean(temps):.2f}", "^-")]:
            ece = M.metrics(data)["ece"]
            ax.plot(*curve(data), style, label=f"{label} (ECE {ece:.3f})")
        ax.set_title(f"{title}, n={len(rows)}")
        ax.set_xlabel("mean confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    out = ROOT / "paper" / "latex" / "fig_reliability_scale.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
