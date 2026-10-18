#!/usr/bin/env python3
"""Plot Track A/B/C ablation (+ optional param-matched controls).

Equivariant / coord-update ablation is intentionally excluded.
Fill ``pearson_r`` / ``rmse`` etc. from ``results/ablation_tracks/table_SX_ablation.csv``
after ``make_ablation_table.py`` completes, or pass --from-table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PROJECT_ROOT / "results" / "ablation_tracks" / "table_SX_ablation.csv"

# Fallback placeholders (replace after suite finishes).
FALLBACK = {
    "variants": [
        "Full Model",
        "w/o Track A",
        "w/o Track B",
        "w/o Track C",
        "w/o A + matched",
        "w/o B + matched",
        "w/o C + matched",
    ],
    "pearson_r": [np.nan] * 7,
    "rmse": [np.nan] * 7,
}


def _parse_mean(cell: str) -> float:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return float("nan")
    text = str(cell)
    if text in {"—", "-", "nan"}:
        return float("nan")
    return float(text.split("±")[0])


def load_from_table(path: Path):
    df = pd.read_csv(path)
    # Prefer plain ablations + full; include param-matched if present.
    order = [
        "Full model",
        "w/o Track A",
        "w/o Track B",
        "w/o Track C",
        "w/o Track A + param-matched",
        "w/o Track B + param-matched",
        "w/o Track C + param-matched",
    ]
    short = {
        "Full model": "Full Model",
        "w/o Track A": "w/o Track A",
        "w/o Track B": "w/o Track B",
        "w/o Track C": "w/o Track C",
        "w/o Track A + param-matched": "w/o A + matched",
        "w/o Track B + param-matched": "w/o B + matched",
        "w/o Track C + param-matched": "w/o C + matched",
    }
    variants, pearson_r, rmse = [], [], []
    for name in order:
        hit = df[df["Model"] == name]
        if hit.empty:
            continue
        row = hit.iloc[0]
        variants.append(short[name])
        pearson_r.append(_parse_mean(row["PCC"]))
        rmse.append(_parse_mean(row["RMSE"]))
    return variants, pearson_r, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-table", default=str(DEFAULT_TABLE))
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "results" / "ablation_tracks" / "fig_ablation_tracks.png"),
    )
    args = parser.parse_args()

    table_path = Path(args.from_table)
    if table_path.exists():
        variants, pearson_r, rmse = load_from_table(table_path)
    else:
        variants = FALLBACK["variants"]
        pearson_r = FALLBACK["pearson_r"]
        rmse = FALLBACK["rmse"]
        print(f"[WARN] {table_path} missing; plotting placeholders.")

    x = np.arange(len(variants))
    width = 0.35
    sns.set_theme(style="white")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0), dpi=300)

    colors_p = "#3282B8"
    colors_r = "#E05638"

    ax1.bar(x, pearson_r, width, color=colors_p, edgecolor="black", linewidth=0.7)
    ax1.set_title("A  Binding affinity correlation (PCC)", fontsize=13, fontweight="bold", loc="left")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=9, rotation=25, ha="right")
    ax1.set_ylabel("Pearson $r$", fontsize=11, fontweight="bold")
    finite_p = [v for v in pearson_r if np.isfinite(v)]
    if finite_p:
        ax1.set_ylim(min(finite_p) - 0.03, max(finite_p) + 0.02)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.bar(x, rmse, width, color=colors_r, edgecolor="black", linewidth=0.7)
    ax2.set_title("B  Prediction error (RMSE)", fontsize=13, fontweight="bold", loc="left")
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, fontsize=9, rotation=25, ha="right")
    ax2.set_ylabel("RMSE", fontsize=11, fontweight="bold")
    finite_r = [v for v in rmse if np.isfinite(v)]
    if finite_r:
        ax2.set_ylim(min(finite_r) - 0.05, max(finite_r) + 0.08)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Track readout ablation (no equivariant ablation); param-matched controls included",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[OK] saved -> {out}")


if __name__ == "__main__":
    main()
