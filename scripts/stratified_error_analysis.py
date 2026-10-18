#!/usr/bin/env python3
"""Stratified error analysis on test_master_table.csv (steps 5.1–5.4).

Outputs under results/T-split/:
  fig5_1_ligand_similarity_bins.{csv,png}
  fig5_2_pocket_tmscore_bins.{csv,png}
  fig5_3_interaction_groups.{csv,png}
  fig5_4_ligand_pocket_heatmap.{csv,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

PROJECT_ROOT = Path("/home/dldx/Boltz-EGNN-Affinity")
T_SPLIT = PROJECT_ROOT / "results" / "T-split"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_theme(style="ticks")

COLOR_MAIN = "#E07A3D"
COLOR_A = "#4C8C6A"
COLOR_B = "#C44E52"
COLOR_C = "#5B8FA8"
COLOR_D = "#7A7A7A"

LIG_EDGES = [0.0, 0.4, 0.5, 0.6, 0.7, 1.01]
LIG_LABELS = ["<0.4", "0.4–0.5", "0.5–0.6", "0.6–0.7", "≥0.7"]

POCKET_EDGES = [-0.01, 0.5, 0.8, 1.01]
POCKET_LABELS = ["<0.5", "0.5–0.8", ">0.8"]

HEAT_LIG_EDGES = [0.0, 0.4, 0.5, 0.6, 1.01]
HEAT_LIG_LABELS = ["<0.4", "0.4–0.5", "0.5–0.6", "≥0.6"]


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def metrics_block(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {
            "n": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "median_abs_error": np.nan,
            "mean_abs_error": np.nan,
        }
    return {
        "n": int(len(df)),
        "RMSE": rmse(df["experimental"], df["predicted"]),
        "MAE": mae(df["experimental"], df["predicted"]),
        "median_abs_error": float(df["abs_error"].median()),
        "mean_abs_error": float(df["abs_error"].mean()),
    }


def annotate_bars(ax, bars, ns, vals, dy=0.02):
    for bar, n, v in zip(bars, ns, vals):
        if np.isfinite(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + dy,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def plot_5_1(df: pd.DataFrame, out_dir: Path):
    d = df.copy()
    d["ligand_bin"] = pd.cut(
        d["max_ligand_tanimoto"],
        bins=LIG_EDGES,
        labels=LIG_LABELS,
        right=False,
        include_lowest=True,
    )
    rows = []
    for label in LIG_LABELS:
        sub = d[d["ligand_bin"] == label]
        rows.append({"ligand_bin": label, **metrics_block(sub)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "fig5_1_ligand_similarity_bins.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
    x = np.arange(len(LIG_LABELS))
    vals = summary["RMSE"].to_numpy(dtype=float)
    bars = ax.bar(x, vals, width=0.55, color=COLOR_MAIN, edgecolor="#222", lw=0.8, zorder=3)
    annotate_bars(ax, bars, summary["n"].tolist(), vals)
    ax.set_xticks(x)
    ax.set_xticklabels(LIG_LABELS)
    ax.set_xlabel("Max ligand Tanimoto to Tanimoto train", fontweight="bold")
    ax.set_ylabel("Test RMSE", fontweight="bold")
    ax.set_title("5.1 Error by ligand similarity bin", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_1_ligand_similarity_bins.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[5.1]", out_dir / "fig5_1_ligand_similarity_bins.png")
    return summary


def plot_5_2(df: pd.DataFrame, out_dir: Path):
    d = df.copy()
    d["pocket_bin"] = pd.cut(
        d["max_pocket_tmscore"],
        bins=POCKET_EDGES,
        labels=POCKET_LABELS,
        right=True,
        include_lowest=True,
    )
    # fix first bin label semantics: (-0.01,0.5] -> <0.5; (0.5,0.8] -> 0.5-0.8; (0.8,1.01] -> >0.8
    rows = []
    for label in POCKET_LABELS:
        sub = d[d["pocket_bin"] == label]
        rows.append({"pocket_bin": label, **metrics_block(sub)})
    summary = pd.DataFrame(rows)
    rho, pval = spearmanr(d["max_pocket_tmscore"], d["abs_error"])
    summary.attrs["spearman_rho"] = float(rho)
    summary.attrs["spearman_p"] = float(pval)
    extra = pd.DataFrame(
        [{"metric": "spearman_abs_error_vs_max_pocket_tmscore", "rho": rho, "pvalue": pval}]
    )
    summary.to_csv(out_dir / "fig5_2_pocket_tmscore_bins.csv", index=False)
    extra.to_csv(out_dir / "fig5_2_spearman.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=300)

    ax = axes[0]
    x = np.arange(len(POCKET_LABELS))
    vals = summary["RMSE"].to_numpy(dtype=float)
    bars = ax.bar(x, vals, width=0.55, color=COLOR_C, edgecolor="#222", lw=0.8, zorder=3)
    annotate_bars(ax, bars, summary["n"].tolist(), vals)
    ax.set_xticks(x)
    ax.set_xticklabels(POCKET_LABELS)
    ax.set_xlabel("Max pocket TM-score to Tanimoto train", fontweight="bold")
    ax.set_ylabel("Test RMSE", fontweight="bold")
    ax.set_title("5.2a RMSE by pocket TM bin", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
    sns.despine(ax=ax)

    ax = axes[1]
    order = POCKET_LABELS
    plot_df = d.dropna(subset=["pocket_bin", "abs_error"])
    sns.boxplot(
        data=plot_df,
        x="pocket_bin",
        y="abs_error",
        order=order,
        ax=ax,
        color="#A7C4D4",
        fliersize=2,
    )
    ax.set_xlabel("Max pocket TM-score bin", fontweight="bold")
    ax.set_ylabel("|error|", fontweight="bold")
    ax.set_title(
        f"5.2b |error| vs pocket TM\nSpearman ρ={rho:.3f}, p={pval:.2e}",
        fontweight="bold",
    )
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_2_pocket_tmscore_bins.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[5.2]", out_dir / "fig5_2_pocket_tmscore_bins.png", f"ρ={rho:.3f} p={pval:.2e}")
    return summary


def _mw_test(a: pd.Series, b: pd.Series):
    a = a.dropna().to_numpy(dtype=float)
    b = b.dropna().to_numpy(dtype=float)
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(stat), float(p)


def plot_5_3(df: pd.DataFrame, out_dir: Path):
    d = df.copy()
    # interaction density = sum of major interaction counts
    dens_cols = [
        "n_hbonds",
        "n_salt_bridges",
        "n_pi_pi",
        "n_t_stacking",
        "n_cation_pi",
        "n_halogen",
        "n_hydrophobic",
        "n_water_bridges",
    ]
    dens_cols = [c for c in dens_cols if c in d.columns]
    d["interaction_density"] = d[dens_cols].sum(axis=1)
    med = float(d["interaction_density"].median())
    d["density_group"] = np.where(
        d["interaction_density"] >= med, "High interaction density", "Low interaction density"
    )

    comparisons = [
        ("Salt bridge", "has_salt_bridge", "With salt bridge", "Without salt bridge"),
        ("Water bridge", "has_water_bridge", "With water bridge", "Without water bridge"),
        ("Halogen bond", "has_halogen", "With halogen", "Without halogen"),
    ]

    rows = []
    plot_rows = []
    for name, col, pos_lab, neg_lab in comparisons:
        if col not in d.columns:
            continue
        pos = d[d[col] == 1]
        neg = d[d[col] == 0]
        m_pos, m_neg = metrics_block(pos), metrics_block(neg)
        u, p = _mw_test(pos["abs_error"], neg["abs_error"])
        rows.append({"group_family": name, "group": pos_lab, **m_pos, "mannwhitney_U": u, "mannwhitney_p": p, "compare_to": neg_lab})
        rows.append({"group_family": name, "group": neg_lab, **m_neg, "mannwhitney_U": u, "mannwhitney_p": p, "compare_to": pos_lab})
        plot_rows.append({"family": name, "group": pos_lab, "RMSE": m_pos["RMSE"], "n": m_pos["n"], "p": p})
        plot_rows.append({"family": name, "group": neg_lab, "RMSE": m_neg["RMSE"], "n": m_neg["n"], "p": p})

    hi = d[d["density_group"] == "High interaction density"]
    lo = d[d["density_group"] == "Low interaction density"]
    m_hi, m_lo = metrics_block(hi), metrics_block(lo)
    u, p = _mw_test(hi["abs_error"], lo["abs_error"])
    rows.append(
        {
            "group_family": "Interaction density",
            "group": f"High (≥median={med:.1f})",
            **m_hi,
            "mannwhitney_U": u,
            "mannwhitney_p": p,
            "compare_to": "Low",
        }
    )
    rows.append(
        {
            "group_family": "Interaction density",
            "group": f"Low (<median={med:.1f})",
            **m_lo,
            "mannwhitney_U": u,
            "mannwhitney_p": p,
            "compare_to": "High",
        }
    )
    plot_rows.append({"family": "Density", "group": "High", "RMSE": m_hi["RMSE"], "n": m_hi["n"], "p": p})
    plot_rows.append({"family": "Density", "group": "Low", "RMSE": m_lo["RMSE"], "n": m_lo["n"], "p": p})

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "fig5_3_interaction_groups.csv", index=False)

    plot_df = pd.DataFrame(plot_rows)
    families = list(dict.fromkeys(plot_df["family"].tolist()))
    fig, axes = plt.subplots(1, len(families), figsize=(3.2 * len(families), 4.4), dpi=300, sharey=True)
    if len(families) == 1:
        axes = [axes]
    palette = [COLOR_A, COLOR_B]
    for ax, fam in zip(axes, families):
        sub = plot_df[plot_df["family"] == fam].reset_index(drop=True)
        x = np.arange(len(sub))
        bars = ax.bar(
            x,
            sub["RMSE"],
            color=palette[: len(sub)],
            edgecolor="#222",
            lw=0.8,
            width=0.55,
            zorder=3,
        )
        annotate_bars(ax, bars, sub["n"].tolist(), sub["RMSE"].tolist())
        pval = sub["p"].iloc[0]
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{fam}\nMW p={pval:.2e}", fontweight="bold", fontsize=10)
        ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Test RMSE", fontweight="bold")
    fig.suptitle("5.3 Error by non-covalent interaction groups", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_3_interaction_groups.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[5.3]", out_dir / "fig5_3_interaction_groups.png")
    return summary


def plot_5_4(df: pd.DataFrame, out_dir: Path):
    d = df.copy()
    d["lig_bin"] = pd.cut(
        d["max_ligand_tanimoto"],
        bins=HEAT_LIG_EDGES,
        labels=HEAT_LIG_LABELS,
        right=False,
        include_lowest=True,
    )
    d["pocket_bin"] = pd.cut(
        d["max_pocket_tmscore"],
        bins=POCKET_EDGES,
        labels=POCKET_LABELS,
        right=True,
        include_lowest=True,
    )

    heat_mae = (
        d.groupby(["pocket_bin", "lig_bin"], observed=False)["abs_error"]
        .mean()
        .unstack("lig_bin")
        .reindex(index=POCKET_LABELS, columns=HEAT_LIG_LABELS)
    )
    heat_n = (
        d.groupby(["pocket_bin", "lig_bin"], observed=False)["abs_error"]
        .size()
        .unstack("lig_bin")
        .reindex(index=POCKET_LABELS, columns=HEAT_LIG_LABELS)
        .fillna(0)
        .astype(int)
    )
    heat_rmse = pd.DataFrame(index=POCKET_LABELS, columns=HEAT_LIG_LABELS, dtype=float)
    for pb in POCKET_LABELS:
        for lb in HEAT_LIG_LABELS:
            sub = d[(d["pocket_bin"] == pb) & (d["lig_bin"] == lb)]
            heat_rmse.loc[pb, lb] = rmse(sub["experimental"], sub["predicted"]) if len(sub) else np.nan

    long_rows = []
    for pb in POCKET_LABELS:
        for lb in HEAT_LIG_LABELS:
            long_rows.append(
                {
                    "pocket_bin": pb,
                    "ligand_bin": lb,
                    "n": int(heat_n.loc[pb, lb]),
                    "mean_abs_error": float(heat_mae.loc[pb, lb]) if pd.notna(heat_mae.loc[pb, lb]) else np.nan,
                    "RMSE": float(heat_rmse.loc[pb, lb]) if pd.notna(heat_rmse.loc[pb, lb]) else np.nan,
                }
            )
    pd.DataFrame(long_rows).to_csv(out_dir / "fig5_4_ligand_pocket_heatmap.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300)
    data = heat_mae.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean |error|", fontweight="bold")

    ax.set_xticks(np.arange(len(HEAT_LIG_LABELS)))
    ax.set_yticks(np.arange(len(POCKET_LABELS)))
    ax.set_xticklabels(HEAT_LIG_LABELS)
    ax.set_yticklabels(POCKET_LABELS)
    ax.set_xlabel("Ligand Tanimoto bin", fontweight="bold")
    ax.set_ylabel("Pocket TM-score bin", fontweight="bold")
    ax.set_title("5.4 Joint ligand × pocket novelty", fontweight="bold")

    for i in range(len(POCKET_LABELS)):
        for j in range(len(HEAT_LIG_LABELS)):
            val = data[i, j]
            n = heat_n.iloc[i, j]
            if np.isfinite(val):
                txt = f"{val:.2f}\nn={n}"
            else:
                txt = f"n={n}"
            color = "white" if np.isfinite(val) and val >= np.nanmedian(data) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_4_ligand_pocket_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[5.4]", out_dir / "fig5_4_ligand_pocket_heatmap.png")
    return heat_mae, heat_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", default=str(T_SPLIT / "test_master_table.csv"))
    ap.add_argument("--out_dir", default=str(T_SPLIT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.master_csv)
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    required = ["experimental", "predicted", "abs_error", "max_ligand_tanimoto", "max_pocket_tmscore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns in master table: {missing}")

    print(f"[DATA] n={len(df)} from {args.master_csv}")
    plot_5_1(df, out_dir)
    plot_5_2(df, out_dir)
    plot_5_3(df, out_dir)
    plot_5_4(df, out_dir)
    print(f"[DONE] outputs in {out_dir}")


if __name__ == "__main__":
    main()
