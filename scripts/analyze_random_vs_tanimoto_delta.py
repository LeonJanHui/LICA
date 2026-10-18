#!/usr/bin/env python3
"""Per-complex Δ analysis: Random-trained (M_R) vs Tanimoto-trained (M_T) on CASF-2016.

Δ_abs = |y - ŷ_T| - |y - ŷ_R|
  >0  → harder under Tanimoto than under Random
  <0  → Random worse (rare if leakage helps)

Outputs -> results/T-split/delta_MR_vs_MT/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

PROJECT_ROOT = Path("/home/dldx/Boltz-EGNN-Affinity")
SRC_DIR = PROJECT_ROOT / "src"
OUT_DIR = PROJECT_ROOT / "results" / "T-split" / "delta_MR_vs_MT"
SIM_CSV = PROJECT_ROOT / "results" / "T-split" / "fig1_max_tanimoto_per_complex.csv"
SCAF_CSV = PROJECT_ROOT / "results" / "T-split" / "fig3_per_complex_scaffold.csv"

COLOR_R = "#5B8FA8"
COLOR_T = "#E07A3D"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
sns.set_theme(style="ticks")


def _load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_predictions(
    cfg: dict,
    checkpoint_dir: Path,
    protocol: str,
    device,
) -> pd.DataFrame:
    sys.path.insert(0, str(SRC_DIR))
    from train import build_dataset, build_model, load_checkpoint, make_loader, predict

    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["save_model_dir"] = str(checkpoint_dir)

    test_set = build_dataset(cfg, cfg["dataset"]["test_split"])
    loader = make_loader(test_set, cfg, False)
    pdbids = list(test_set.valid_ids)

    preds = []
    y_true = None
    for repeat in range(int(cfg["training"].get("num_repeats", 3))):
        ckpt = checkpoint_dir / f"repeat{repeat}" / "best.pt"
        if not ckpt.is_file():
            print(f"[WARN] missing {ckpt}")
            continue
        model = build_model(cfg, device)
        load_checkpoint(ckpt, model, device)
        yt, yp = predict(model, loader, device, f"{protocol} R{repeat}")
        y_true = yt if y_true is None else y_true
        preds.append(yp)
        print(f"[PRED] {protocol} repeat{repeat}: n={len(yp)}")

    if not preds:
        raise FileNotFoundError(f"No checkpoints under {checkpoint_dir}")

    y_pred = np.mean(np.stack(preds, axis=0), axis=0)
    return pd.DataFrame(
        {
            "pdbid": pdbids,
            "y_true": y_true.astype(np.float64),
            "y_pred": y_pred.astype(np.float64),
            "abs_error": np.abs(y_true - y_pred).astype(np.float64),
            "protocol": protocol,
            "n_checkpoints": len(preds),
        }
    )


def metrics_block(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pcc = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan
    sp = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan
    return {"n": len(y_true), "RMSE": rmse, "MAE": mae, "PCC": pcc, "Spearman": sp}


def sim_bins(s: pd.Series, edges=(0.0, 0.4, 0.55, 0.7, 0.85, 1.01)) -> pd.Series:
    labels = [f"[{edges[i]:.2f},{edges[i+1]:.2f})" for i in range(len(edges) - 1)]
    return pd.cut(s, bins=edges, labels=labels, right=False, include_lowest=True)


def plot_delta_vs_similarity(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300)

    ax = axes[0]
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.scatter(
        df["max_random"],
        df["delta_abs"],
        c=COLOR_T,
        s=28,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.3,
    )
    if df["max_random"].notna().sum() > 2:
        slope, intercept = np.polyfit(df["max_random"], df["delta_abs"], 1)
        xs = np.linspace(df["max_random"].min(), df["max_random"].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="black", ls="--", lw=1.2)
        r, p = pearsonr(df["max_random"], df["delta_abs"])
        ax.set_title(f"A  Δ vs max Tanimoto to Random-train\n(r={r:.2f}, p={p:.2g})", loc="left", fontweight="bold")
    else:
        ax.set_title("A  Δ vs max Tanimoto to Random-train", loc="left", fontweight="bold")
    ax.set_xlabel("Max ligand Tanimoto to Random train")
    ax.set_ylabel(r"$\Delta_{abs}=|e_T|-|e_R|$")

    ax = axes[1]
    tmp = df.dropna(subset=["max_random_bin"]).copy()
    order = [c for c in tmp["max_random_bin"].cat.categories if (tmp["max_random_bin"] == c).any()]
    sns.boxplot(
        data=tmp,
        x="max_random_bin",
        y="delta_abs",
        order=order,
        color=COLOR_R,
        ax=ax,
        fliersize=2,
    )
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.set_title("B  Δ by Random-train similarity bin", loc="left", fontweight="bold")
    ax.set_xlabel("Max Tanimoto to Random train")
    ax.set_ylabel(r"$\Delta_{abs}$")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")


def plot_group_bars(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w / 2, summary["MAE_R"], w, label=r"$M_R$ (Random)", color=COLOR_R, edgecolor="black", lw=0.5)
    ax.bar(x + w / 2, summary["MAE_T"], w, label=r"$M_T$ (Tanimoto)", color=COLOR_T, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(summary["group"], summary["n"])], fontsize=8)
    ax.set_ylabel("MAE (pK)")
    ax.set_title("Error by stratum: Random vs Tanimoto model", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--random-config",
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
        help="Config whose ledger has casf2016; checkpoints overridden below.",
    )
    ap.add_argument(
        "--tanimoto-ledger-config",
        default=str(PROJECT_ROOT / "configs" / "ablation" / "00_full.yaml"),
        help="Config pointing at mydata/Tanimoto ledger (for test ID order).",
    )
    ap.add_argument(
        "--random-ckpt",
        default=str(PROJECT_ROOT / "checkpoints" / "mydata_repeats_random"),
    )
    ap.add_argument(
        "--tanimoto-ckpt",
        default=str(PROJECT_ROOT / "checkpoints" / "mydata_repeats"),
    )
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--skip-predict", action="store_true", help="Reuse cached prediction CSVs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    cfg_r = _load_cfg(Path(args.random_config))
    cfg_t = _load_cfg(Path(args.tanimoto_ledger_config))

    pred_r_path = out_dir / "predictions_random_model.csv"
    pred_t_path = out_dir / "predictions_tanimoto_model.csv"

    if args.skip_predict and pred_r_path.exists() and pred_t_path.exists():
        pred_r = pd.read_csv(pred_r_path)
        pred_t = pd.read_csv(pred_t_path)
        print("[LOAD] cached predictions")
    else:
        pred_r = export_predictions(cfg_r, Path(args.random_ckpt), "random", device)
        pred_t = export_predictions(cfg_t, Path(args.tanimoto_ckpt), "tanimoto", device)
        pred_r.to_csv(pred_r_path, index=False)
        pred_t.to_csv(pred_t_path, index=False)

    # Merge on pdbid (CASF order may differ across ledgers)
    m = pred_r.merge(
        pred_t,
        on="pdbid",
        suffixes=("_R", "_T"),
        how="inner",
    )
    # y_true should match
    if not np.allclose(m["y_true_R"], m["y_true_T"], atol=1e-5):
        bad = (m["y_true_R"] - m["y_true_T"]).abs().max()
        print(f"[WARN] y_true mismatch max |diff|={bad}")
    m["y_true"] = m["y_true_R"]
    m["pred_R"] = m["y_pred_R"]
    m["pred_T"] = m["y_pred_T"]
    m["abs_err_R"] = m["abs_error_R"]
    m["abs_err_T"] = m["abs_error_T"]
    m["delta_abs"] = m["abs_err_T"] - m["abs_err_R"]
    m["delta_sq"] = (m["y_true"] - m["pred_T"]) ** 2 - (m["y_true"] - m["pred_R"]) ** 2

    if SIM_CSV.exists():
        sim = pd.read_csv(SIM_CSV)
        m = m.merge(sim, on="pdbid", how="left")
    else:
        print(f"[WARN] missing {SIM_CSV}")

    if SCAF_CSV.exists():
        scaf = pd.read_csv(SCAF_CSV)[
            ["pdbid", "scaffold", "scaffold_group", "scaffold_in_random_train", "scaffold_in_tanimoto_train"]
        ]
        m = m.merge(scaf, on="pdbid", how="left")
    else:
        print(f"[WARN] missing {SCAF_CSV}")

    if "max_random" in m.columns:
        m["max_random_bin"] = sim_bins(m["max_random"])
    if "max_tanimoto" in m.columns:
        m["max_tanimoto_bin"] = sim_bins(m["max_tanimoto"])

    # Hurt / helped
    q75 = m["delta_abs"].quantile(0.75)
    q25 = m["delta_abs"].quantile(0.25)
    m["delta_class"] = "stable"
    m.loc[m["delta_abs"] >= q75, "delta_class"] = "Tanimoto-hurt (top25% Δ)"
    m.loc[m["delta_abs"] <= q25, "delta_class"] = "Tanimoto-helped (bottom25% Δ)"

    cols = [
        "pdbid",
        "y_true",
        "pred_R",
        "pred_T",
        "abs_err_R",
        "abs_err_T",
        "delta_abs",
        "delta_sq",
        "delta_class",
        "max_random",
        "top10_random",
        "max_tanimoto",
        "top10_tanimoto",
        "max_random_bin",
        "max_tanimoto_bin",
        "scaffold",
        "scaffold_group",
        "scaffold_in_random_train",
        "scaffold_in_tanimoto_train",
    ]
    cols = [c for c in cols if c in m.columns]
    per_path = out_dir / "per_complex_delta.csv"
    m[cols].sort_values("delta_abs", ascending=False).to_csv(per_path, index=False)
    print(f"[CSV] {per_path}")

    # Overall metrics
    overall = pd.DataFrame(
        [
            {"model": "M_R (Random)", **metrics_block(m["y_true"], m["pred_R"])},
            {"model": "M_T (Tanimoto)", **metrics_block(m["y_true"], m["pred_T"])},
        ]
    )
    overall["mean_delta_abs"] = [np.nan, float(m["delta_abs"].mean())]
    overall.to_csv(out_dir / "overall_metrics.csv", index=False)

    # Strata summaries
    strata_rows = []

    def add_stratum(label, sub):
        if len(sub) == 0:
            return
        row = {
            "group": label,
            "n": len(sub),
            "mean_delta_abs": float(sub["delta_abs"].mean()),
            "median_delta_abs": float(sub["delta_abs"].median()),
            "frac_delta_pos": float((sub["delta_abs"] > 0).mean()),
            "MAE_R": float(sub["abs_err_R"].mean()),
            "MAE_T": float(sub["abs_err_T"].mean()),
            "RMSE_R": float(np.sqrt((sub["abs_err_R"] ** 2).mean())),
            "RMSE_T": float(np.sqrt((sub["abs_err_T"] ** 2).mean())),
            "mean_max_random": float(sub["max_random"].mean()) if "max_random" in sub else np.nan,
            "mean_max_tanimoto": float(sub["max_tanimoto"].mean()) if "max_tanimoto" in sub else np.nan,
        }
        strata_rows.append(row)

    add_stratum("All CASF", m)
    if "max_random_bin" in m.columns:
        for b, sub in m.groupby("max_random_bin", observed=True):
            add_stratum(f"max_random {b}", sub)
    if "scaffold_group" in m.columns:
        for g, sub in m.groupby("scaffold_group"):
            add_stratum(f"scaffold: {g}", sub)
    # High leakage candidates: high max_random & low max_tanimoto
    if "max_random" in m.columns and "max_tanimoto" in m.columns:
        leak = m[(m["max_random"] >= 0.7) & (m["max_tanimoto"] < 0.7)]
        add_stratum("leakish: max_R≥0.7 & max_T<0.7", leak)
        hard = m[(m["max_random"] < 0.5) & (m["max_tanimoto"] < 0.5)]
        add_stratum("hard both: max_R<0.5 & max_T<0.5", hard)

    strata = pd.DataFrame(strata_rows)
    strata_path = out_dir / "strata_summary.csv"
    strata.to_csv(strata_path, index=False)
    print(f"[CSV] {strata_path}")

    # Top hurt / helped tables
    m.nlargest(20, "delta_abs")[cols].to_csv(out_dir / "top20_tanimoto_hurt.csv", index=False)
    m.nsmallest(20, "delta_abs")[cols].to_csv(out_dir / "top20_tanimoto_helped.csv", index=False)

    # Stats
    r_s, p_s = pearsonr(m["max_random"], m["delta_abs"]) if "max_random" in m else (np.nan, np.nan)
    stats = {
        "n": len(m),
        "mean_delta_abs": float(m["delta_abs"].mean()),
        "median_delta_abs": float(m["delta_abs"].median()),
        "frac_delta_pos": float((m["delta_abs"] > 0).mean()),
        "corr_delta_vs_max_random": float(r_s),
        "corr_delta_vs_max_random_p": float(p_s),
        "q25_delta": float(q25),
        "q75_delta": float(q75),
    }
    if "max_random" in m.columns:
        hi = m[m["max_random"] >= 0.7]["delta_abs"]
        lo = m[m["max_random"] < 0.5]["delta_abs"]
        if len(hi) > 2 and len(lo) > 2:
            stats["mwu_delta_highR_vs_lowR_p"] = float(
                mannwhitneyu(hi, lo, alternative="greater").pvalue
            )
            stats["mean_delta_maxR_ge_0.7"] = float(hi.mean())
            stats["mean_delta_maxR_lt_0.5"] = float(lo.mean())
    pd.DataFrame([stats]).to_csv(out_dir / "delta_stats.csv", index=False)

    # Plots
    if "max_random" in m.columns:
        plot_delta_vs_similarity(m, out_dir / "fig_delta_vs_random_similarity.png")
    # compact bar for key groups
    key_groups = strata[
        strata["group"].str.startswith("scaffold:")
        | strata["group"].str.startswith("leakish")
        | strata["group"].str.startswith("hard both")
        | (strata["group"] == "All CASF")
    ].copy()
    if len(key_groups):
        plot_group_bars(key_groups.reset_index(drop=True), out_dir / "fig_mae_by_stratum.png")

    # Markdown brief
    md = [
        "# $M_R$ vs $M_T$ per-complex Δ (CASF-2016)",
        "",
        r"$\Delta_{abs}=|y-\hat y_T|-|y-\hat y_R|$. Positive ⇒ harder under Tanimoto.",
        "",
        "## Overall",
        "",
        overall.to_markdown(index=False),
        "",
        f"- mean Δ_abs = {stats['mean_delta_abs']:.4f}",
        f"- fraction Δ>0 = {stats['frac_delta_pos']:.3f}",
        f"- corr(Δ, max_random) = {stats['corr_delta_vs_max_random']:.3f} (p={stats['corr_delta_vs_max_random_p']:.3g})",
        "",
        "## Strata (excerpt)",
        "",
        strata.head(20).to_markdown(index=False),
        "",
        "Files: `per_complex_delta.csv`, `strata_summary.csv`, `top20_*.csv`, figures.",
    ]
    try:
        (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    except Exception:
        # tabulate may be missing
        (out_dir / "README.md").write_text(
            "\n".join(
                [
                    "# MR vs MT per-complex delta",
                    f"mean_delta_abs={stats['mean_delta_abs']:.4f}",
                    f"frac_delta_pos={stats['frac_delta_pos']:.3f}",
                    f"corr_delta_max_random={stats['corr_delta_vs_max_random']:.3f}",
                    "See CSV files in this folder.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    print("\n[SUMMARY]")
    print(overall.to_string(index=False))
    print(f"mean Δ_abs={stats['mean_delta_abs']:.4f}  fracΔ>0={stats['frac_delta_pos']:.3f}")
    if "mean_delta_maxR_ge_0.7" in stats:
        print(
            f"mean Δ (max_R≥0.7)={stats['mean_delta_maxR_ge_0.7']:.4f}  "
            f"(max_R<0.5)={stats['mean_delta_maxR_lt_0.5']:.4f}  "
            f"MWU p={stats.get('mwu_delta_highR_vs_lowR_p', float('nan')):.3g}"
        )
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
