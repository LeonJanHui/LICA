#!/usr/bin/env python3
"""Does Track C's per-complex error reduction track non-covalent interaction counts?

Compares independently trained Full vs w/o Track C on CASF-2016 (3-seed mean).
Δ|ε| = |y - ŷ_w/oC| - |y - ŷ_Full|  (>0 ⇒ Track C lowers absolute error).

Joins PLIP interaction counts in results/T-split/test_interactions.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import mannwhitneyu, spearmanr

PROJECT = Path("/home/dldx/Boltz-EGNN-Affinity")
SRC = PROJECT / "src"
OUT = PROJECT / "results" / "ablation_tracks" / "trackc_by_interaction"
FULL_PRED = PROJECT / "results" / "T-split" / "delta_MR_vs_MT" / "predictions_tanimoto_model.csv"
INTER_CSV = PROJECT / "results" / "T-split" / "test_interactions.csv"
WOC_CFG = PROJECT / "configs" / "ablation" / "03_wo_track_c.yaml"

COUNT_COLS = [
    ("n_hbonds", "H-bonds"),
    ("n_salt_bridges", "Salt bridges"),
    ("n_hydrophobic", "Hydrophobic"),
    ("n_pi_pi", "π–π"),
    ("n_t_stacking", "T-stacking"),
    ("n_cation_pi", "Cation–π"),
    ("n_halogen", "Halogen"),
    ("n_water_bridges", "Water bridges"),
    ("n_metal_complexes", "Metal"),
    ("n_polar", "Polar (H-bond+salt)"),
    ("n_aromatic", "Aromatic (π–π+T+cation–π)"),
    ("n_contacts", "All counted contacts"),
]
BIN_COLS = [
    ("has_hbond", "H-bond"),
    ("has_salt_bridge", "Salt bridge"),
    ("has_hydrophobic", "Hydrophobic"),
    ("has_pi_stack", "π-stack (π–π or T)"),
    ("has_cation_pi", "Cation–π"),
    ("has_halogen", "Halogen"),
    ("has_water_bridge", "Water bridge"),
    ("has_metal", "Metal"),
]


def export_woc(device: str) -> pd.DataFrame:
    sys.path.insert(0, str(SRC))
    from train import build_dataset, build_model, load_checkpoint, make_loader, predict

    cfg = yaml.safe_load(WOC_CFG.read_text(encoding="utf-8"))
    ckpt_dir = Path(cfg["paths"]["save_model_dir"])
    ds = build_dataset(cfg, cfg["dataset"]["test_split"])
    loader = make_loader(ds, cfg, False)
    pdbids = [str(x).lower() for x in ds.valid_ids]
    preds, y_true = [], None
    for r in range(int(cfg["training"].get("num_repeats", 3))):
        model = build_model(cfg, device)
        load_checkpoint(ckpt_dir / f"repeat{r}" / "best.pt", model, device)
        yt, yp = predict(model, loader, device, f"w/o-C R{r}")
        y_true = yt if y_true is None else y_true
        preds.append(yp)
        print(f"[PRED] w/o-C repeat{r} n={len(yp)}")
    y_pred = np.mean(np.stack(preds, axis=0), axis=0)
    return pd.DataFrame(
        {
            "pdbid": pdbids,
            "y_true_woc": y_true.astype(np.float64),
            "y_pred_woc": y_pred.astype(np.float64),
        }
    )


def build_table(woc: pd.DataFrame) -> pd.DataFrame:
    full = pd.read_csv(FULL_PRED)
    full["pdbid"] = full["pdbid"].astype(str).str.lower()
    inter = pd.read_csv(INTER_CSV)
    inter["pdbid"] = inter["pdb_id"].astype(str).str.lower()
    df = full.merge(woc, on="pdbid", how="inner")
    df = df.merge(inter, on="pdbid", how="left")
    if not np.allclose(df["y_true"], df["y_true_woc"], atol=1e-4):
        raise RuntimeError("label mismatch between Full and w/o-C predictions")
    df["abs_full"] = (df["y_true"] - df["y_pred"]).abs()
    df["abs_woc"] = (df["y_true"] - df["y_pred_woc"]).abs()
    df["delta_abs"] = df["abs_woc"] - df["abs_full"]  # >0 Track C helps
    df["n_polar"] = df["n_hbonds"].fillna(0) + df["n_salt_bridges"].fillna(0)
    df["n_aromatic"] = df["n_pi_pi"].fillna(0) + df["n_t_stacking"].fillna(0) + df["n_cation_pi"].fillna(0)
    df["n_contacts"] = (
        df["n_hbonds"].fillna(0)
        + df["n_salt_bridges"].fillna(0)
        + df["n_hydrophobic"].fillna(0)
        + df["n_pi_pi"].fillna(0)
        + df["n_t_stacking"].fillna(0)
        + df["n_cation_pi"].fillna(0)
        + df["n_halogen"].fillna(0)
        + df["n_water_bridges"].fillna(0)
        + df["n_metal_complexes"].fillna(0)
    )
    df["has_pi_stack"] = ((df["n_pi_pi"].fillna(0) + df["n_t_stacking"].fillna(0)) > 0).astype(int)
    return df


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in COUNT_COLS:
        sub = df.dropna(subset=[col, "delta_abs"])
        rho, p = spearmanr(sub[col], sub["delta_abs"])
        rows.append(
            {
                "feature": col,
                "label": label,
                "n": len(sub),
                "spearman_delta_abs": float(rho),
                "p": float(p),
            }
        )
    return pd.DataFrame(rows)


def binary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in BIN_COLS:
        if col not in df.columns:
            continue
        a = df.loc[df[col] == 1, "delta_abs"].dropna()
        b = df.loc[df[col] == 0, "delta_abs"].dropna()
        if len(a) < 5 or len(b) < 5:
            p = float("nan")
        else:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        rows.append(
            {
                "feature": col,
                "label": label,
                "n_present": int(len(a)),
                "n_absent": int(len(b)),
                "mean_delta_present": float(a.mean()) if len(a) else float("nan"),
                "mean_delta_absent": float(b.mean()) if len(b) else float("nan"),
                "median_delta_present": float(a.median()) if len(a) else float("nan"),
                "median_delta_absent": float(b.median()) if len(b) else float("nan"),
                "mean_abs_full_present": float(df.loc[df[col] == 1, "abs_full"].mean()),
                "mean_abs_woc_present": float(df.loc[df[col] == 1, "abs_woc"].mean()),
                "mean_abs_full_absent": float(df.loc[df[col] == 0, "abs_full"].mean()),
                "mean_abs_woc_absent": float(df.loc[df[col] == 0, "abs_woc"].mean()),
                "p_mannwhitney": p,
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, spear: pd.DataFrame, binary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ax = axes[0]
    s = spear.iloc[::-1]
    colors = ["#c0392b" if p < 0.05 else "#1f4e79" for p in s["p"]]
    ax.barh(s["label"], s["spearman_delta_abs"], color=colors)
    ax.axvline(0, color="#888", lw=1)
    ax.set_xlabel(r"Spearman $\rho$ with $\Delta|\varepsilon|$")
    ax.set_title("a  Gain vs interaction count")

    ax = axes[1]
    show = binary[binary["label"].isin(["H-bond", "Salt bridge", "Hydrophobic", "π-stack (π–π or T)", "Metal"])]
    x = np.arange(len(show))
    w = 0.36
    ax.bar(x - w / 2, show["mean_delta_absent"], w, color="#7f8c8d", label="Absent")
    ax.bar(x + w / 2, show["mean_delta_present"], w, color="#c0392b", label="Present")
    ax.axhline(0, color="#888", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(show["label"], rotation=25, ha="right")
    ax.set_ylabel(r"Mean $\Delta|\varepsilon|$ (w/o C − Full)")
    ax.set_title("b  Gain by interaction type")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.scatter(df["n_polar"], df["delta_abs"], s=16, c="#1f4e79", alpha=0.55, edgecolors="none")
    ax.axhline(0, color="#888", lw=1)
    rho = spear.loc[spear.feature == "n_polar", "spearman_delta_abs"].iloc[0]
    p = spear.loc[spear.feature == "n_polar", "p"].iloc[0]
    ax.set_xlabel("Polar contacts (H-bonds + salt bridges)")
    ax.set_ylabel(r"$\Delta|\varepsilon|$")
    ax.set_title(f"c  Polar contacts (ρ={rho:.2f}, p={p:.2g})")

    fig.suptitle(
        r"Track C error reduction vs PLIP interactions (CASF-2016; $\Delta|\varepsilon|>0$ means Track C helps)",
        y=1.03,
        fontsize=11,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"trackc_gain_by_interaction.{ext}")
    plt.close(fig)


def write_readme(df: pd.DataFrame, spear: pd.DataFrame, binary: pd.DataFrame) -> None:
    helped = float((df["delta_abs"] > 0).mean())
    lines = [
        "# Track C affinity gain vs PLIP interaction type",
        "",
        "## Design",
        "",
        "- Full: `checkpoints/mydata_repeats` (3-seed mean), Tanimoto/mydata split.",
        "- w/o Track C: `checkpoints/ablation_wo_track_c` (3-seed mean), same split.",
        r"- Per complex: $\Delta|\varepsilon| = |y-\hat y_{w/o C}| - |y-\hat y_{Full}|$. Positive means Track C lowers absolute pK error.",
        "- Interaction counts: `results/T-split/test_interactions.csv` (PLIP).",
        "- This is a **prediction-error** stratification. It is separate from attention–PLIP overlap.",
        "",
        "## Cohort",
        "",
        f"- n = {len(df)}",
        f"- mean Δ|ε| = {df['delta_abs'].mean():.3f} (median {df['delta_abs'].median():.3f})",
        f"- fraction with Δ|ε| > 0 (Track C helps) = {helped:.3f}",
        f"- mean |ε| Full = {df['abs_full'].mean():.3f}; w/o C = {df['abs_woc'].mean():.3f}",
        "",
        "## Spearman (Δ|ε| vs count)",
        "",
        spear.to_string(index=False),
        "",
        "## Present vs absent",
        "",
        binary.to_string(index=False),
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device)
    woc_path = OUT / "predictions_wo_track_c.csv"
    if woc_path.exists():
        woc = pd.read_csv(woc_path)
        print("[LOAD]", woc_path)
    else:
        woc = export_woc(device)
        woc.to_csv(woc_path, index=False)
    df = build_table(woc)
    df.to_csv(OUT / "per_complex_trackc_gain.csv", index=False)
    spear = spearman_table(df)
    spear.to_csv(OUT / "spearman_delta_vs_counts.csv", index=False)
    binary = binary_table(df)
    binary.to_csv(OUT / "binary_present_vs_absent.csv", index=False)
    plot(df, spear, binary)
    write_readme(df, spear, binary)
    print(spear.to_string(index=False))
    print(binary[["label", "n_present", "mean_delta_present", "mean_delta_absent", "p_mannwhitney"]].to_string(index=False))
    print("[DONE]", OUT)


if __name__ == "__main__":
    main()
