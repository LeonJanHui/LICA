#!/usr/bin/env python3
"""Plot three Tanimoto-split difficulty figures and export supporting CSVs.

Figure 1: ECDF of max train–test ligand Tanimoto (Random vs Tanimoto train).
Figure 2: Test RMSE stratified by max-Tanimoto similarity bins.
Figure 3: Test RMSE by Murcko-scaffold exposure (seen / lost / unseen).

Outputs -> /home/dldx/Boltz-EGNN-Affinity/results/T-split/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/dldx/Boltz-EGNN-Affinity")
SRC_DIR = PROJECT_ROOT / "src"
OUT_DIR = PROJECT_ROOT / "results" / "T-split"
SIM_DIR = PROJECT_ROOT / "data" / "similarity"
PSM_DIR = SIM_DIR / "pairwise_similarity_matrix"
CURATED = PROJECT_ROOT / "data" / "curated_pdbbind"

RANDOM_SPLIT = SIM_DIR / "random_purified.json"
TANIMOTO_SPLIT = SIM_DIR / "PDBbind_split_leakage_removed_c11_purified.json"
MYDATA_SPLIT = SIM_DIR / "PDBbind_split_mydata.json"
DATA_DICT = SIM_DIR / "PDBbind_data_dict.json"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_theme(style="ticks")

COLOR_RANDOM = "#5B8FA8"
COLOR_TANIMOTO = "#E07A3D"
COLOR_LOST = "#C44E52"
COLOR_SEEN = "#4C8C6A"
COLOR_UNSEEN = "#7A7A7A"

BIN_EDGES = [0.0, 0.4, 0.5, 0.6, 0.7, 1.01]
BIN_LABELS = ["<0.4", "0.4–0.5", "0.5–0.6", "0.6–0.7", "≥0.7"]


def load_split(path: Path, key_candidates):
    data = json.load(open(path))
    for key in key_candidates:
        if key in data:
            return [str(x).strip().lower() for x in data[key]]
    raise KeyError(f"None of {key_candidates} in {path}")


def load_similarity():
    complexes = [str(c).strip().lower() for c in json.load(open(PSM_DIR / "pairwise_similarity_complexes.json"))]
    index = {c: i for i, c in enumerate(complexes)}
    mat = np.load(PSM_DIR / "pairwise_similarity_matrix_tanimoto.npy", mmap_mode="r")
    return complexes, index, mat


def max_sim_to_train(test_ids, train_ids, index, mat):
    train_idx = np.array([index[c] for c in train_ids if c in index], dtype=np.int64)
    rows = []
    for cid in test_ids:
        if cid not in index or train_idx.size == 0:
            rows.append({"pdbid": cid, "max_tanimoto": np.nan, "top10_tanimoto": np.nan})
            continue
        sims = np.asarray(mat[index[cid], train_idx], dtype=np.float64)
        order = np.sort(sims)
        rows.append(
            {
                "pdbid": cid,
                "max_tanimoto": float(order[-1]),
                "top10_tanimoto": float(order[-10:].mean()) if order.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ligand_path(cid: str) -> Path | None:
    folder = CURATED / cid
    for ext in ("sdf", "mol2"):
        p = folder / f"{cid}_ligand.{ext}"
        if p.is_file():
            return p
    return None


def mol_from_path(path: Path):
    path = str(path)
    if path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
    else:
        mol = Chem.MolFromMol2File(path, removeHs=False)
    return mol


def murcko_smiles(mol) -> str | None:
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(core, isomericSmiles=False)
    except Exception:
        return None


def scaffold_set(ids, cache_path: Path) -> set[str]:
    if cache_path.is_file():
        df = pd.read_csv(cache_path)
        return set(df["scaffold"].dropna().astype(str))

    rows = []
    for cid in tqdm(ids, desc=f"scaffolds->{cache_path.name}"):
        path = ligand_path(cid)
        if path is None:
            continue
        smi = murcko_smiles(mol_from_path(path))
        if smi:
            rows.append({"pdbid": cid, "scaffold": smi})
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return set(df["scaffold"].dropna().astype(str))


def scaffold_of_ids(ids) -> dict[str, str | None]:
    out = {}
    for cid in tqdm(ids, desc="test scaffolds"):
        path = ligand_path(cid)
        out[cid] = murcko_smiles(mol_from_path(path)) if path else None
    return out


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def pcc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def metrics_by_bin(df, sim_col, pred_col="y_pred", true_col="y_true"):
    df = df.copy()
    df["sim_bin"] = pd.cut(
        df[sim_col], bins=BIN_EDGES, labels=BIN_LABELS, right=False, include_lowest=True
    )
    rows = []
    for label in BIN_LABELS:
        sub = df[df["sim_bin"] == label]
        if len(sub) == 0:
            rows.append(
                {
                    "sim_bin": label,
                    "n": 0,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "PCC": np.nan,
                    "mean_abs_error": np.nan,
                }
            )
            continue
        rows.append(
            {
                "sim_bin": label,
                "n": int(len(sub)),
                "RMSE": rmse(sub[true_col], sub[pred_col]),
                "MAE": mae(sub[true_col], sub[pred_col]),
                "PCC": pcc(sub[true_col], sub[pred_col]),
                "mean_abs_error": float(np.mean(np.abs(sub[true_col] - sub[pred_col]))),
            }
        )
    return pd.DataFrame(rows)


def export_tanimoto_predictions(config_path: Path, out_csv: Path) -> pd.DataFrame:
    """Mean prediction over available repeat*/best.pt checkpoints."""
    sys.path.insert(0, str(SRC_DIR))
    import torch
    from train import (
        build_dataset,
        build_model,
        load_checkpoint,
        load_config,
        make_loader,
        predict,
    )

    cfg = load_config(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["paths"]["save_model_dir"])
    test_set = build_dataset(cfg, cfg["dataset"]["test_split"])
    loader = make_loader(test_set, cfg, False)
    pdbids = list(test_set.valid_ids)

    preds = []
    y_true = None
    for repeat in range(int(cfg["training"].get("num_repeats", 3))):
        ckpt = save_dir / f"repeat{repeat}" / "best.pt"
        if not ckpt.is_file():
            print(f"[WARN] missing {ckpt}")
            continue
        model = build_model(cfg, device)
        load_checkpoint(ckpt, model, device)
        yt, yp = predict(model, loader, device, f"export R{repeat}")
        y_true = yt if y_true is None else y_true
        preds.append(yp)
        print(f"[PRED] repeat{repeat}: n={len(yp)}")

    if not preds:
        raise FileNotFoundError(f"No checkpoints under {save_dir}/repeat*/best.pt")

    y_pred = np.mean(np.stack(preds, axis=0), axis=0)
    df = pd.DataFrame(
        {
            "pdbid": pdbids,
            "y_true": y_true.astype(np.float64),
            "y_pred": y_pred.astype(np.float64),
            "abs_error": np.abs(y_true - y_pred).astype(np.float64),
            "protocol": "tanimoto",
            "n_checkpoints": len(preds),
        }
    )
    df.to_csv(out_csv, index=False)
    print(f"[CSV] {out_csv}")
    return df


def plot_fig1_ecdf(df_sim: pd.DataFrame, out_png: Path, out_csv: Path):
    """df_sim columns: pdbid, max_random, max_tanimoto."""
    df_sim.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)
    for col, label, color in [
        ("max_random", "Random train", COLOR_RANDOM),
        ("max_tanimoto", "Tanimoto train", COLOR_TANIMOTO),
    ]:
        vals = np.sort(df_sim[col].dropna().to_numpy())
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, color=color, lw=2.2, label=label)
        ax.axvline(vals.mean(), color=color, ls="--", lw=1.0, alpha=0.8)

    ax.set_xlabel("Max ligand Tanimoto to training set", fontweight="bold")
    ax.set_ylabel("ECDF", fontweight="bold")
    ax.set_title("CASF-2016 vs train chemical leakage", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.grid(True, ls="--", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[FIG1] {out_png}")


def plot_fig2_binned_rmse(
    per_complex: pd.DataFrame,
    bin_tanimoto: pd.DataFrame,
    bin_random_protocol: pd.DataFrame | None,
    out_png: Path,
    out_csv: Path,
):
    """Grouped bars: RMSE by max-sim bins.

    Primary series uses bins of max_tanimoto (Tanimoto-trained model).
    If random-protocol predictions exist, a second series is added with the
    same bin edges on max_random (Random-trained model).
    """
    rows = []
    for _, r in bin_tanimoto.iterrows():
        rows.append({**r.to_dict(), "protocol": "Tanimoto model", "sim_ref": "max_tanimoto"})
    if bin_random_protocol is not None:
        for _, r in bin_random_protocol.iterrows():
            rows.append({**r.to_dict(), "protocol": "Random model", "sim_ref": "max_random"})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)

    # Also export per-complex for reproducibility
    per_path = out_csv.with_name("fig2_per_complex_errors.csv")
    per_complex.to_csv(per_path, index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
    protocols = summary["protocol"].unique().tolist()
    x = np.arange(len(BIN_LABELS))
    width = 0.36 if len(protocols) > 1 else 0.55
    colors = {"Tanimoto model": COLOR_TANIMOTO, "Random model": COLOR_RANDOM}

    for i, proto in enumerate(protocols):
        sub = summary[summary["protocol"] == proto].set_index("sim_bin").reindex(BIN_LABELS)
        offset = (i - (len(protocols) - 1) / 2) * width
        vals = sub["RMSE"].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=colors.get(proto, "#888888"),
            edgecolor="#222222",
            linewidth=0.8,
            label=proto,
            zorder=3,
        )
        for bar, n, v in zip(bars, sub["n"].fillna(0).astype(int), vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.02,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Max Tanimoto to corresponding training set", fontweight="bold")
    ax.set_ylabel("Test RMSE", fontweight="bold")
    ax.set_title("Error vs chemical neighborhood", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[FIG2] {out_png}")
    print(f"[CSV]  {per_path}")


def plot_fig3_scaffold(
    per_complex: pd.DataFrame,
    out_png: Path,
    out_csv: Path,
):
    order = ["Seen in Tanimoto train", "Lost after Tanimoto split", "Unseen in both"]
    colors = {
        "Seen in Tanimoto train": COLOR_SEEN,
        "Lost after Tanimoto split": COLOR_LOST,
        "Unseen in both": COLOR_UNSEEN,
    }

    rows = []
    for label in order:
        sub = per_complex[per_complex["scaffold_group"] == label]
        rows.append(
            {
                "scaffold_group": label,
                "n": int(len(sub)),
                "RMSE": rmse(sub["y_true"], sub["y_pred"]) if len(sub) else np.nan,
                "MAE": mae(sub["y_true"], sub["y_pred"]) if len(sub) else np.nan,
                "PCC": pcc(sub["y_true"], sub["y_pred"]) if len(sub) else np.nan,
                "mean_abs_error": float(np.mean(sub["abs_error"])) if len(sub) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)
    per_complex.to_csv(out_csv.with_name("fig3_per_complex_scaffold.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=300)
    xs = np.arange(len(order))
    vals = summary["RMSE"].to_numpy(dtype=float)
    ns = summary["n"].to_numpy(dtype=int)
    bars = ax.bar(
        xs,
        vals,
        width=0.55,
        color=[colors[k] for k in order],
        edgecolor="#222222",
        linewidth=0.8,
        zorder=3,
    )
    for bar, n, v in zip(bars, ns, vals):
        if np.isfinite(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["Seen\n(Tanimoto train)", "Lost\n(Random only)", "Unseen\n(both)"],
        fontsize=9,
    )
    ax.set_ylabel("Test RMSE (Tanimoto-trained model)", fontweight="bold")
    ax.set_title("Hard cases by scaffold exposure", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[FIG3] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    ap.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
    )
    ap.add_argument(
        "--tanimoto_pred_csv",
        type=str,
        default="",
        help="Optional precomputed predictions (pdbid,y_true,y_pred). "
        "If empty, export from mydata_repeats checkpoints.",
    )
    ap.add_argument(
        "--random_pred_csv",
        type=str,
        default="",
        help="Optional Random-protocol predictions for dual bars in Fig2.",
    )
    ap.add_argument("--skip_model", action="store_true", help="Only Fig1 (no checkpoints).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- splits & similarity -----
    test_ids = load_split(MYDATA_SPLIT, ["casf2016", "test", "casf2016_indep"])
    random_train = load_split(RANDOM_SPLIT, ["train"])
    # Use full Tanimoto purified train (includes mydata valid) for leakage ECDF,
    # matching the published Random vs Tanimoto comparison.
    tanimoto_train = load_split(TANIMOTO_SPLIT, ["train"])
    mydata_train = load_split(MYDATA_SPLIT, ["train"])

    _, index, mat = load_similarity()
    print(f"[DATA] test={len(test_ids)} random_train={len(random_train)} "
          f"tanimoto_train={len(tanimoto_train)}")

    df_rand = max_sim_to_train(test_ids, random_train, index, mat).rename(
        columns={"max_tanimoto": "max_random", "top10_tanimoto": "top10_random"}
    )
    df_tan = max_sim_to_train(test_ids, tanimoto_train, index, mat).rename(
        columns={"max_tanimoto": "max_tanimoto", "top10_tanimoto": "top10_tanimoto"}
    )
    df_sim = df_rand.merge(df_tan, on="pdbid", how="inner")
    plot_fig1_ecdf(
        df_sim,
        out_dir / "fig1_max_tanimoto_ecdf.png",
        out_dir / "fig1_max_tanimoto_per_complex.csv",
    )

    # summary stats csv
    stats = pd.DataFrame(
        [
            {
                "split": "Random",
                "mean_max": df_sim["max_random"].mean(),
                "median_max": df_sim["max_random"].median(),
                "frac_gt_0.7": float((df_sim["max_random"] > 0.7).mean()),
                "frac_gt_0.8": float((df_sim["max_random"] > 0.8).mean()),
            },
            {
                "split": "Tanimoto",
                "mean_max": df_sim["max_tanimoto"].mean(),
                "median_max": df_sim["max_tanimoto"].median(),
                "frac_gt_0.7": float((df_sim["max_tanimoto"] > 0.7).mean()),
                "frac_gt_0.8": float((df_sim["max_tanimoto"] > 0.8).mean()),
            },
        ]
    )
    stats.to_csv(out_dir / "fig1_summary_stats.csv", index=False)

    if args.skip_model:
        print("[DONE] Fig1 only (--skip_model)")
        return

    # ----- predictions (Tanimoto-trained model) -----
    tan_pred_path = Path(args.tanimoto_pred_csv) if args.tanimoto_pred_csv else out_dir / "tanimoto_model_predictions.csv"
    if args.tanimoto_pred_csv and tan_pred_path.is_file():
        pred_tan = pd.read_csv(tan_pred_path)
        pred_tan["pdbid"] = pred_tan["pdbid"].astype(str).str.lower()
    else:
        pred_tan = export_tanimoto_predictions(Path(args.config), tan_pred_path)

    per = pred_tan.merge(df_sim, on="pdbid", how="inner")
    if "abs_error" not in per.columns:
        per["abs_error"] = (per["y_true"] - per["y_pred"]).abs()

    bin_tan = metrics_by_bin(per, "max_tanimoto")
    bin_rand_model = None
    if args.random_pred_csv and Path(args.random_pred_csv).is_file():
        pred_rand = pd.read_csv(args.random_pred_csv)
        pred_rand["pdbid"] = pred_rand["pdbid"].astype(str).str.lower()
        per_r = pred_rand.merge(df_sim, on="pdbid", how="inner")
        bin_rand_model = metrics_by_bin(per_r, "max_random")
    else:
        print(
            "[INFO] No --random_pred_csv; Fig2 shows Tanimoto-trained model "
            "stratified by max_tanimoto only."
        )

    plot_fig2_binned_rmse(
        per,
        bin_tan,
        bin_rand_model,
        out_dir / "fig2_rmse_by_similarity_bin.png",
        out_dir / "fig2_rmse_by_similarity_bin.csv",
    )

    # ----- scaffolds -----
    scaf_random = scaffold_set(
        random_train, out_dir / "cache_scaffolds_random_train.csv"
    )
    scaf_tanimoto = scaffold_set(
        mydata_train, out_dir / "cache_scaffolds_tanimoto_train.csv"
    )
    # also include purified tanimoto train scaffolds (superset of mydata train+valid)
    scaf_tanimoto |= scaffold_set(
        tanimoto_train, out_dir / "cache_scaffolds_tanimoto_purified_train.csv"
    )

    test_scaf = scaffold_of_ids(per["pdbid"].tolist())
    groups = []
    for cid in per["pdbid"]:
        s = test_scaf.get(cid)
        if s and s in scaf_tanimoto:
            groups.append("Seen in Tanimoto train")
        elif s and s in scaf_random:
            groups.append("Lost after Tanimoto split")
        else:
            groups.append("Unseen in both")
    per = per.copy()
    per["scaffold"] = [test_scaf.get(c) for c in per["pdbid"]]
    per["scaffold_group"] = groups
    per["scaffold_in_random_train"] = [
        (test_scaf.get(c) in scaf_random) if test_scaf.get(c) else False for c in per["pdbid"]
    ]
    per["scaffold_in_tanimoto_train"] = [
        (test_scaf.get(c) in scaf_tanimoto) if test_scaf.get(c) else False for c in per["pdbid"]
    ]

    plot_fig3_scaffold(
        per,
        out_dir / "fig3_rmse_by_scaffold_exposure.png",
        out_dir / "fig3_rmse_by_scaffold_exposure.csv",
    )
    print(f"[DONE] outputs in {out_dir}")


if __name__ == "__main__":
    main()
