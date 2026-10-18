#!/usr/bin/env python3
"""Step 3 — chemical / structural features of high-Δ (Tanimoto-hurt) complexes.

Joins per_complex_delta.csv with test_master_table.csv and compares the
top-quartile Δ subset vs the rest (and vs bottom-quartile) on:

  • continuous ligand / pocket / affinity / interaction-count features
  • binary interaction & chemotype flags (enrichment)
  • stratified mean Δ by chemical bins

Outputs -> results/T-split/delta_MR_vs_MT/chemotype/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

PROJECT_ROOT = Path("/home/dldx/Boltz-EGNN-Affinity")
DELTA_CSV = PROJECT_ROOT / "results" / "T-split" / "delta_MR_vs_MT" / "per_complex_delta.csv"
MASTER_CSV = PROJECT_ROOT / "results" / "T-split" / "test_master_table.csv"
OUT_DIR = PROJECT_ROOT / "results" / "T-split" / "delta_MR_vs_MT" / "chemotype"

COLOR_HI = "#E07A3D"
COLOR_LO = "#5B8FA8"
COLOR_MID = "#7A7A7A"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
sns.set_theme(style="ticks")

CONT_FEATURES = [
    ("y_true", "Experimental pK"),
    ("mw", "Molecular weight"),
    ("logp", "cLogP"),
    ("tpsa", "TPSA"),
    ("hbd", "H-bond donors"),
    ("hba", "H-bond acceptors"),
    ("n_rotatable_bonds", "Rotatable bonds"),
    ("n_aromatic_rings", "Aromatic rings"),
    ("n_heavy_atoms", "Heavy atoms"),
    ("formal_charge", "Formal charge"),
    ("max_pocket_tmscore", "Max pocket TM-score (train)"),
    ("max_pocket_identity", "Max pocket identity (train)"),
    ("test_pocket_length", "Pocket length"),
    ("delta_sasa", "ΔSASA"),
    ("n_hbonds", "H-bonds"),
    ("n_hydrophobic", "Hydrophobic contacts"),
    ("n_salt_bridges", "Salt bridges"),
    ("n_water_bridges", "Water bridges"),
    ("n_pi_pi", "π–π"),
    ("interaction_density", "Interaction density"),
    ("max_random", "Max ligand Tc (Random train)"),
    ("max_tanimoto", "Max ligand Tc (Tanimoto train)"),
    ("leakage_gap", "max_R − max_T"),
]

BINARY_FLAGS = [
    ("has_hbond", "Has H-bond"),
    ("has_salt_bridge", "Has salt bridge"),
    ("has_water_bridge", "Has water bridge"),
    ("has_halogen", "Has halogen bond"),
    ("has_pi_pi", "Has π–π"),
    ("has_t_stacking", "Has T-stacking"),
    ("has_cation_pi", "Has cation–π"),
    ("has_hydrophobic", "Has hydrophobic"),
    ("has_metal", "Has metal"),
    ("scaffold_in_random_train", "Scaffold in Random train"),
    ("scaffold_in_tanimoto_train", "Scaffold in Tanimoto train"),
    ("is_peptide_like", "Peptide-like"),
    ("is_nucleotide_like", "Nucleotide-like"),
    ("is_fragment", "Fragment (MW<300)"),
    ("is_large", "Large ligand (MW≥500)"),
    ("is_flexible", "Flexible (≥8 rotbonds)"),
    ("is_polar", "Polar (TPSA≥90)"),
    ("is_lipophilic", "Lipophilic (cLogP≥3)"),
    ("high_affinity", "High affinity (pK≥8)"),
    ("low_affinity", "Low affinity (pK≤5)"),
    ("leakish", "Leakish (max_R≥0.7 & max_T<0.7)"),
    ("high_pocket_tm", "High pocket TM (≥0.8)"),
]


# SMARTS for coarse chemotypes (interpretive only)
SMARTS = {
    "is_peptide_like": "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+]",  # peptide bond + Cα
    "is_nucleotide_like": "[$(O=P([O-])([O-])O),$(n1cnc2ncnc12),$(n1cnc2nc[nH]c12)]",
}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ: P(x>y) - P(x<y). Positive ⇒ x tends larger than y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    # efficient via ranks
    nx, ny = len(x), len(y)
    more = 0
    less = 0
    for xi in x:
        more += np.sum(xi > y)
        less += np.sum(xi < y)
    return float((more - less) / (nx * ny))


def annotate_chemotypes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    peptide = []
    nucleo = []
    for smi in d["smiles"].fillna(""):
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        if mol is None:
            peptide.append(False)
            nucleo.append(False)
            continue
        pep_q = Chem.MolFromSmarts(SMARTS["is_peptide_like"])
        nuc_q = Chem.MolFromSmarts(SMARTS["is_nucleotide_like"])
        peptide.append(bool(mol.HasSubstructMatch(pep_q)) if pep_q else False)
        nucleo.append(bool(mol.HasSubstructMatch(nuc_q)) if nuc_q else False)
    d["is_peptide_like"] = peptide
    d["is_nucleotide_like"] = nucleo
    d["is_fragment"] = d["mw"] < 300
    d["is_large"] = d["mw"] >= 500
    d["is_flexible"] = d["n_rotatable_bonds"] >= 8
    d["is_polar"] = d["tpsa"] >= 90
    d["is_lipophilic"] = d["logp"] >= 3
    d["high_affinity"] = d["y_true"] >= 8
    d["low_affinity"] = d["y_true"] <= 5
    d["leakish"] = (d["max_random"] >= 0.7) & (d["max_tanimoto"] < 0.7)
    d["high_pocket_tm"] = d["max_pocket_tmscore"] >= 0.8
    dens_cols = [
        "n_hbonds",
        "n_salt_bridges",
        "n_pi_pi",
        "n_t_stacking",
        "n_cation_pi",
        "n_halogen",
        "n_hydrophobic",
        "n_water_bridges",
        "n_metal_complexes",
    ]
    d["interaction_density"] = d[dens_cols].fillna(0).sum(axis=1)
    d["leakage_gap"] = d["max_random"] - d["max_tanimoto"]
    return d


def load_merged(delta_csv: Path, master_csv: Path) -> pd.DataFrame:
    delta = pd.read_csv(delta_csv)
    master = pd.read_csv(master_csv)
    master = master.rename(columns={"pdb_id": "pdbid"})
    chem_cols = [
        "pdbid",
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "n_rotatable_bonds",
        "n_aromatic_rings",
        "formal_charge",
        "n_heavy_atoms",
        "smiles",
        "max_pocket_tmscore",
        "nearest_train_pdb_pocket",
        "max_pocket_identity",
        "nearest_train_pdb_identity",
        "tmscore_norm_by_test",
        "tmscore_norm_by_train",
        "test_pocket_length",
        "n_tmalign",
        "n_hbonds",
        "n_hbonds_backbone",
        "n_hbonds_sidechain",
        "n_salt_bridges",
        "n_pi_pi",
        "n_t_stacking",
        "n_cation_pi",
        "n_halogen",
        "n_hydrophobic",
        "n_water_bridges",
        "n_metal_complexes",
        "delta_sasa",
        "has_hbond",
        "has_salt_bridge",
        "has_water_bridge",
        "has_halogen",
        "has_pi_pi",
        "has_t_stacking",
        "has_cation_pi",
        "has_hydrophobic",
        "has_metal",
    ]
    chem_cols = [c for c in chem_cols if c in master.columns]
    m = delta.merge(master[chem_cols], on="pdbid", how="left", validate="one_to_one")
    m = annotate_chemotypes(m)

    # groups from existing delta_class or recompute quartiles
    q75 = float(m["delta_abs"].quantile(0.75))
    q25 = float(m["delta_abs"].quantile(0.25))
    m["delta_quartile"] = np.where(
        m["delta_abs"] >= q75,
        "High-Δ (top25%)",
        np.where(m["delta_abs"] <= q25, "Low-Δ (bottom25%)", "Mid-Δ"),
    )
    m["is_high_delta"] = m["delta_quartile"] == "High-Δ (top25%)"
    m["is_low_delta"] = m["delta_quartile"] == "Low-Δ (bottom25%)"
    return m


def compare_continuous(df: pd.DataFrame) -> pd.DataFrame:
    hi = df[df["is_high_delta"]]
    rest = df[~df["is_high_delta"]]
    lo = df[df["is_low_delta"]]
    rows = []
    for col, label in CONT_FEATURES:
        if col not in df.columns:
            continue
        a = hi[col].astype(float).to_numpy()
        b = rest[col].astype(float).to_numpy()
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        cd = cliffs_delta(a, b)
        # also vs low-Δ
        c = lo[col].astype(float).to_numpy()
        c = c[np.isfinite(c)]
        if len(c) >= 3:
            _, p_lo = mannwhitneyu(a, c, alternative="two-sided")
            cd_lo = cliffs_delta(a, c)
        else:
            p_lo, cd_lo = float("nan"), float("nan")
        rows.append(
            {
                "feature": col,
                "label": label,
                "n_high": int(len(a)),
                "n_rest": int(len(b)),
                "mean_high": float(np.mean(a)),
                "mean_rest": float(np.mean(b)),
                "median_high": float(np.median(a)),
                "median_rest": float(np.median(b)),
                "mean_diff": float(np.mean(a) - np.mean(b)),
                "cliffs_delta_vs_rest": cd,
                "mwu_p_vs_rest": float(p),
                "mean_low": float(np.mean(c)) if len(c) else float("nan"),
                "cliffs_delta_vs_low": cd_lo,
                "mwu_p_vs_low": float(p_lo),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_cliffs"] = out["cliffs_delta_vs_rest"].abs()
    out = out.sort_values("abs_cliffs", ascending=False).drop(columns=["abs_cliffs"])
    return out


def compare_binary(df: pd.DataFrame) -> pd.DataFrame:
    hi = df[df["is_high_delta"]]
    rest = df[~df["is_high_delta"]]
    rows = []
    for col, label in BINARY_FLAGS:
        if col not in df.columns:
            continue
        a = hi[col].astype(bool)
        b = rest[col].astype(bool)
        # contingency: [[hi+, hi-], [rest+, rest-]]
        t11 = int(a.sum())
        t10 = int((~a).sum())
        t01 = int(b.sum())
        t00 = int((~b).sum())
        table = np.array([[t11, t10], [t01, t00]])
        try:
            odds, p = fisher_exact(table, alternative="two-sided")
        except ValueError:
            odds, p = float("nan"), float("nan")
        rate_hi = t11 / max(t11 + t10, 1)
        rate_rest = t01 / max(t01 + t00, 1)
        rows.append(
            {
                "feature": col,
                "label": label,
                "n_high_pos": t11,
                "n_high": t11 + t10,
                "rate_high": rate_hi,
                "n_rest_pos": t01,
                "n_rest": t01 + t00,
                "rate_rest": rate_rest,
                "rate_diff": rate_hi - rate_rest,
                "odds_ratio": float(odds),
                "fisher_p": float(p),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_rate_diff"] = out["rate_diff"].abs()
    out = out.sort_values("abs_rate_diff", ascending=False).drop(columns=["abs_rate_diff"])
    return out


def stratify_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Δ within chemical / structural bins (whole CASF)."""
    specs = [
        ("mw_bin", pd.cut(df["mw"], bins=[0, 300, 400, 500, 2000], labels=["<300", "300–400", "400–500", "≥500"])),
        ("logp_bin", pd.cut(df["logp"], bins=[-20, 0, 3, 5, 20], labels=["≤0", "0–3", "3–5", ">5"])),
        ("flex_bin", pd.cut(df["n_rotatable_bonds"], bins=[-0.1, 3, 7, 12, 100], labels=["0–3", "4–7", "8–12", "≥13"])),
        ("tpsa_bin", pd.cut(df["tpsa"], bins=[-0.1, 60, 90, 140, 1000], labels=["<60", "60–90", "90–140", "≥140"])),
        ("pk_bin", pd.cut(df["y_true"], bins=[0, 5, 7, 9, 15], labels=["≤5", "5–7", "7–9", "≥9"])),
        (
            "pocket_tm_bin",
            pd.cut(df["max_pocket_tmscore"], bins=[-0.01, 0.5, 0.8, 1.01], labels=["<0.5", "0.5–0.8", "≥0.8"]),
        ),
        (
            "density_bin",
            pd.qcut(df["interaction_density"], q=3, labels=["Low", "Mid", "High"], duplicates="drop"),
        ),
        ("chemotype", np.where(df["is_peptide_like"], "Peptide-like", np.where(df["is_nucleotide_like"], "Nucleotide-like", "Other"))),
        ("leakish_flag", np.where(df["leakish"], "Leakish", "Not leakish")),
        ("scaffold_group", df["scaffold_group"]),
    ]
    rows = []
    for name, series in specs:
        tmp = df.copy()
        tmp["_bin"] = series
        for g, sub in tmp.groupby("_bin", observed=True):
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "axis": name,
                    "bin": str(g),
                    "n": int(len(sub)),
                    "mean_delta": float(sub["delta_abs"].mean()),
                    "median_delta": float(sub["delta_abs"].median()),
                    "frac_high_delta": float(sub["is_high_delta"].mean()),
                    "MAE_R": float(sub["abs_err_R"].mean()),
                    "MAE_T": float(sub["abs_err_T"].mean()),
                    "mean_max_R": float(sub["max_random"].mean()),
                    "mean_max_T": float(sub["max_tanimoto"].mean()),
                    "mean_mw": float(sub["mw"].mean()),
                }
            )
    return pd.DataFrame(rows)


def spearman_vs_delta(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in CONT_FEATURES:
        if col not in df.columns:
            continue
        x = df[col].astype(float)
        y = df["delta_abs"].astype(float)
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        rho, p = spearmanr(x[mask], y[mask])
        rows.append({"feature": col, "label": label, "spearman_rho": float(rho), "pvalue": float(p), "n": int(mask.sum())})
    out = pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    return out


def plot_continuous_forest(cont: pd.DataFrame, out: Path):
    # top features by |Cliff's δ|
    d = cont.copy()
    d["abs_cd"] = d["cliffs_delta_vs_rest"].abs()
    d = d.sort_values("abs_cd", ascending=True).tail(14)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    colors = [COLOR_HI if v > 0 else COLOR_LO for v in d["cliffs_delta_vs_rest"]]
    ax.barh(d["label"], d["cliffs_delta_vs_rest"], color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="k", lw=0.8)
    for y, (_, r) in enumerate(d.iterrows()):
        star = ""
        if r["mwu_p_vs_rest"] < 0.001:
            star = "***"
        elif r["mwu_p_vs_rest"] < 0.01:
            star = "**"
        elif r["mwu_p_vs_rest"] < 0.05:
            star = "*"
        ax.text(
            r["cliffs_delta_vs_rest"] + (0.02 if r["cliffs_delta_vs_rest"] >= 0 else -0.02),
            y,
            f"{star}  Δmean={r['mean_diff']:+.2g}",
            va="center",
            ha="left" if r["cliffs_delta_vs_rest"] >= 0 else "right",
            fontsize=7.5,
            color="#333",
        )
    ax.set_xlabel("Cliff's δ (High-Δ vs rest)\n→ High-Δ larger | ← High-Δ smaller")
    ax.set_title("Chemical / structural shift of High-Δ complexes", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out / "fig_highdelta_feature_cliffs.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_highdelta_feature_cliffs.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_binary_enrichment(bin_df: pd.DataFrame, out: Path):
    d = bin_df.copy()
    # drop near-universal flags that clutter
    d = d[~d["feature"].isin(["has_hbond", "has_hydrophobic"])]
    d["abs_rd"] = d["rate_diff"].abs()
    d = d.sort_values("abs_rd", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = [COLOR_HI if v > 0 else COLOR_LO for v in d["rate_diff"]]
    ax.barh(d["label"], d["rate_diff"] * 100, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="k", lw=0.8)
    for y, (_, r) in enumerate(d.iterrows()):
        star = ""
        if r["fisher_p"] < 0.001:
            star = "***"
        elif r["fisher_p"] < 0.01:
            star = "**"
        elif r["fisher_p"] < 0.05:
            star = "*"
        ax.text(
            r["rate_diff"] * 100 + (1.5 if r["rate_diff"] >= 0 else -1.5),
            y,
            f"{star}  {r['rate_high']*100:.0f}% vs {r['rate_rest']*100:.0f}%",
            va="center",
            ha="left" if r["rate_diff"] >= 0 else "right",
            fontsize=7.5,
        )
    ax.set_xlabel("Rate difference (pp): High-Δ − rest")
    ax.set_title("Enrichment of chemotypes / interactions in High-Δ", fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out / "fig_highdelta_binary_enrichment.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_highdelta_binary_enrichment.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_strata_bars(strata: pd.DataFrame, out: Path):
    axes_keep = ["mw_bin", "flex_bin", "tpsa_bin", "pk_bin", "pocket_tm_bin", "leakish_flag"]
    titles = {
        "mw_bin": "MW",
        "flex_bin": "Rotatable bonds",
        "tpsa_bin": "TPSA",
        "pk_bin": "Experimental pK",
        "pocket_tm_bin": "Pocket TM-score",
        "leakish_flag": "Ligand leakage",
    }
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, key in zip(axs.ravel(), axes_keep):
        sub = strata[strata["axis"] == key].copy()
        if sub.empty:
            ax.axis("off")
            continue
        # preserve categorical order as written
        bars = ax.bar(sub["bin"].astype(str), sub["mean_delta"], color=COLOR_HI, edgecolor="none", width=0.72)
        ax.axhline(0, color="k", lw=0.7)
        for bar, (_, r) in zip(bars, sub.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                r["mean_delta"] + (0.01 if r["mean_delta"] >= 0 else -0.02),
                f"n={r['n']}",
                ha="center",
                va="bottom" if r["mean_delta"] >= 0 else "top",
                fontsize=7,
            )
        ax.set_title(titles[key], fontsize=10, fontweight="bold")
        ax.set_ylabel("mean Δ" if ax in (axs[0, 0], axs[1, 0]) else "")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        sns.despine(ax=ax)
    fig.suptitle("Where mean Δ concentrates (chemical / structural bins)", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "fig_delta_by_chem_bins.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_delta_by_chem_bins.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_key_boxplots(df: pd.DataFrame, out: Path):
    keys = [
        ("mw", "MW"),
        ("n_rotatable_bonds", "Rotatable bonds"),
        ("tpsa", "TPSA"),
        ("leakage_gap", "max_R − max_T"),
        ("max_pocket_tmscore", "Pocket TM"),
        ("interaction_density", "Interaction density"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.0))
    order = ["High-Δ (top25%)", "Mid-Δ", "Low-Δ (bottom25%)"]
    palette = {"High-Δ (top25%)": COLOR_HI, "Mid-Δ": COLOR_MID, "Low-Δ (bottom25%)": COLOR_LO}
    for ax, (col, lab) in zip(axs.ravel(), keys):
        sns.boxplot(
            data=df,
            x="delta_quartile",
            y=col,
            order=order,
            hue="delta_quartile",
            palette=palette,
            legend=False,
            ax=ax,
            fliersize=2,
            linewidth=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7.5)
        sns.despine(ax=ax)
    fig.suptitle("Feature distributions by Δ quartile", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig_highdelta_boxplots.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig_highdelta_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(
    df: pd.DataFrame,
    cont: pd.DataFrame,
    bin_df: pd.DataFrame,
    strata: pd.DataFrame,
    spear: pd.DataFrame,
    out: Path,
):
    n_hi = int(df["is_high_delta"].sum())
    q75 = float(df.loc[df["is_high_delta"], "delta_abs"].min())
    # top continuous
    top_c = cont.head(8)
    top_b = bin_df.head(8)
    sig_c = cont[cont["mwu_p_vs_rest"] < 0.05]
    sig_b = bin_df[bin_df["fisher_p"] < 0.05]

    lines = [
        "# Step 3 — Chemical / structural profile of High-Δ complexes",
        "",
        "High-Δ = top quartile of $\\Delta_{abs}=|e_T|-|e_R|$ "
        f"(n={n_hi}, $\\Delta \\ge {q75:.3f}$).",
        "",
        "## Headline",
        "",
    ]
    # auto headline from top signals
    if not sig_c.empty:
        lines.append("Significant continuous shifts (High-Δ vs rest, MWU p<0.05):")
        for _, r in sig_c.iterrows():
            direction = "higher" if r["mean_diff"] > 0 else "lower"
            lines.append(
                f"- **{r['label']}**: {direction} "
                f"(mean {r['mean_high']:.3g} vs {r['mean_rest']:.3g}, "
                f"Cliff's δ={r['cliffs_delta_vs_rest']:+.2f}, p={r['mwu_p_vs_rest']:.3g})"
            )
        lines.append("")
    else:
        lines.append("No continuous feature reached p<0.05 vs rest; see effect sizes below.")
        lines.append("")

    if not sig_b.empty:
        lines.append("Significant binary enrichments (Fisher p<0.05):")
        for _, r in sig_b.iterrows():
            tag = "enriched" if r["rate_diff"] > 0 else "depleted"
            lines.append(
                f"- **{r['label']}**: {tag} "
                f"({r['rate_high']*100:.0f}% vs {r['rate_rest']*100:.0f}%, "
                f"OR={r['odds_ratio']:.2g}, p={r['fisher_p']:.3g})"
            )
        lines.append("")
    else:
        lines.append("No binary flag reached Fisher p<0.05; report rate differences as descriptive.")
        lines.append("")

    lines += [
        "## Files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `high_delta_complexes.csv` | High-Δ rows + chem/struct columns |",
        "| `continuous_feature_tests.csv` | MWU + Cliff's δ vs rest / vs Low-Δ |",
        "| `binary_enrichment.csv` | Rates + Fisher OR |",
        "| `delta_by_chem_bins.csv` | mean Δ by MW / flex / TPSA / pK / pocket / leakage |",
        "| `spearman_delta_vs_features.csv` | Rank correlation of features with Δ |",
        "| `fig_highdelta_feature_cliffs.*` | Effect-size forest (continuous) |",
        "| `fig_highdelta_binary_enrichment.*` | Rate-diff enrichment |",
        "| `fig_delta_by_chem_bins.*` | mean Δ bars by bins |",
        "| `fig_highdelta_boxplots.*` | Distributions by Δ quartile |",
        "",
        "## Re-run",
        "",
        "```bash",
        "/home/dldx/miniconda3/envs/boltz-egnn/bin/python scripts/analyze_high_delta_chemotype.py",
        "```",
        "",
        "## Top continuous (by |Cliff's δ|)",
        "",
    ]
    for _, r in top_c.iterrows():
        lines.append(
            f"- {r['label']}: mean_diff={r['mean_diff']:+.3g}, "
            f"δ={r['cliffs_delta_vs_rest']:+.2f}, p={r['mwu_p_vs_rest']:.3g}"
        )
    lines += ["", "## Top binary (by |rate diff|)", ""]
    for _, r in top_b.iterrows():
        lines.append(
            f"- {r['label']}: {r['rate_high']*100:.0f}% vs {r['rate_rest']*100:.0f}% "
            f"(OR={r['odds_ratio']:.2g}, p={r['fisher_p']:.3g})"
        )

    # highlight strata extremes
    lines += ["", "## Bins with largest mean Δ", ""]
    for axis in ["mw_bin", "flex_bin", "tpsa_bin", "pk_bin", "leakish_flag", "chemotype"]:
        sub = strata[strata["axis"] == axis].sort_values("mean_delta", ascending=False)
        if sub.empty:
            continue
        best = sub.iloc[0]
        worst = sub.iloc[-1]
        lines.append(
            f"- **{axis}**: highest mean Δ in `{best['bin']}` "
            f"(Δ={best['mean_delta']:+.3f}, n={best['n']}); "
            f"lowest in `{worst['bin']}` (Δ={worst['mean_delta']:+.3f}, n={worst['n']})"
        )

    if not spear.empty:
        lines += ["", "## Strongest Spearman(feature, Δ)", ""]
        for _, r in spear.head(6).iterrows():
            lines.append(f"- {r['label']}: ρ={r['spearman_rho']:+.3f}, p={r['pvalue']:.3g}")

    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_csv", default=str(DELTA_CSV))
    ap.add_argument("--master_csv", default=str(MASTER_CSV))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_merged(Path(args.delta_csv), Path(args.master_csv))
    print(f"[DATA] n={len(df)} highΔ={df['is_high_delta'].sum()} lowΔ={df['is_low_delta'].sum()}")

    cont = compare_continuous(df)
    bin_df = compare_binary(df)
    strata = stratify_delta(df)
    spear = spearman_vs_delta(df)

    hi_cols = [
        "pdbid",
        "y_true",
        "pred_R",
        "pred_T",
        "abs_err_R",
        "abs_err_T",
        "delta_abs",
        "max_random",
        "max_tanimoto",
        "leakage_gap",
        "scaffold_group",
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "n_rotatable_bonds",
        "n_aromatic_rings",
        "n_heavy_atoms",
        "max_pocket_tmscore",
        "max_pocket_identity",
        "interaction_density",
        "n_hbonds",
        "n_hydrophobic",
        "n_salt_bridges",
        "n_water_bridges",
        "delta_sasa",
        "is_peptide_like",
        "is_nucleotide_like",
        "is_flexible",
        "is_polar",
        "leakish",
        "smiles",
    ]
    hi_cols = [c for c in hi_cols if c in df.columns]
    df.loc[df["is_high_delta"], hi_cols].sort_values("delta_abs", ascending=False).to_csv(
        out / "high_delta_complexes.csv", index=False
    )
    cont.to_csv(out / "continuous_feature_tests.csv", index=False)
    bin_df.to_csv(out / "binary_enrichment.csv", index=False)
    strata.to_csv(out / "delta_by_chem_bins.csv", index=False)
    spear.to_csv(out / "spearman_delta_vs_features.csv", index=False)
    # full annotated table for reuse
    df.to_csv(out / "per_complex_with_chemotype.csv", index=False)

    plot_continuous_forest(cont, out)
    plot_binary_enrichment(bin_df, out)
    plot_strata_bars(strata, out)
    plot_key_boxplots(df, out)
    write_readme(df, cont, bin_df, strata, spear, out)

    print("[OK] wrote", out)
    print(cont[["label", "mean_diff", "cliffs_delta_vs_rest", "mwu_p_vs_rest"]].head(10).to_string(index=False))
    print("---")
    print(bin_df[["label", "rate_high", "rate_rest", "odds_ratio", "fisher_p"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
