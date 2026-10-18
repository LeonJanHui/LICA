#!/usr/bin/env python3
"""Fig A/B from Full-model Track C CSVs (no re-inference).

Fig A: Top-12 contact-residue bars (weight summed over protein residue; ★ = PLIP)
Fig B: weight vs distance_A scatter
Optional: Top-15 atom-pair bars
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path("/home/dldx/Boltz-EGNN-Affinity")
OUT = PROJECT / "results" / "interpretability" / "trackc_response"
CASES = ("4j3l", "2qbr", "4tmn")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def load_plip(cid: str) -> set[str]:
    p = OUT / cid / "plip_contact_residues.txt"
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}


def residue_weights(pairs: pd.DataFrame, amap: pd.DataFrame) -> pd.DataFrame:
    prot = amap[amap.role == "protein"][["protein_atom_idx", "residue_id"]].drop_duplicates()
    idx2res = dict(zip(prot.protein_atom_idx.astype(int), prot.residue_id.astype(str)))
    rows = []
    for _, r in pairs.iterrows():
        rid = idx2res.get(int(r.protein_atom_idx))
        if rid is None:
            continue
        rows.append({"residue_id": rid, "weight": float(r.weight)})
    if not rows:
        return pd.DataFrame(columns=["residue_id", "weight"])
    g = pd.DataFrame(rows).groupby("residue_id", as_index=False)["weight"].sum()
    return g.sort_values("weight", ascending=False).reset_index(drop=True)


def pair_labels(pairs: pd.DataFrame, amap: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    lig = amap[amap.role == "ligand"]
    prot = amap[amap.role == "protein"]
    lig_el = dict(zip(lig.ligand_atom_idx.astype(int), lig.element.astype(str)))
    prot_res = dict(zip(prot.protein_atom_idx.astype(int), prot.residue_id.astype(str)))
    top = pairs.head(n).copy()
    labels = []
    for _, r in top.iterrows():
        li, pj = int(r.ligand_atom_idx), int(r.protein_atom_idx)
        el = lig_el.get(li, "?")
        res = prot_res.get(pj, f"p{pj}")
        labels.append(f"{el}{li}–{res}")
    top = top.assign(label=labels)
    return top


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)


def plot_residue_bars(cid: str, res_df: pd.DataFrame, plip: set[str], top_n: int = 12) -> None:
    top = res_df.head(top_n).iloc[::-1]  # bottom→top for horizontal bars
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colors = ["#c0392b" if r in plip else "#2c3e50" for r in top.residue_id]
    y = np.arange(len(top))
    ax.barh(y, top.weight, color=colors, height=0.7)
    ytick = [f"★ {rid}" if rid in plip else rid for rid in top.residue_id]
    ax.set_yticks(y)
    ax.set_yticklabels(ytick)
    ax.set_xlabel("Track C weight (summed over residue)")
    ax.set_title(f"{cid.upper()} — Top-{top_n} contact residues (Full)")
    n_hit = sum(1 for r in top.residue_id if r in plip)
    ax.text(
        0.98,
        0.02,
        f"★ PLIP ({n_hit}/{len(top)} in Top-{top_n})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#c0392b",
    )
    save_fig(fig, OUT / cid / "full" / f"{cid}_trackc_residue_bars")


def plot_weight_vs_dist(cid: str, pairs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.scatter(
        pairs.distance_A,
        pairs.weight,
        s=22,
        c="#1f4e79",
        alpha=0.65,
        edgecolors="none",
    )
    top10 = pairs.head(10)
    ax.scatter(
        top10.distance_A,
        top10.weight,
        s=48,
        facecolors="none",
        edgecolors="#c0392b",
        linewidths=1.4,
        label=f"Top-10 (mean {top10.distance_A.mean():.2f} Å)",
    )
    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("Track C weight")
    ax.set_title(f"{cid.upper()} — weight vs distance (Full)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save_fig(fig, OUT / cid / "full" / f"{cid}_trackc_weight_vs_dist")


def plot_pair_bars(cid: str, labeled: pd.DataFrame) -> None:
    top = labeled.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.barh(top.label, top.weight, color="#1f4e79", height=0.7)
    ax.set_xlabel("Track C weight")
    ax.set_title(f"{cid.upper()} — Top-{len(labeled)} atom pairs (Full)")
    save_fig(fig, OUT / cid / "full" / f"{cid}_trackc_pair_bars")


def process(cid: str) -> dict:
    full = OUT / cid / "full"
    pairs = pd.read_csv(full / f"{cid}_attention_pairs_full.csv")
    amap = pd.read_csv(full / f"{cid}_atom_index_map.csv")
    plip = load_plip(cid)
    res_df = residue_weights(pairs, amap)
    plot_residue_bars(cid, res_df, plip)
    plot_weight_vs_dist(cid, pairs)
    plot_pair_bars(cid, pair_labels(pairs, amap, n=15))
    top12 = set(res_df.head(12).residue_id)
    return {
        "pdbid": cid,
        "n_pairs": len(pairs),
        "top12_plip_hits": len(top12 & plip),
        "mean_dist_top10": float(pairs.head(10).distance_A.mean()),
    }


def main() -> None:
    rows = [process(c) for c in CASES]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "trackc_2d_plot_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("[DONE] figures under results/interpretability/trackc_response/*/full/")


if __name__ == "__main__":
    main()
