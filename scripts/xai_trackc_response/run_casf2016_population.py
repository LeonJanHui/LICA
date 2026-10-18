#!/usr/bin/env python3
"""CASF-2016 population: Full vs independently-trained w/o-Track-C attention.

Metrics (per complex, 3-seed mean):
  - spearman_prot_atom_FullC_vs_woC  : Track-C→protein-atom vs w/o-C Track-B
  - overlap@5/10 TrackC vs PLIP
  - overlap@5/10 w/o-C TrackB vs PLIP
  - delta_overlap@5 = Full − w/o-C
  - mean_dist_top10_TrackC

Outputs under results/interpretability/trackc_response/population/
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr, wilcoxon
from torch_geometric.data import Batch

PROJECT = Path("/home/dldx/Boltz-EGNN-Affinity")
SRC = PROJECT / "src"
HERE = Path(__file__).resolve().parent
OUT = PROJECT / "results" / "interpretability" / "trackc_response" / "population"
sys.path[:0] = [str(SRC), str(HERE)]

from atom_mapping import build_mapping  # noqa: E402
from plip_util import run_plip_residues  # noqa: E402
from train import build_dataset, build_model, load_checkpoint  # noqa: E402


def _native():
    p = SRC / "interpretability" / "extract_attention.py"
    spec = importlib.util.spec_from_file_location("native_attn", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.NativeAttentionExtractor


NativeAttentionExtractor = _native()


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_models(cfg: dict, device: str) -> List[torch.nn.Module]:
    models = []
    d = Path(cfg["paths"]["save_model_dir"])
    for r in range(int(cfg["training"].get("num_repeats", 3))):
        m = build_model(cfg, device)
        load_checkpoint(d / f"repeat{r}" / "best.pt", m, device)
        m.eval()
        models.append(m)
    return models


def complex_id(data) -> str:
    cid = getattr(data, "complex_id", None) or getattr(data, "pdb_id", None)
    if isinstance(cid, (list, tuple)):
        cid = cid[0]
    return str(cid).lower()


def extract_track_c(models, data, root, device, n_lig) -> pd.DataFrame:
    batch = Batch.from_data_list([data])
    dfs = []
    for m in models:
        ext = NativeAttentionExtractor(m, device=device, root_dir=root)
        res = ext.extract_from_data(batch, use_raw_scores=False, root_dir=root)
        df = res["tracks"]["track_c_pairs"].copy()
        df["ligand_atom_idx"] = df["ligand_node_idx"].astype(int)
        df["protein_atom_idx"] = df["protein_node_idx"].astype(int) - n_lig
        dfs.append(df.rename(columns={"attention_score": "weight"}))
    out = dfs[0][["ligand_atom_idx", "protein_atom_idx", "distance_A"]].copy()
    out["weight"] = np.mean([d["weight"].to_numpy(float) for d in dfs], axis=0)
    return out.sort_values("weight", ascending=False).reset_index(drop=True)


def extract_ab(models, data, root, device, n_lig, n_prot):
    batch = Batch.from_data_list([data])
    lig_acc, prot_acc = [], []
    for m in models:
        ext = NativeAttentionExtractor(m, device=device, root_dir=root)
        res = ext.extract_from_data(batch, use_raw_scores=False, root_dir=root)
        ta = res["tracks"]["track_a_ligand"]
        lw = np.zeros(n_lig)
        for _, r in ta.iterrows():
            lw[int(r["ligand_atom_idx"])] = float(r["attention_score"])
        tb = res["tracks"]["track_b_pocket_atom"]
        pw = np.zeros(n_prot)
        for _, r in tb.iterrows():
            j = int(r["global_node_idx"]) - n_lig
            if 0 <= j < n_prot:
                pw[j] = float(r["attention_score"])
        lig_acc.append(lw)
        prot_acc.append(pw)
    return np.mean(lig_acc, 0), np.mean(prot_acc, 0)


def pairs_to_prot(pairs: pd.DataFrame, n_prot: int) -> np.ndarray:
    pw = np.zeros(n_prot)
    for _, r in pairs.iterrows():
        j = int(r["protein_atom_idx"])
        if 0 <= j < n_prot:
            pw[j] += float(r["weight"])
    return pw


def overlap_at_k(ranked: List[str], contact: Set[str], k: int) -> float:
    if not contact:
        return float("nan")
    top = set(ranked[:k])
    c = {x.strip() for x in contact}
    return len(top & c) / min(k, len(c))


def rank_residues_from_pairs(pairs: pd.DataFrame, idx2res: Dict[int, str]) -> List[str]:
    res_w: Dict[str, float] = {}
    for _, r in pairs.iterrows():
        rid = idx2res.get(int(r.protein_atom_idx))
        if rid:
            res_w[rid] = res_w.get(rid, 0.0) + float(r.weight)
    return [r for r, _ in sorted(res_w.items(), key=lambda x: -x[1])]


def rank_residues_from_atoms(pw: np.ndarray, idx2res: Dict[int, str]) -> List[str]:
    res_w: Dict[str, float] = {}
    for j, w in enumerate(pw):
        rid = idx2res.get(j)
        if rid:
            res_w[rid] = max(res_w.get(rid, 0.0), float(w))
    return [r for r, _ in sorted(res_w.items(), key=lambda x: -x[1])]


def process_one(cid, data, full_models, woc_models, root, device, contact: Set[str]) -> dict:
    mapping = build_mapping(Path(root), cid)
    n_lig, n_prot = mapping["n_lig"], mapping["n_prot"]
    if int(data.x.size(0)) != n_lig + n_prot:
        raise RuntimeError(f"atom mismatch {cid}: graph={data.x.size(0)} map={n_lig+n_prot}")

    pairs = extract_track_c(full_models, data, root, device, n_lig)
    pw_c = pairs_to_prot(pairs, n_prot)
    _, pw_woc = extract_ab(woc_models, data, root, device, n_lig, n_prot)

    idx2res = {
        int(r.protein_atom_idx): str(r.residue_id)
        for _, r in mapping["map_df"].query("role=='protein'").iterrows()
    }
    ranked_full = rank_residues_from_pairs(pairs, idx2res)
    ranked_woc = rank_residues_from_atoms(pw_woc, idx2res)

    if pw_c.sum() > 0 and pw_woc.sum() > 0:
        rho, _ = spearmanr(pw_c, pw_woc)
        rho = float(rho)
    else:
        rho = float("nan")

    o5_f = overlap_at_k(ranked_full, contact, 5)
    o5_w = overlap_at_k(ranked_woc, contact, 5)
    o10_f = overlap_at_k(ranked_full, contact, 10)
    o10_w = overlap_at_k(ranked_woc, contact, 10)
    return {
        "pdbid": cid,
        "n_plip": len(contact),
        "n_pairs": int(len(pairs)),
        "overlap@5_TrackC_vs_PLIP": o5_f,
        "overlap@10_TrackC_vs_PLIP": o10_f,
        "overlap@5_woC_TrackB_vs_PLIP": o5_w,
        "overlap@10_woC_TrackB_vs_PLIP": o10_w,
        "delta_overlap@5": (o5_f - o5_w) if (o5_f == o5_f and o5_w == o5_w) else float("nan"),
        "spearman_prot_atom_FullC_vs_woC": rho,
        "mean_dist_top10_TrackC": float(pairs.head(10)["distance_A"].mean()) if len(pairs) else float("nan"),
        "top_TrackC_residues": ";".join(ranked_full[:8]),
        "top_woC_residues": ";".join(ranked_woc[:8]),
        "status": "ok",
    }


def plot_population(df: pd.DataFrame) -> None:
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
    ok = df[df.status == "ok"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    # a) Spearman
    ax = axes[0]
    rho = ok["spearman_prot_atom_FullC_vs_woC"].dropna()
    ax.hist(rho, bins=20, color="#1f4e79", edgecolor="white", alpha=0.9)
    ax.axvline(0.0, color="#888", ls="--", lw=1)
    ax.axvline(rho.median(), color="#c0392b", ls="-", lw=1.4, label=f"median={rho.median():.2f}")
    ax.set_xlabel(r"Spearman $\rho$ (Full-C vs w/o-C Track B)")
    ax.set_ylabel("Count")
    ax.set_title("a  Attention redistribution")
    ax.legend(frameon=False, fontsize=8)

    # b) overlap@5 paired
    ax = axes[1]
    sub = ok.dropna(subset=["overlap@5_TrackC_vs_PLIP", "overlap@5_woC_TrackB_vs_PLIP"])
    data = [sub["overlap@5_woC_TrackB_vs_PLIP"], sub["overlap@5_TrackC_vs_PLIP"]]
    bp = ax.boxplot(data, labels=["w/o-C\nTrack B", "Full\nTrack C"], patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], ["#7f8c8d", "#c0392b"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for i in range(len(sub)):
        ax.plot(
            [1, 2],
            [sub["overlap@5_woC_TrackB_vs_PLIP"].iloc[i], sub["overlap@5_TrackC_vs_PLIP"].iloc[i]],
            color="#bbb",
            lw=0.4,
            alpha=0.35,
            zorder=0,
        )
    ax.set_ylabel("PLIP overlap@5")
    ax.set_title("b  Contact agreement")
    ax.set_ylim(-0.05, 1.05)

    # c) delta + distance
    ax = axes[2]
    d5 = sub["delta_overlap@5"].dropna()
    ax.hist(d5, bins=15, color="#16a085", edgecolor="white", alpha=0.9)
    ax.axvline(0.0, color="#888", ls="--", lw=1)
    ax.axvline(d5.median(), color="#c0392b", lw=1.4, label=f"median Δ={d5.median():.2f}")
    ax.set_xlabel(r"$\Delta$ overlap@5 (Full − w/o-C)")
    ax.set_ylabel("Count")
    ax.set_title("c  Gain from Track C")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"CASF-2016 Full vs w/o-Track-C attention (n={len(ok)}, independent ablation ckpts)",
        y=1.02,
        fontsize=11,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"casf2016_full_vs_woc_population.{ext}")
    plt.close(fig)

    # extra: top10 distance
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    dist = ok["mean_dist_top10_TrackC"].dropna()
    ax.hist(dist, bins=20, color="#1f4e79", edgecolor="white")
    ax.axvline(dist.median(), color="#c0392b", label=f"median={dist.median():.2f} Å")
    ax.set_xlabel("Mean distance of Top-10 Track C pairs (Å)")
    ax.set_ylabel("Count")
    ax.set_title("Track C Top-10 pair distance (CASF-2016)")
    ax.legend(frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"casf2016_trackc_top10_distance.{ext}")
    plt.close(fig)


def write_readme(df: pd.DataFrame) -> None:
    ok = df[df.status == "ok"]
    sub = ok.dropna(subset=["overlap@5_TrackC_vs_PLIP", "overlap@5_woC_TrackB_vs_PLIP"])
    rho = ok["spearman_prot_atom_FullC_vs_woC"].dropna()
    d5 = sub["delta_overlap@5"].dropna()
    dist = ok["mean_dist_top10_TrackC"].dropna()

    # paired test on overlap
    p_wil = float("nan")
    try:
        if len(sub) >= 10:
            stat = wilcoxon(
                sub["overlap@5_TrackC_vs_PLIP"],
                sub["overlap@5_woC_TrackB_vs_PLIP"],
                zero_method="wilcox",
                alternative="greater",
            )
            p_wil = float(stat.pvalue)
    except Exception:
        pass

    lines = [
        "# CASF-2016 population: Full vs w/o Track C attention",
        "",
        "## Design",
        "",
        "- **Full**: `checkpoints/mydata_repeats` (3-seed mean), Track C → protein-atom / residue ranks",
        "- **w/o-C**: `checkpoints/ablation_wo_track_c` (3-seed mean), Track B pocket atoms",
        "- Not inference-time masking of Track C in the Full model.",
        "- Contacts: PLIP residues on curated complex PDBs.",
        "",
        "## Cohort",
        "",
        f"- CASF-2016 graphs: **{len(df)}**",
        f"- Successful metrics: **{len(ok)}**",
        f"- With PLIP contacts (n_plip>0): **{(ok.n_plip > 0).sum()}**",
        "",
        "## Key results",
        "",
        f"- Spearman(Full-C atom, w/o-C Track B) median = **{rho.median():.3f}** "
        f"(mean {rho.mean():.3f}; fraction ρ<0.3 = {(rho < 0.3).mean():.2f})",
        f"- PLIP overlap@5: Full Track C median = **{sub['overlap@5_TrackC_vs_PLIP'].median():.3f}** "
        f"vs w/o-C Track B = **{sub['overlap@5_woC_TrackB_vs_PLIP'].median():.3f}** "
        f"(median Δ = **{d5.median():.3f}**)",
        f"- Wilcoxon signed-rank (Full > w/o-C overlap@5): p = **{p_wil:.2e}**",
        f"- Top-10 Track C pair distance median = **{dist.median():.2f} Å**",
        "",
        "## Figures",
        "",
        "- `casf2016_full_vs_woc_population.png` — Spearman / overlap@5 / Δoverlap",
        "- `casf2016_trackc_top10_distance.png` — short-range check",
        "",
        "## Suggested Results sentence",
        "",
        "> Across CASF-2016, protein-atom attention from independently trained Full (Track C) "
        f"and w/o-Track-C (Track B) models is largely uncorrelated "
        f"(median Spearman ρ ≈ {rho.median():.2f}), indicating redistribution of pocket focus. "
        f"Track C residue ranks agree better with PLIP contacts than w/o-C Track B "
        f"(median overlap@5 {sub['overlap@5_TrackC_vs_PLIP'].median():.2f} vs "
        f"{sub['overlap@5_woC_TrackB_vs_PLIP'].median():.2f}; median Δ ≈ {d5.median():.2f}), "
        f"and Top-10 Track C pairs remain short-range (median ≈ {dist.median():.1f} Å).",
        "",
        "## Re-run",
        "",
        "```bash",
        "/home/dldx/miniconda3/envs/boltz-egnn/bin/python \\",
        "  scripts/xai_trackc_response/run_casf2016_population.py --device cuda",
        "```",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-n", type=int, default=0, help="debug: limit complexes (0=all)")
    ap.add_argument("--skip-plip", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip pdbids already in CSV")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "casf2016_full_vs_woc_metrics.csv"
    plip_scratch = OUT / "_plip"
    plip_scratch.mkdir(exist_ok=True)

    full_cfg = load_cfg(PROJECT / "configs" / "ablation" / "00_full.yaml")
    woc_cfg = load_cfg(PROJECT / "configs" / "ablation" / "03_wo_track_c.yaml")
    device = args.device
    print(f"[LOAD] device={device}")
    full_models = load_models(full_cfg, device)
    woc_models = load_models(woc_cfg, device)
    ds = build_dataset(full_cfg, full_cfg["dataset"]["test_split"])
    root = str(full_cfg["paths"]["root_dir"])

    done: Set[str] = set()
    rows: List[dict] = []
    if args.resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = {str(r["pdbid"]).lower() for r in rows}
        print(f"[RESUME] {len(done)} done")

    n_total = len(ds) if args.max_n <= 0 else min(len(ds), args.max_n)
    t0 = time.time()
    for i in range(n_total):
        data = ds[i]
        cid = complex_id(data)
        if cid in done:
            continue
        try:
            if args.skip_plip:
                contact: Set[str] = set()
            else:
                contact = run_plip_residues(cid, plip_scratch)
            row = process_one(cid, data, full_models, woc_models, root, device, contact)
        except Exception as e:
            row = {
                "pdbid": cid,
                "n_plip": 0,
                "n_pairs": 0,
                "overlap@5_TrackC_vs_PLIP": float("nan"),
                "overlap@10_TrackC_vs_PLIP": float("nan"),
                "overlap@5_woC_TrackB_vs_PLIP": float("nan"),
                "overlap@10_woC_TrackB_vs_PLIP": float("nan"),
                "delta_overlap@5": float("nan"),
                "spearman_prot_atom_FullC_vs_woC": float("nan"),
                "mean_dist_top10_TrackC": float("nan"),
                "top_TrackC_residues": "",
                "top_woC_residues": "",
                "status": f"error:{type(e).__name__}:{e}",
            }
        rows.append(row)
        done.add(cid)
        if (len(rows) % 10 == 0) or (i + 1 == n_total):
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            elapsed = time.time() - t0
            print(
                f"[{len(rows)}/{n_total}] {cid} status={row['status']} "
                f"ρ={row.get('spearman_prot_atom_FullC_vs_woC', float('nan'))} "
                f"Δ@5={row.get('delta_overlap@5', float('nan'))} "
                f"elapsed={elapsed/60:.1f}m"
            )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    plot_population(df)
    write_readme(df)
    ok = df[df.status == "ok"]
    print("[DONE]", OUT)
    print(
        ok[
            [
                "spearman_prot_atom_FullC_vs_woC",
                "overlap@5_TrackC_vs_PLIP",
                "overlap@5_woC_TrackB_vs_PLIP",
                "delta_overlap@5",
                "mean_dist_top10_TrackC",
            ]
        ]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()
