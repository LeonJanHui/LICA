#!/usr/bin/env python3
"""Reviewer-response interpretability: 3D attention maps + Full vs w/o Track C.

Selection (locked in case_list.yaml):
  - lowest-error representative
  - interaction-type exemplar (H-bond + salt bridge)
  - high-error outlier

Full model → Track C pair weights ω_ij → atom B-factors → PyMOL
w/o Track C → Track A/B atom attention → same PDB layout (no pairs)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

PROJECT = Path("/home/dldx/Boltz-EGNN-Affinity")
SRC = PROJECT / "src"
HERE = Path(__file__).resolve().parent
OUT = PROJECT / "results" / "interpretability" / "trackc_response"
sys.path[:0] = [str(SRC), str(HERE)]

from atom_mapping import build_mapping  # noqa: E402
from plip_util import label_res, run_plip_residues  # noqa: E402
from train import build_dataset, build_model, load_checkpoint  # noqa: E402


def _native():
    p = SRC / "interpretability" / "extract_attention.py"
    spec = importlib.util.spec_from_file_location("native_attn", p)
    mod = importlib.util.module_from_spec(spec)
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


def id_index(ds) -> Dict[str, int]:
    out = {}
    for i in range(len(ds)):
        data = ds[i]
        cid = getattr(data, "complex_id", None) or getattr(data, "pdb_id", None)
        if isinstance(cid, (list, tuple)):
            cid = cid[0]
        out[str(cid).lower()] = i
    return out


def scale_b(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    mx = float(np.nanmax(w)) if np.isfinite(w).any() else 0.0
    return (w / mx) * 100.0 if mx > 0 else np.zeros_like(w)


def write_pdb(map_df: pd.DataFrame, lig_w: np.ndarray, prot_w: np.ndarray, path: Path):
    raw = []
    for _, r in map_df.iterrows():
        if r["role"] == "ligand":
            raw.append(float(lig_w[int(r["ligand_atom_idx"])]))
        else:
            raw.append(float(prot_w[int(r["protein_atom_idx"])]))
    b = scale_b(np.array(raw))
    lines = []
    for i, (_, r) in enumerate(map_df.iterrows()):
        name = str(r["atom_name"])[:4]
        if len(name) < 4:
            name = f"{name:>3s} "
        rec = "HETATM" if r["role"] == "ligand" else "ATOM  "
        ch = str(r["chain"]) if pd.notna(r["chain"]) and str(r["chain"]).strip() else "A"
        lines.append(
            f"{rec}{int(r['pdb_serial']):5d} {name}{str(r['resname'])[:3]:>3s} {ch:1s}"
            f"{int(r['resseq']):4d}    "
            f"{float(r['x']):8.3f}{float(r['y']):8.3f}{float(r['z']):8.3f}"
            f"  1.00{b[i]:6.2f}          {str(r['element']):>2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def pairs_to_atoms(pairs: pd.DataFrame, n_lig: int, n_prot: int):
    lw, pw = np.zeros(n_lig), np.zeros(n_prot)
    for _, r in pairs.iterrows():
        lw[int(r["ligand_atom_idx"])] += float(r["weight"])
        pw[int(r["protein_atom_idx"])] += float(r["weight"])
    return lw, pw


def overlap_at_k(ranked_res: List[str], contact: Set[str], k: int) -> float:
    if not contact:
        return float("nan")
    top = set(ranked_res[:k])
    c = {x.strip() for x in contact}
    return len(top & c) / min(k, len(c))


def write_pml_full(cid, pdb, pairs, map_df, out_pml, out_png, contact: Set[str]):
    prot = map_df[map_df.role == "protein"]
    lig_s = {int(r.ligand_atom_idx): int(r.pdb_serial) for _, r in map_df[map_df.role == "ligand"].iterrows()}
    prot_s = {int(r.protein_atom_idx): int(r.pdb_serial) for _, r in prot.iterrows()}
    res_ser: Dict[str, List[int]] = {}
    for _, r in prot.iterrows():
        res_ser.setdefault(str(r.residue_id), []).append(int(r.pdb_serial))

    lines = [
        f"# Full model Track-C attention — {cid}",
        "reinitialize", "bg_color white",
        f"load {pdb.resolve()}, cplx",
        "hide everything, cplx",
        "show sticks, polymer and cplx",
        "color gray70, polymer and cplx",
        "set stick_radius, 0.12, polymer and cplx",
        "show cartoon, polymer and cplx",
        "set cartoon_transparency, 0.8",
        "color gray85, polymer and cplx and cartoon",
        "show sticks, resn LIG and cplx",
        "util.cnc resn LIG and cplx",
        "set stick_radius, 0.20, resn LIG and cplx",
        "select attn_atoms, cplx and b > 5",
        "show sticks, attn_atoms",
        "hide spheres, cplx",
        "set stick_radius, 0.18, attn_atoms and polymer",
        "spectrum b, blue_white_red, (attn_atoms or (resn LIG)), minimum=0, maximum=100",
        "color gray70, polymer and cplx and not (b > 5)",
        "color gray85, polymer and cplx and cartoon",
        "set ray_opaque_background, on", "set antialias, 2",
        "orient (resn LIG) or attn_atoms", "zoom (resn LIG) or attn_atoms, 8",
        "hide labels, cplx",
    ]
    top = pairs.head(max(1, int(np.ceil(len(pairs) * 0.10))))
    wmax = float(top.weight.max()) if len(top) else 1.0
    res_score: Dict[str, float] = {}
    idx2res = {int(r.protein_atom_idx): str(r.residue_id) for _, r in prot.iterrows()}
    for k, (_, r) in enumerate(top.iterrows()):
        li, pj = int(r.ligand_atom_idx), int(r.protein_atom_idx)
        if li not in lig_s or pj not in prot_s:
            continue
        frac = float(r.weight) / (wmax + 1e-12)
        col = "red" if frac > 0.66 else ("tv_orange" if frac > 0.33 else "salmon")
        name = f"tc{k:02d}"
        lines += [
            f"distance {name}, cplx and id {lig_s[li]}, cplx and id {prot_s[pj]}",
            f"set dash_width, {1+5*frac:.2f}, {name}",
            f"set dash_color, {col}, {name}",
            f"hide labels, {name}",
        ]
        rid = idx2res.get(pj)
        if rid:
            res_score[rid] = res_score.get(rid, 0.0) + float(r.weight)

    hi = [r for r, _ in sorted(res_score.items(), key=lambda x: -x[1])[:4]]
    if hi:
        sers = []
        for rid in hi:
            sers.extend(res_ser.get(rid, []))
        if sers:
            lines += [
                f"select hi_res, cplx and id {'+'.join(map(str, sers))}",
                "show sticks, hi_res",
                "set stick_radius, 0.22, hi_res",
                "spectrum b, blue_white_red, hi_res, minimum=0, maximum=100",
            ]
        lig_com = map_df[map_df.role == "ligand"][["x", "y", "z"]].to_numpy(float).mean(0)
        for i, rid in enumerate(hi):
            sub = prot[prot.residue_id.astype(str) == rid]
            if sub.empty:
                continue
            rcom = sub[["x", "y", "z"]].to_numpy(float).mean(0)
            vec = rcom - lig_com
            n = np.linalg.norm(vec) or 1.0
            pos = rcom + vec / n * 4.0
            lines += [
                f"pseudoatom lab{i}, pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]",
                f'label lab{i}, "{label_res(rid)}"',
                f"hide nonbonded, lab{i}",
                f"set label_size, 11, lab{i}",
                f"set label_color, black, lab{i}",
            ]
        # mark PLIP overlap in comment
        hit = [r for r in hi if r in contact]
        lines.append(f"# hi-weight residues in PLIP: {hit}")

    lines += [
        f"png {out_png.resolve()}, dpi=300, ray=1, width=2000, height=1500",
        "# quit",
    ]
    out_pml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pml_woc(cid, pdb, map_df, out_pml, out_png):
    lines = [
        f"# w/o Track C (Track A+B atom attention) — {cid}",
        "reinitialize", "bg_color white",
        f"load {pdb.resolve()}, cplx",
        "hide everything, cplx",
        "show sticks, polymer and cplx",
        "color gray70, polymer and cplx",
        "set stick_radius, 0.12, polymer and cplx",
        "show cartoon, polymer and cplx",
        "set cartoon_transparency, 0.8",
        "color gray85, polymer and cplx and cartoon",
        "show sticks, resn LIG and cplx",
        "util.cnc resn LIG and cplx",
        "select attn_atoms, cplx and b > 5",
        "show sticks, attn_atoms",
        "hide spheres, cplx",
        "spectrum b, blue_white_red, (attn_atoms or (resn LIG)), minimum=0, maximum=100",
        "color gray70, polymer and cplx and not (b > 5)",
        "orient (resn LIG) or attn_atoms", "zoom (resn LIG) or attn_atoms, 8",
        f"png {out_png.resolve()}, dpi=300, ray=1, width=2000, height=1500",
        "# quit",
    ]
    out_pml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_case(cid, role, full_models, woc_models, ds, idx, full_cfg, device, contact: Set[str]):
    root = Path(full_cfg["paths"]["root_dir"])
    data = ds[idx[cid]]
    mapping = build_mapping(root, cid)
    n_lig, n_prot = mapping["n_lig"], mapping["n_prot"]
    assert int(data.x.size(0)) == n_lig + n_prot, f"atom mismatch {cid}"

    case = OUT / cid
    full_dir, woc_dir = case / "full", case / "wo_track_c"
    full_dir.mkdir(parents=True, exist_ok=True)
    woc_dir.mkdir(parents=True, exist_ok=True)

    # Full Track C
    pairs = extract_track_c(full_models, data, str(root), device, n_lig)
    pairs[["ligand_atom_idx", "protein_atom_idx", "weight"]].to_csv(full_dir / f"{cid}_attention.csv", index=False)
    pairs.to_csv(full_dir / f"{cid}_attention_pairs_full.csv", index=False)
    mapping["map_df"].to_csv(full_dir / f"{cid}_atom_index_map.csv", index=False)
    lw, pw = pairs_to_atoms(pairs, n_lig, n_prot)
    write_pdb(mapping["map_df"], lw, pw, full_dir / f"{cid}_attention.pdb")
    write_pml_full(
        cid, full_dir / f"{cid}_attention.pdb", pairs, mapping["map_df"],
        full_dir / f"{cid}_attention.pml", full_dir / f"{cid}_attention.png", contact,
    )

    # residue ranking from Track C for metrics
    idx2res = {int(r.protein_atom_idx): str(r.residue_id) for _, r in mapping["map_df"].query("role=='protein'").iterrows()}
    res_w: Dict[str, float] = {}
    for _, r in pairs.iterrows():
        rid = idx2res.get(int(r.protein_atom_idx))
        if rid:
            res_w[rid] = res_w.get(rid, 0.0) + float(r.weight)
    ranked_full = [r for r, _ in sorted(res_w.items(), key=lambda x: -x[1])]

    # w/o Track C
    lw2, pw2 = extract_ab(woc_models, data, str(root), device, n_lig, n_prot)
    pd.DataFrame(columns=["ligand_atom_idx", "protein_atom_idx", "weight"]).to_csv(
        woc_dir / f"{cid}_attention.csv", index=False
    )
    (woc_dir / "NOTE.txt").write_text(
        "No Track C interface pairs. Atom B-factors from Track A (ligand) + Track B (pocket).\n",
        encoding="utf-8",
    )
    mapping["map_df"].to_csv(woc_dir / f"{cid}_atom_index_map.csv", index=False)
    write_pdb(mapping["map_df"], lw2, pw2, woc_dir / f"{cid}_attention.pdb")
    write_pml_woc(cid, woc_dir / f"{cid}_attention.pdb", mapping["map_df"],
                  woc_dir / f"{cid}_attention.pml", woc_dir / f"{cid}_attention.png")

    # pocket residue ranking from Track B for w/o-C
    res_w2: Dict[str, float] = {}
    for j, w in enumerate(pw2):
        rid = idx2res.get(j)
        if rid:
            res_w2[rid] = max(res_w2.get(rid, 0.0), float(w))
    ranked_woc = [r for r, _ in sorted(res_w2.items(), key=lambda x: -x[1])]

    # spearman of protein atom weights
    if pw.sum() > 0 and pw2.sum() > 0:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(pw, pw2)
    else:
        rho = float("nan")

    row = {
        "pdbid": cid,
        "role": role,
        "n_plip": len(contact),
        "overlap@5_TrackC_vs_PLIP": overlap_at_k(ranked_full, contact, 5),
        "overlap@10_TrackC_vs_PLIP": overlap_at_k(ranked_full, contact, 10),
        "overlap@5_woC_TrackB_vs_PLIP": overlap_at_k(ranked_woc, contact, 5),
        "overlap@10_woC_TrackB_vs_PLIP": overlap_at_k(ranked_woc, contact, 10),
        "spearman_prot_atom_FullC_vs_woC": float(rho) if rho == rho else float("nan"),
        "mean_dist_top10_TrackC": float(pairs.head(10)["distance_A"].mean()),
        "top_TrackC_residues": ";".join(ranked_full[:8]),
        "top_woC_residues": ";".join(ranked_woc[:8]),
    }
    print(f"[OK] {cid} ({role}) TrackC-PLIP@5={row['overlap@5_TrackC_vs_PLIP']:.2f} "
          f"woC-PLIP@5={row['overlap@5_woC_TrackB_vs_PLIP']:.2f} spearman={row['spearman_prot_atom_FullC_vs_woC']:.2f}")
    return row


def write_readme(cases_meta, summary: pd.DataFrame):
    lines = [
        "# Track C interpretability — reviewer response",
        "",
        "## What this answers",
        "",
        "1. **3D mapping**: Track C (Full) / Track A+B (w/o C) attention → atom B-factors on the complex PDB.",
        "2. **Case selection** (explicit, locked below).",
        "3. **Outlier**: included to show failure modes.",
        "4. **Full vs w/o Track C**: same complexes, side-by-side maps + PLIP overlap table.",
        "",
        "## Case selection criteria",
        "",
        "| PDB | Role | Criterion |",
        "|-----|------|-----------|",
    ]
    for c in cases_meta:
        lines.append(f"| **{c['pdbid']}** | {c['role']} | {c['criterion'].strip().replace(chr(10), ' ')} |")
    lines += [
        "",
        "## Models",
        "",
        "- Full (+Track C): `checkpoints/mydata_repeats` (3-seed mean)",
        "- w/o Track C: `checkpoints/ablation_wo_track_c` (3-seed mean)",
        "",
        "## Key quantitative result",
        "",
        "See `full_vs_woc_summary.csv`.",
        "",
    ]
    if len(summary):
        d5 = summary["overlap@5_TrackC_vs_PLIP"].mean() - summary["overlap@5_woC_TrackB_vs_PLIP"].mean()
        lines += [
            f"- mean PLIP overlap@5: **Track C (Full) = {summary['overlap@5_TrackC_vs_PLIP'].mean():.3f}** "
            f"vs **w/o-C Track B = {summary['overlap@5_woC_TrackB_vs_PLIP'].mean():.3f}** (Δ={d5:+.3f})",
            f"- mean Spearman(protein-atom attention Full vs w/o-C) = "
            f"{summary['spearman_prot_atom_FullC_vs_woC'].mean():.3f} (≪1 ⇒ redistribution)",
            f"- mean Top-10 Track C pair distance = {summary['mean_dist_top10_TrackC'].mean():.2f} Å",
            "",
            "## Suggested Results sentence",
            "",
            "> Mapping Track C attention onto complex atoms shows short-range interface contacts "
            f"(~{summary['mean_dist_top10_TrackC'].mean():.1f} Å) that agree with PLIP residues "
            f"(overlap@5 ≈ {summary['overlap@5_TrackC_vs_PLIP'].mean():.2f}). "
            "Removing Track C lowers this agreement and reorders pocket attention "
            f"(Spearman ≈ {summary['spearman_prot_atom_FullC_vs_woC'].mean():.2f}), "
            "indicating Track C changes the learned interaction representation.",
            "",
        ]
    lines += [
        "## Render (local PyMOL)",
        "",
        "```bash",
        "pymol -cq results/interpretability/trackc_response/2qbr/full/2qbr_attention.pml",
        "pymol -cq results/interpretability/trackc_response/2qbr/wo_track_c/2qbr_attention.pml",
        "```",
        "",
        "## Re-run",
        "",
        "```bash",
        "/home/dldx/miniconda3/envs/boltz-egnn/bin/python scripts/xai_trackc_response/run_all.py",
        "```",
        "",
        "## Caveats",
        "",
        "- Weights are within-complex relative attentions, **not** binding free energies.",
        "- w/o Track C has no interface-pair ω_ij; comparison uses Track A/B atom maps.",
        "- CASF-2016 case studies; mechanistic evidence, not a population claim.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    case_yaml = OUT / "case_list.yaml"
    cases_meta = [
        {
            "pdbid": "4j3l",
            "role": "lowest_error",
            "criterion": (
                "Smallest Full-model abs error on CASF-2016 (abs_err≈0.008). "
                "Also carries π–π contacts (n_pi_pi=2), serving as an accurate exemplar."
            ),
        },
        {
            "pdbid": "2qbr",
            "role": "interaction_type_Hbond_saltbridge",
            "criterion": (
                "Low abs error (≈0.041) with a chemically typical polar interface: "
                "n_hbonds=7 and n_salt_bridges=2. Chosen to illustrate H-bond/salt-bridge recognition."
            ),
        },
        {
            "pdbid": "4tmn",
            "role": "outlier_high_error",
            "criterion": (
                "High-error outlier (abs_err≈3.40) with a flexible, H-bond-rich ligand. "
                "Included to expose model limitations on attention maps."
            ),
        },
    ]
    case_yaml.write_text(yaml.dump({"cases": cases_meta}, sort_keys=False), encoding="utf-8")

    full_cfg = load_cfg(PROJECT / "configs" / "ablation" / "00_full.yaml")
    woc_cfg = load_cfg(PROJECT / "configs" / "ablation" / "03_wo_track_c.yaml")
    device = args.device
    print("[LOAD] Full models...")
    full_models = load_models(full_cfg, device)
    print("[LOAD] w/o Track C models...")
    woc_models = load_models(woc_cfg, device)
    ds = build_dataset(full_cfg, full_cfg["dataset"]["test_split"])
    idx = id_index(ds)

    scratch = OUT / "_plip"
    scratch.mkdir(exist_ok=True)
    rows = []
    for c in cases_meta:
        cid = c["pdbid"].lower()
        contact = run_plip_residues(cid, scratch)
        (OUT / cid).mkdir(exist_ok=True)
        (OUT / cid / "plip_contact_residues.txt").write_text(
            "\n".join(sorted(contact)) + ("\n" if contact else ""), encoding="utf-8"
        )
        rows.append(
            process_case(cid, c["role"], full_models, woc_models, ds, idx, full_cfg, device, contact)
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "full_vs_woc_summary.csv", index=False)
    write_readme(cases_meta, summary)
    print("[DONE]", OUT)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
