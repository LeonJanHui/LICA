#!/usr/bin/env python3
"""Graph atom index ↔ output PDB serial (matches ComplexGraphDataset)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SRC = Path("/home/dldx/Boltz-EGNN-Affinity/src")
sys.path.insert(0, str(SRC))
from dataset import _read_ligand, _read_protein  # noqa: E402

WATER = {"HOH", "WAT", "H2O", "DOD"}


def locate_files(root_dir: Path, cid: str) -> Tuple[Path, Path]:
    folder = root_dir / cid
    lig = next((folder / f"{cid}_ligand.{e}" for e in ("sdf", "mol2") if (folder / f"{cid}_ligand.{e}").is_file()), None)
    prot = next((folder / f"{cid}_{k}.pdb" for k in ("pocket", "protein") if (folder / f"{cid}_{k}.pdb").is_file()), None)
    if lig is None or prot is None:
        raise FileNotFoundError(cid)
    return lig, prot


def build_mapping(root_dir: Path, cid: str, include_hydrogens=False, water_policy="remove", water_cutoff=4.0) -> Dict:
    lig_path, prot_path = locate_files(root_dir, cid)
    ligand = _read_ligand(str(lig_path), include_hydrogens)
    protein = _read_protein(str(prot_path), include_hydrogens)
    if ligand is None or protein is None:
        raise RuntimeError(f"parse failed {cid}")
    lig_pos = np.asarray(ligand.GetConformer().GetPositions(), dtype=np.float32)
    n_lig = ligand.GetNumAtoms()
    valid, labels = [], []
    for idx, atom in enumerate(protein.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        res_name = info.GetResidueName().strip().upper() if info else "UNK"
        if res_name in WATER and water_policy == "remove":
            continue
        if (not include_hydrogens) and atom.GetAtomicNum() == 1:
            continue
        valid.append(idx)
        if info:
            ch = (info.GetChainId() or "A").strip() or "A"
            labels.append(f"{ch}:{res_name}{int(info.GetResidueNumber())}{(info.GetInsertionCode() or '').strip()}")
        else:
            labels.append("A:UNK")
    n_prot = len(valid)
    rows = []
    for i, orig in enumerate(valid):
        atom = protein.GetAtomWithIdx(orig)
        info = atom.GetPDBResidueInfo()
        pos = protein.GetConformer().GetAtomPosition(orig)
        rows.append(dict(
            pdb_serial=i + 1, role="protein", ligand_atom_idx=-1, protein_atom_idx=i,
            graph_node_idx=n_lig + i, element=atom.GetSymbol(),
            residue_id=labels[i],
            atom_name=(info.GetName().strip() if info else atom.GetSymbol()),
            chain=((info.GetChainId() or "A").strip() or "A") if info else "A",
            resname=(info.GetResidueName().strip()[:3] if info else "UNK"),
            resseq=int(info.GetResidueNumber()) if info else i,
            x=float(pos.x), y=float(pos.y), z=float(pos.z),
        ))
    for j in range(n_lig):
        atom = ligand.GetAtomWithIdx(j)
        pos = ligand.GetConformer().GetAtomPosition(j)
        rows.append(dict(
            pdb_serial=n_prot + j + 1, role="ligand", ligand_atom_idx=j, protein_atom_idx=-1,
            graph_node_idx=j, element=atom.GetSymbol(), residue_id="Z:LIG1",
            atom_name=atom.GetSymbol(), chain="Z", resname="LIG", resseq=1,
            x=float(pos.x), y=float(pos.y), z=float(pos.z),
        ))
    return {"n_lig": n_lig, "n_prot": n_prot, "map_df": pd.DataFrame(rows)}
