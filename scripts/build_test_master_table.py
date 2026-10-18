#!/usr/bin/env python3
"""Merge prediction, ligand/pocket similarity, interactions, and RDKit ligand props.

Actual project inputs (not the generic CSV names in the protocol draft):
  - results/T-split/tanimoto_model_predictions.csv
  - results/T-split/fig1_max_tanimoto_per_complex.csv
  - results/T-split/test_pocket_similarity.csv
  - results/T-split/test_interactions.csv
  - optional: results/T-split/fig3_per_complex_scaffold.csv

Output:
  results/T-split/test_master_table.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from tqdm import tqdm

PROJECT_ROOT = Path("/home/dldx/Boltz-EGNN-Affinity")
CURATED = PROJECT_ROOT / "data" / "curated_pdbbind"
T_SPLIT = PROJECT_ROOT / "results" / "T-split"


def load_ligand(cid: str):
    folder = CURATED / cid
    mol = None
    mol2 = folder / f"{cid}_ligand.mol2"
    sdf = folder / f"{cid}_ligand.sdf"
    if mol2.is_file():
        mol = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=False)
    if mol is None and sdf.is_file():
        suppl = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def ligand_properties(cid: str) -> dict:
    mol = load_ligand(cid)
    out = {
        "mw": np.nan,
        "logp": np.nan,
        "tpsa": np.nan,
        "hbd": np.nan,
        "hba": np.nan,
        "n_rotatable_bonds": np.nan,
        "n_aromatic_rings": np.nan,
        "formal_charge": np.nan,
        "n_heavy_atoms": np.nan,
        "smiles": "",
    }
    if mol is None:
        return out
    try:
        out["mw"] = float(Descriptors.MolWt(mol))
        out["logp"] = float(Crippen.MolLogP(mol))
        out["tpsa"] = float(Descriptors.TPSA(mol))
        out["hbd"] = int(Lipinski.NumHDonors(mol))
        out["hba"] = int(Lipinski.NumHAcceptors(mol))
        out["n_rotatable_bonds"] = int(Lipinski.NumRotatableBonds(mol))
        out["n_aromatic_rings"] = int(rdMolDescriptors.CalcNumAromaticRings(mol))
        out["formal_charge"] = int(Chem.GetFormalCharge(mol))
        out["n_heavy_atoms"] = int(mol.GetNumHeavyAtoms())
        out["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    except Exception:
        pass
    return out


def normalize_id_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pdb_id" in df.columns and "pdbid" not in df.columns:
        df = df.rename(columns={"pdb_id": "pdbid"})
    df["pdbid"] = df["pdbid"].astype(str).str.strip().str.lower()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default=str(T_SPLIT / "tanimoto_model_predictions.csv"))
    ap.add_argument("--ligand_sim_csv", default=str(T_SPLIT / "fig1_max_tanimoto_per_complex.csv"))
    ap.add_argument("--pocket_sim_csv", default=str(T_SPLIT / "test_pocket_similarity.csv"))
    ap.add_argument("--interactions_csv", default=str(T_SPLIT / "test_interactions.csv"))
    ap.add_argument("--scaffold_csv", default=str(T_SPLIT / "fig3_per_complex_scaffold.csv"))
    ap.add_argument("--out_csv", default=str(T_SPLIT / "test_master_table.csv"))
    args = ap.parse_args()

    pred = normalize_id_col(pd.read_csv(args.pred_csv))
    # ensure abs_error
    if "abs_error" not in pred.columns:
        pred["abs_error"] = (pred["y_pred"] - pred["y_true"]).abs()
    else:
        pred["abs_error"] = pred["abs_error"].astype(float)
    pred["signed_error"] = pred["y_pred"] - pred["y_true"]
    pred = pred.rename(
        columns={
            "y_true": "experimental",
            "y_pred": "predicted",
        }
    )

    lig_sim = normalize_id_col(pd.read_csv(args.ligand_sim_csv)).rename(
        columns={
            "max_tanimoto": "max_ligand_tanimoto",
            "top10_tanimoto": "top10_ligand_tanimoto",
            "max_random": "max_ligand_tanimoto_random_train",
            "top10_random": "top10_ligand_tanimoto_random_train",
        }
    )
    pocket = normalize_id_col(pd.read_csv(args.pocket_sim_csv))
    inter = normalize_id_col(pd.read_csv(args.interactions_csv))
    # drop plip bookkeeping from master if present
    inter = inter.drop(columns=[c for c in ("success", "error") if c in inter.columns])

    master = pred.merge(lig_sim, on="pdbid", how="left")
    master = master.merge(pocket, on="pdbid", how="left")
    master = master.merge(inter, on="pdbid", how="left")

    scaffold_path = Path(args.scaffold_csv)
    if scaffold_path.is_file():
        scaf = normalize_id_col(pd.read_csv(scaffold_path))
        keep = [
            c
            for c in (
                "pdbid",
                "scaffold",
                "scaffold_group",
                "scaffold_in_random_train",
                "scaffold_in_tanimoto_train",
            )
            if c in scaf.columns
        ]
        master = master.merge(scaf[keep], on="pdbid", how="left")

    prop_rows = []
    for cid in tqdm(master["pdbid"].tolist(), desc="RDKit props"):
        row = ligand_properties(cid)
        row["pdbid"] = cid
        prop_rows.append(row)
    props = pd.DataFrame(prop_rows)
    master = master.merge(props, on="pdbid", how="left")

    # friendly alias
    master = master.rename(columns={"pdbid": "pdb_id"})

    # column order: ids / labels / errors / ligand sim / pocket / interactions / chem props
    front = [
        "pdb_id",
        "experimental",
        "predicted",
        "abs_error",
        "signed_error",
        "max_ligand_tanimoto",
        "top10_ligand_tanimoto",
        "max_ligand_tanimoto_random_train",
        "top10_ligand_tanimoto_random_train",
        "max_pocket_tmscore",
        "nearest_train_pdb_pocket",
        "max_pocket_identity",
        "nearest_train_pdb_identity",
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
        "scaffold",
        "scaffold_group",
    ]
    ordered = [c for c in front if c in master.columns]
    ordered += [c for c in master.columns if c not in ordered]
    master = master[ordered].sort_values("pdb_id").reset_index(drop=True)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out, index=False)
    print(f"[DONE] {out}  n={len(master)}  cols={len(master.columns)}")
    print(master[["pdb_id", "experimental", "predicted", "abs_error", "max_ligand_tanimoto", "max_pocket_tmscore", "mw", "logp"]].head(5).to_string(index=False))
    print(
        "missing ligand sim:",
        master["max_ligand_tanimoto"].isna().sum(),
        "| pocket:",
        master["max_pocket_tmscore"].isna().sum(),
        "| mw:",
        master["mw"].isna().sum(),
    )


if __name__ == "__main__":
    main()
