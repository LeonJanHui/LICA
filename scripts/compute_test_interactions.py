#!/usr/bin/env python3
"""Extract non-covalent interaction fingerprints for CASF test complexes via PLIP.

Inputs: curated_pdbbind/{pdb}/{pdb}_protein.pdb + ligand sdf/mol2
Output: results/T-split/test_interactions.csv
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFreeSASA
from tqdm import tqdm

PROJECT_ROOT = Path("/home/dldx/Boltz-EGNN-Affinity")
CURATED = PROJECT_ROOT / "data" / "curated_pdbbind"
MYDATA_SPLIT = PROJECT_ROOT / "data" / "similarity" / "PDBbind_split_mydata.json"
OUT_CSV = PROJECT_ROOT / "results" / "T-split" / "test_interactions.csv"
PLIP_BIN = Path("/home/dldx/miniconda3/envs/boltz-egnn/bin/plip")


def load_test_ids(path: Path):
    data = json.load(open(path))
    for key in ("casf2016", "test", "casf2016_indep"):
        if key in data:
            return [str(x).strip().lower() for x in data[key]]
    raise KeyError(path)


def load_ligand(cid: str):
    folder = CURATED / cid
    mol2 = folder / f"{cid}_ligand.mol2"
    sdf = folder / f"{cid}_ligand.sdf"
    mol = None
    if mol2.is_file():
        mol = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=False)
    if mol is None and sdf.is_file():
        suppl = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    return mol


def element_from_pdb_line(line: str) -> str:
    if len(line) >= 78:
        raw = line[76:78].strip()
    else:
        raw = line[12:16].strip()
    letters = "".join(c for c in raw if c.isalpha())
    if not letters:
        return "C"
    if len(letters) == 1:
        return letters.upper()
    return letters[0].upper() + letters[1].lower()


def write_complex_pdb(cid: str, work_dir: Path) -> Path | None:
    prot = CURATED / cid / f"{cid}_protein.pdb"
    if not prot.is_file():
        return None
    lig = load_ligand(cid)
    if lig is None:
        return None

    out_lines = []
    with open(prot) as handle:
        for line in handle:
            if line.startswith("END"):
                continue
            out_lines.append(line.rstrip("\n"))

    serial = 90000
    for line in Chem.MolToPDBBlock(lig).splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        serial += 1
        name_alt = line[12:17]
        coords = line[30:54] if len(line) >= 54 else f"{0:8.3f}{0:8.3f}{0:8.3f}"
        elem = element_from_pdb_line(line)
        out_lines.append(
            f"HETATM{serial:5d} {name_alt}LIG Z{1:4d}    "
            f"{coords}  1.00  0.00          {elem:>2}"
        )
    out_lines.append("END")
    out_path = work_dir / f"{cid}_complex.pdb"
    out_path.write_text("\n".join(out_lines) + "\n")
    return out_path


def parse_plip_xml(xml_path: Path) -> dict:
    """Aggregate PLIP interaction counts for LIG / largest interacting site."""
    counts = {
        "n_hbonds": 0,
        "n_hbonds_backbone": 0,
        "n_hbonds_sidechain": 0,
        "n_salt_bridges": 0,
        "n_pi_pi": 0,
        "n_t_stacking": 0,
        "n_cation_pi": 0,
        "n_halogen": 0,
        "n_hydrophobic": 0,
        "n_water_bridges": 0,
        "n_metal_complexes": 0,
    }
    if not xml_path.is_file():
        return counts

    root = ET.parse(xml_path).getroot()
    best = None
    best_n = -1
    for bs in root.findall("bindingsite"):
        idents = bs.find("identifiers")
        hetid = (idents.findtext("hetid") if idents is not None else "") or ""
        inter = bs.find("interactions")
        if inter is None:
            continue
        n = sum(len(list(child)) for child in inter)
        # prefer LIG / UNL site; else richest site
        score = n + (1000 if hetid.upper() in {"LIG", "UNL"} else 0)
        if score > best_n:
            best_n = score
            best = inter

    if best is None:
        return counts

    hbonds = best.find("hydrogen_bonds")
    if hbonds is not None:
        for hb in hbonds:
            counts["n_hbonds"] += 1
            side = (hb.findtext("sidechain") or "").strip().lower()
            if side in {"true", "1", "yes"}:
                counts["n_hbonds_sidechain"] += 1
            else:
                counts["n_hbonds_backbone"] += 1

    for tag, key in [
        ("salt_bridges", "n_salt_bridges"),
        ("hydrophobic_interactions", "n_hydrophobic"),
        ("water_bridges", "n_water_bridges"),
        ("halogen_bonds", "n_halogen"),
        ("pi_cation_interactions", "n_cation_pi"),
        ("metal_complexes", "n_metal_complexes"),
    ]:
        node = best.find(tag)
        if node is not None:
            counts[key] = len(list(node))

    stacks = best.find("pi_stacks")
    if stacks is not None:
        for ps in stacks:
            typ = (ps.findtext("type") or "").strip().upper()
            # PLIP: P = parallel/pi-pi, T = T-shaped
            if typ == "T":
                counts["n_t_stacking"] += 1
            else:
                counts["n_pi_pi"] += 1

    return counts


def compute_delta_sasa(cid: str, complex_pdb: Path) -> float:
    """ΔSASA ≈ SASA(protein) + SASA(ligand) - SASA(complex)."""
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley
    except Exception:
        return float("nan")

    lig = load_ligand(cid)
    if lig is None:
        return float("nan")
    try:
        radii = rdFreeSASA.classifyAtoms(lig)
        lig_sasa = float(rdFreeSASA.CalcSASA(lig, radii))
    except Exception:
        lig_sasa = float("nan")

    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()
    try:
        prot = parser.get_structure("p", str(CURATED / cid / f"{cid}_protein.pdb"))
        sr.compute(prot, level="S")
        prot_sasa = float(sum(a.sasa for a in prot.get_atoms() if hasattr(a, "sasa")))
        cplx = parser.get_structure("c", str(complex_pdb))
        sr.compute(cplx, level="S")
        cplx_sasa = float(sum(a.sasa for a in cplx.get_atoms() if hasattr(a, "sasa")))
    except Exception:
        return float("nan")

    if any(not (x == x) for x in (prot_sasa, lig_sasa, cplx_sasa)):  # NaN check
        return float("nan")
    return prot_sasa + lig_sasa - cplx_sasa


def analyze_one(cid: str, plip_bin: str, scratch_root: str) -> dict:
    work = Path(scratch_root) / cid
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    row = {
        "pdb_id": cid,
        "success": 0,
        "error": "",
        "n_hbonds": 0,
        "n_hbonds_backbone": 0,
        "n_hbonds_sidechain": 0,
        "n_salt_bridges": 0,
        "n_pi_pi": 0,
        "n_t_stacking": 0,
        "n_cation_pi": 0,
        "n_halogen": 0,
        "n_hydrophobic": 0,
        "n_water_bridges": 0,
        "n_metal_complexes": 0,
        "delta_sasa": float("nan"),
    }

    try:
        complex_pdb = write_complex_pdb(cid, work)
        if complex_pdb is None:
            row["error"] = "missing_protein_or_ligand"
            return row

        out_dir = work / "plip_out"
        out_dir.mkdir(exist_ok=True)
        cmd = [
            plip_bin,
            "-f",
            str(complex_pdb),
            "-x",
            "-o",
            str(out_dir),
            "--name",
            cid,
            "--nofixfile",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180, check=False
        )
        xml_path = out_dir / f"{cid}.xml"
        if proc.returncode != 0 or not xml_path.is_file():
            # PLIP sometimes names xml from input stem
            xmls = list(out_dir.glob("*.xml"))
            xml_path = xmls[0] if xmls else xml_path
        if not xml_path.is_file():
            row["error"] = f"plip_failed:{proc.returncode}"
            return row

        counts = parse_plip_xml(xml_path)
        row.update(counts)
        row["delta_sasa"] = compute_delta_sasa(cid, complex_pdb)
        row["success"] = 1
    except subprocess.TimeoutExpired:
        row["error"] = "timeout"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}:{exc}"
    finally:
        # keep disk light
        shutil.rmtree(work, ignore_errors=True)

    # binary flags
    row["has_salt_bridge"] = int(row["n_salt_bridges"] > 0)
    row["has_water_bridge"] = int(row["n_water_bridges"] > 0)
    row["has_halogen"] = int(row["n_halogen"] > 0)
    row["has_pi_pi"] = int(row["n_pi_pi"] > 0)
    row["has_t_stacking"] = int(row["n_t_stacking"] > 0)
    row["has_cation_pi"] = int(row["n_cation_pi"] > 0)
    row["has_hbond"] = int(row["n_hbonds"] > 0)
    row["has_hydrophobic"] = int(row["n_hydrophobic"] > 0)
    row["has_metal"] = int(row["n_metal_complexes"] > 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_json", default=str(MYDATA_SPLIT))
    ap.add_argument("--out_csv", default=str(OUT_CSV))
    ap.add_argument("--plip", default=str(PLIP_BIN))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--scratch", default=str(PROJECT_ROOT / "results" / "T-split" / "_plip_scratch"))
    args = ap.parse_args()

    if not Path(args.plip).is_file():
        raise FileNotFoundError(
            f"PLIP not found at {args.plip}. Install with: pip install plip"
        )

    test_ids = load_test_ids(Path(args.split_json))
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[TEST] {len(test_ids)} complexes | workers={args.workers}")
    rows = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = {
            ex.submit(analyze_one, cid, args.plip, str(scratch)): cid for cid in test_ids
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="PLIP"):
            rows.append(fut.result())
            if len(rows) % 25 == 0:
                pd.DataFrame(rows).sort_values("pdb_id").to_csv(out_csv, index=False)

    df = pd.DataFrame(rows).sort_values("pdb_id").reset_index(drop=True)
    preferred = [
        "pdb_id",
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
        "success",
        "error",
    ]
    df = df[[c for c in preferred if c in df.columns]]
    df.to_csv(out_csv, index=False)

    ok = int(df["success"].sum())
    print(f"[DONE] {out_csv}  success={ok}/{len(df)}")
    if ok:
        print(df.loc[df.success == 1, [
            "n_hbonds", "n_salt_bridges", "n_pi_pi", "n_t_stacking",
            "n_cation_pi", "n_halogen", "n_hydrophobic", "n_water_bridges", "delta_sasa",
        ]].describe().to_string())
    fail = df.loc[df.success == 0]
    if len(fail):
        print("[FAIL] examples:")
        print(fail[["pdb_id", "error"]].head(10).to_string(index=False))

    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
