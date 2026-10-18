#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set

from rdkit import Chem

PLIP = Path("/home/dldx/miniconda3/envs/boltz-egnn/bin/plip")
CURATED = Path("/home/dldx/Boltz-EGNN-Affinity/data/curated_pdbbind")


def run_plip_residues(cid: str, scratch: Path) -> Set[str]:
    if not PLIP.is_file():
        return set()
    work = scratch / cid
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    prot = CURATED / cid / f"{cid}_protein.pdb"
    if not prot.is_file():
        prot = CURATED / cid / f"{cid}_pocket.pdb"
    lig_sdf = CURATED / cid / f"{cid}_ligand.sdf"
    lines = [ln for ln in prot.read_text(errors="ignore").splitlines() if ln.startswith(("ATOM", "HETATM"))]
    mol = Chem.SDMolSupplier(str(lig_sdf), removeHs=False, sanitize=False)[0]
    if mol is None:
        return set()
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        el = atom.GetSymbol()
        lines.append(
            f"HETATM{i+1:5d} {el:<3s}  LIG A   1    "
            f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00           {el:>2s}"
        )
    lines.append("END")
    cplx = work / f"{cid}_complex.pdb"
    cplx.write_text("\n".join(lines) + "\n")
    try:
        subprocess.run([str(PLIP), "-f", str(cplx), "-x", "-o", str(work)], check=True, capture_output=True, timeout=180)
    except Exception:
        return set()
    xmls = list(work.glob("*.xml"))
    if not xmls:
        return set()
    root = ET.parse(xmls[0]).getroot()
    best, best_n = None, -1
    for bs in root.findall("bindingsite"):
        inter = bs.find("interactions")
        if inter is None:
            continue
        n = sum(len(list(c)) for c in inter)
        if n > best_n:
            best, best_n = inter, n
    residues: Set[str] = set()
    if best is None:
        return residues
    for tag in (
        "hydrogen_bonds", "salt_bridges", "hydrophobic_interactions",
        "pi_stacks", "pi_cation_interactions", "halogen_bonds", "water_bridges",
    ):
        node = best.find(tag)
        if node is None:
            continue
        for item in node:
            nr = (item.findtext("resnr") or "").strip()
            ch = (item.findtext("reschain") or "").strip()
            rt = (item.findtext("restype") or "").strip()
            if nr:
                residues.add(f"{ch}:{rt}{nr}" if ch else f"{rt}{nr}")
    return residues


def label_res(rid: str) -> str:
    s = rid.split(":")[-1]
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    return f"{s[:i].title()}{s[i:]}"
