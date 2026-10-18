from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures


FEATURE_SCHEMA_VERSION = "semantic-atom-v4-protein-spatial"

# All nodes share the same "isotopic‑synonymous" feature layout.
ELEMENTS: Tuple[str, ...] = (
    "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "B", "SE", "SI",
    "METAL", "UNKNOWN",
)
DEGREES: Tuple[object, ...] = (0, 1, 2, 3, 4, 5, "UNKNOWN")
HYBRIDIZATIONS: Tuple[str, ...] = (
    "S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED",
)
NUM_HS: Tuple[object, ...] = (0, 1, 2, 3, "UNKNOWN")
RESIDUES: Tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK", "LIG",
)
ATOM_ROLES: Tuple[str, ...] = (
    "BACKBONE", "SIDECHAIN", "METAL", "HETERO", "WATER", "UNKNOWN",
)

# Four edge relation types:
# 0 Real covalent bonds of ligand
# 1 Real / inferred covalent bonds of protein
# 2 Intra‑protein non‑covalent spatial edges
# 3 Ligand‑protein interface edges
EDGE_LIGAND_COVALENT = 0
EDGE_PROTEIN_COVALENT = 1
EDGE_PROTEIN_SPATIAL = 2
EDGE_INTERFACE = 3
EDGE_TYPE_NAMES: Tuple[str, ...] = (
    "ligand_covalent",
    "protein_covalent",
    "protein_spatial",
    "interface",
)
NUM_EDGE_TYPES = len(EDGE_TYPE_NAMES)

BOND_STATES: Tuple[str, ...] = (
    "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "NONE",
)

# 14 element + 7 degree + 7 hybridization + 1 charge + 4 chemistry flags
# + 5 hydrogen count + 2 node type + 22 residue + 6 atom role = 68
NODE_FEATURE_DIM = (
    len(ELEMENTS)
    + len(DEGREES)
    + len(HYBRIDIZATIONS)
    + 1
    + 4
    + len(NUM_HS)
    + 2
    + len(RESIDUES)
    + len(ATOM_ROLES)
)

# 5 bond type + conjugated + in‑ring + same‑residue = 8
EDGE_STATIC_DIM = len(BOND_STATES) + 3

# Common metal and metalloid atomic numbers. They are no longer all collapsed to Unknown.
METAL_ATOMIC_NUMBERS: Set[int] = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83,
}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

# Approximate donor‑acceptor rules based on standard residue / atom names.
# Produces fewer obvious errors such as "amide nitrogen as acceptor" compared
# with raw element‑based judgment.
PROTEIN_DONOR_ATOMS = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("ASN", "ND2"), ("GLN", "NE2"), ("HIS", "ND1"), ("HIS", "NE2"),
    ("LYS", "NZ"), ("SER", "OG"), ("THR", "OG1"), ("TRP", "NE1"),
    ("TYR", "OH"), ("CYS", "SG"),
}
PROTEIN_ACCEPTOR_ATOMS = {
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"), ("GLN", "OE1"), ("HIS", "ND1"), ("HIS", "NE2"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"),
    ("MET", "SD"),
}
WATER_RESIDUES = {"HOH", "WAT", "H2O", "DOD"}
AROMATIC_RESIDUES = {"HIS", "PHE", "TRP", "TYR"}
AROMATIC_SIDECHAIN_ATOMS = {
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
}


def _one_hot(value: object, allowable: Sequence[object]) -> list[float]:
    """Map unknown values to the last bin uniformly."""
    normalized = value if value in allowable else allowable[-1]
    return [1.0 if normalized == item else 0.0 for item in allowable]


def _safe_call(func, default):
    try:
        return func()
    except Exception:
        return default


def _element_bucket(atom: Chem.Atom) -> str:
    atomic_num = atom.GetAtomicNum()
    if atomic_num in METAL_ATOMIC_NUMBERS:
        return "METAL"
    symbol = atom.GetSymbol().upper()
    return symbol if symbol in ELEMENTS else "UNKNOWN"


def _hybridization_bucket(atom: Chem.Atom) -> str:
    hybridization = _safe_call(lambda: str(atom.GetHybridization()), "UNSPECIFIED")
    hybridization = hybridization.upper()
    return hybridization if hybridization in HYBRIDIZATIONS else "UNSPECIFIED"


def _formal_charge(atom: Chem.Atom) -> float:
    """Clip to prevent extreme values from anomalous structures dominating training."""
    charge = float(_safe_call(atom.GetFormalCharge, 0))
    return float(np.clip(charge, -3.0, 3.0) / 3.0)


def _num_h_bucket(atom: Chem.Atom) -> object:
    num_h = int(_safe_call(atom.GetTotalNumHs, 0))
    return num_h if num_h in NUM_HS[:-1] else "UNKNOWN"


@lru_cache(maxsize=1)
def _feature_factory():
    fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(fdef_path)


def ligand_pharmacophore_atom_sets(mol: Chem.Mol) -> tuple[Set[int], Set[int]]:
    """
    Identify donors / acceptors using RDKit standard BaseFeatures instead of
    coarse element‑only rules. Returns empty sets on failure; caller can still
    complete graph construction.
    """
    donors: Set[int] = set()
    acceptors: Set[int] = set()
    try:
        for feature in _feature_factory().GetFeaturesForMol(mol):
            family = feature.GetFamily()
            atom_ids = set(feature.GetAtomIds())
            if family == "Donor":
                donors.update(atom_ids)
            elif family == "Acceptor":
                acceptors.update(atom_ids)
    except Exception:
        pass
    return donors, acceptors


def _common_atom_features(
    atom: Chem.Atom,
    *,
    is_ligand: bool,
    residue_name: str,
    atom_role: str,
    donor: bool,
    acceptor: bool,
    aromatic_override: Optional[bool] = None,
    ring_override: Optional[bool] = None,
) -> np.ndarray:
    element = _element_bucket(atom)
    degree = int(_safe_call(atom.GetDegree, 0))
    degree = degree if degree in DEGREES[:-1] else "UNKNOWN"
    hybridization = _hybridization_bucket(atom)

    aromatic = (
        bool(aromatic_override)
        if aromatic_override is not None
        else bool(_safe_call(atom.GetIsAromatic, False))
    )
    in_ring = (
        bool(ring_override)
        if ring_override is not None
        else bool(_safe_call(atom.IsInRing, False))
    )

    residue_name = residue_name if residue_name in RESIDUES else "UNK"
    atom_role = atom_role if atom_role in ATOM_ROLES else "UNKNOWN"

    # Explicit concatenation guarantees identical semantics for ligand and protein nodes.
    base = (
        _one_hot(element, ELEMENTS)
        + _one_hot(degree, DEGREES)
        + _one_hot(hybridization, HYBRIDIZATIONS)
        + [_formal_charge(atom)]
        + [
            float(aromatic),
            float(in_ring),
            float(donor),
            float(acceptor),
        ]
        + _one_hot(_num_h_bucket(atom), NUM_HS)
    )
    node_type = [1.0, 0.0] if is_ligand else [0.0, 1.0]
    vector = base + node_type + _one_hot(residue_name, RESIDUES) + _one_hot(atom_role, ATOM_ROLES)

    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (NODE_FEATURE_DIM,):
        raise RuntimeError(
            f"Node feature dimension mismatch: got {array.shape}, expected {(NODE_FEATURE_DIM,)}"
        )
    return array


def get_ligand_atom_features(
    atom: Chem.Atom,
    *,
    donor_atom_ids: Optional[Set[int]] = None,
    acceptor_atom_ids: Optional[Set[int]] = None,
) -> np.ndarray:
    atom_idx = atom.GetIdx()
    donor_atom_ids = donor_atom_ids or set()
    acceptor_atom_ids = acceptor_atom_ids or set()
    return _common_atom_features(
        atom,
        is_ligand=True,
        residue_name="LIG",
        atom_role="HETERO",
        donor=atom_idx in donor_atom_ids,
        acceptor=atom_idx in acceptor_atom_ids,
    )


def get_ligand_feature_matrix(mol: Chem.Mol) -> np.ndarray:
    donors, acceptors = ligand_pharmacophore_atom_sets(mol)
    return np.stack(
        [
            get_ligand_atom_features(
                atom,
                donor_atom_ids=donors,
                acceptor_atom_ids=acceptors,
            )
            for atom in mol.GetAtoms()
        ],
        axis=0,
    )


def _protein_donor_acceptor(residue_name: str, atom_name: str) -> tuple[bool, bool]:
    donor = (residue_name, atom_name) in PROTEIN_DONOR_ATOMS
    acceptor = (residue_name, atom_name) in PROTEIN_ACCEPTOR_ATOMS

    # Standard backbone: N is usually donor (except Pro), O/OXT are acceptors.
    if atom_name == "N" and residue_name != "PRO":
        donor = True
    if atom_name in {"O", "OXT"}:
        acceptor = True
    return donor, acceptor


def get_protein_atom_features(atom: Chem.Atom) -> np.ndarray:
    residue_info = atom.GetPDBResidueInfo()
    if residue_info is None:
        residue_name = "UNK"
        atom_name = ""
        is_hetero = True
    else:
        residue_name = residue_info.GetResidueName().strip().upper() or "UNK"
        atom_name = residue_info.GetName().strip().upper()
        is_hetero = bool(residue_info.GetIsHeteroAtom())

    is_water = residue_name in WATER_RESIDUES
    is_metal = atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS
    if is_water:
        role = "WATER"
    elif is_metal:
        role = "METAL"
    elif atom_name in BACKBONE_ATOMS:
        role = "BACKBONE"
    elif residue_name in RESIDUES[:-2]:
        role = "SIDECHAIN"
    elif is_hetero:
        role = "HETERO"
    else:
        role = "UNKNOWN"

    donor, acceptor = _protein_donor_acceptor(residue_name, atom_name)
    if is_water and atom.GetAtomicNum() == 8:
        donor, acceptor = True, True
    aromatic = (
        residue_name in AROMATIC_RESIDUES
        and atom_name in AROMATIC_SIDECHAIN_ATOMS.get(residue_name, set())
    )
    in_ring = aromatic

    return _common_atom_features(
        atom,
        is_ligand=False,
        residue_name=residue_name if residue_name in RESIDUES else "UNK",
        atom_role=role,
        donor=donor,
        acceptor=acceptor,
        aromatic_override=aromatic,
        ring_override=in_ring,
    )


def get_bond_edge_features(
    bond: Optional[Chem.Bond],
    *,
    same_residue: bool = False,
) -> np.ndarray:
    if bond is None:
        bond_state = "NONE"
        conjugated = False
        in_ring = False
    else:
        bond_type = bond.GetBondType()
        if bond.GetIsAromatic() or bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_state = "AROMATIC"
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_state = "DOUBLE"
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_state = "TRIPLE"
        else:
            bond_state = "SINGLE"
        conjugated = bool(_safe_call(bond.GetIsConjugated, False))
        in_ring = bool(_safe_call(bond.IsInRing, False))

    vector = (
        _one_hot(bond_state, BOND_STATES)
        + [float(conjugated), float(in_ring), float(same_residue)]
    )
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (EDGE_STATIC_DIM,):
        raise RuntimeError(
            f"Edge feature dimension mismatch: got {array.shape}, expected {(EDGE_STATIC_DIM,)}"
        )
    return array


def gaussian_rbf(
    distances,
    *,
    num_centers: int = 16,
    start: float = 0.0,
    stop: float = 6.0,
):
    """
    Compatible implementation of Gaussian RBF with smooth cosine cutoff.

    Gaussian width equals spacing between adjacent centers to ensure sufficient overlap.
    Cosine envelope decays continuously to zero at stop to mitigate feature jumps caused
    by hard 6‑Å cutoff. The model layer recomputes this feature from coordinates;
    distance‑based RBF is not cached in datasets.
    """
    import math
    import torch

    if num_centers < 2:
        raise ValueError("num_centers must be >= 2")
    if stop <= start:
        raise ValueError("stop must be greater than start")

    centers = torch.linspace(
        start,
        stop,
        num_centers,
        device=distances.device,
        dtype=distances.dtype,
    )
    spacing = (stop - start) / (num_centers - 1)
    coefficient = -0.5 / max(spacing * spacing, 1e-8)
    rbf = torch.exp(
        coefficient * (distances.unsqueeze(-1) - centers) ** 2
    )

    scaled = ((distances - start) / (stop - start)).clamp(0.0, 1.0)
    cutoff = 0.5 * (torch.cos(math.pi * scaled) + 1.0)
    cutoff = cutoff * ((distances >= start) & (distances < stop)).to(distances.dtype)
    return rbf * cutoff.unsqueeze(-1)


# Alias for backward compatibility with legacy code.
get_rbf_features = gaussian_rbf