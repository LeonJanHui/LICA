from __future__ import annotations

import hashlib
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger
from scipy.spatial import cKDTree
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from features import (
    EDGE_INTERFACE,
    EDGE_LIGAND_COVALENT,
    EDGE_PROTEIN_COVALENT,
    EDGE_PROTEIN_SPATIAL,
    get_bond_edge_features,
    get_ligand_feature_matrix,
    get_protein_atom_features,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


class ComplexData(Data):
    """
    Custom PyG Data class to handle automatic offset of the custom attribute residue_id during batching.
    """

    def __inc__(self, key: str, value: torch.Tensor, *args, **kwargs):
        if key == "residue_id":
            # Only apply offset for protein residues (>=0); ligand nodes (-1) remain -1
            max_res = (
                int(self.residue_id.max().item())
                if self.residue_id.numel() > 0
                else -1
            )
            return max_res + 1 if max_res >= 0 else 0
        return super().__inc__(key, value, *args, **kwargs)


def _read_ligand(path: str, include_hs: bool = False) -> Optional[Chem.Mol]:
    """Lightweight fault‑tolerant ligand parsing pipeline (standard parse first, fall back to non‑sanitized parse on failure)"""
    ext = os.path.splitext(path)[1].lower()
    remove_hs = not include_hs
    mol = None

    # Stage 1: Standard RDKit parsing
    try:
        if ext == ".sdf":
            supplier = Chem.SDMolSupplier(
                path, sanitize=True, removeHs=remove_hs, strictParsing=False
            )
            mol = supplier[0] if len(supplier) else None
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(path, sanitize=True, removeHs=remove_hs)
    except Exception:
        mol = None

    if mol is not None and mol.GetNumAtoms() > 0 and mol.GetNumConformers() > 0:
        return mol

    # Stage 2: Fault‑tolerant fallback parsing (preserve original topology without strict chemical validation)
    try:
        if ext == ".sdf":
            supplier = Chem.SDMolSupplier(
                path, sanitize=False, removeHs=False, strictParsing=False
            )
            mol = supplier[0] if len(supplier) else None
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(path, sanitize=False, removeHs=False)

        if mol is not None and mol.GetNumAtoms() > 0:
            mol.UpdatePropertyCache(strict=False)
            if remove_hs:
                try:
                    mol = Chem.RemoveHs(mol, sanitize=False)
                    mol.UpdatePropertyCache(strict=False)
                except Exception:
                    pass
    except Exception:
        return None

    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        return None
    return mol


def _read_protein(path: str, include_hs: bool = False) -> Optional[Chem.Mol]:
    """Read PDB structure file"""
    try:
        mol = Chem.MolFromPDBFile(
            path, sanitize=False, removeHs=not include_hs, proximityBonding=True
        )
    except TypeError:
        mol = Chem.MolFromPDBFile(
            path, sanitize=False, removeHs=not include_hs
        )
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return mol


def _get_radius_pairs(
    pos1: np.ndarray,
    pos2: np.ndarray,
    radius: float,
    self_pairs: bool = False,
) -> List[Tuple[int, int]]:
    """Fast radius‑based neighbor pair query using cKDTree"""
    if len(pos1) == 0 or len(pos2) == 0:
        return []

    tree2 = cKDTree(pos2)
    if self_pairs:
        tree1 = cKDTree(pos1)
        pairs = tree1.query_pairs(radius)
        return sorted((int(i), int(j)) for i, j in pairs)
    else:
        pairs = []
        for i, neighbors in enumerate(tree2.query_ball_point(pos1, radius)):
            for j in neighbors:
                pairs.append((i, int(j)))
        return pairs


class ComplexGraphDataset(Dataset):
    """
    Protein‑ligand atomic‑level graph dataset construction pipeline
    """

    def __init__(
        self,
        root_dir: str,
        ledger_json: str,
        data_dict_json: str,
        save_dir: str,
        split_name: str = "train",
        interface_cutoff: float = 6.0,
        protein_spatial_cutoff: float = 6.0,
        label_col: str = "pk",
        include_hydrogens: bool = False,
        water_policy: str = "remove",
        water_cutoff: float = 4.0,
        transform=None,
        pre_transform=None,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.save_dir = os.path.abspath(save_dir)
        self.split_name = split_name
        self.interface_cutoff = float(interface_cutoff)
        self.protein_spatial_cutoff = float(protein_spatial_cutoff)
        self.label_col = label_col
        self.include_hydrogens = bool(include_hydrogens)
        self.water_policy = str(water_policy).lower()
        self.water_cutoff = float(water_cutoff)

        os.makedirs(self.save_dir, exist_ok=True)

        # 1. Load dataset index and labels
        with open(ledger_json, "r", encoding="utf-8") as f:
            ledger = json.load(f)
            self.complex_ids = [
                str(x).strip().lower() for x in ledger[split_name]
            ]

        with open(data_dict_json, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)
            data_dict = {
                str(k).strip().lower(): v for k, v in raw_dict.items()
            }

        self.label_map: Dict[str, float] = {}
        for cid in self.complex_ids:
            if cid in data_dict and self.label_col in data_dict[cid]:
                val = float(data_dict[cid][self.label_col])
                if np.isfinite(val):
                    self.label_map[cid] = val

        self.complex_ids = [
            cid for cid in self.complex_ids if cid in self.label_map
        ]

        # 2. Generate configuration hash (automatically switch to new cache version when config changes)
        config = {
            "split": split_name,
            "ic": self.interface_cutoff,
            "psc": self.protein_spatial_cutoff,
            "hs": self.include_hydrogens,
            "wp": self.water_policy,
            "wc": self.water_cutoff,
        }
        self.cache_key = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

        super().__init__(
            root=self.root_dir,
            transform=transform,
            pre_transform=pre_transform,
        )
        self._update_valid_ids()

    @property
    def processed_file_names(self) -> List[str]:
        return [f"manifest_{self.split_name}_{self.cache_key}.json"]

    def _graph_path(self, cid: str) -> str:
        return os.path.join(self.save_dir, f"{cid}__{self.cache_key}.pt")

    def _update_valid_ids(self):
        self.valid_ids = [
            cid for cid in self.complex_ids if os.path.exists(self._graph_path(cid))
        ]

    def len(self) -> int:
        return len(self.valid_ids)

    def get(self, idx: int) -> ComplexData:
        cid = self.valid_ids[idx]
        data = torch.load(self._graph_path(cid), map_location="cpu", weights_only=False)
        label = torch.tensor([self.label_map[cid]], dtype=torch.float32)
        data.y = label
        data.complex_id = cid
        return data

    def _build_graph(self, cid: str) -> ComplexData:
        folder = os.path.join(self.root_dir, cid)

        # Locate available file paths
        lig_path = next(
            (
                os.path.join(folder, f"{cid}_ligand.{e}")
                for e in ["sdf", "mol2"]
                if os.path.exists(os.path.join(folder, f"{cid}_ligand.{e}"))
            ),
            None,
        )
        prot_path = next(
            (
                os.path.join(folder, f"{cid}_{k}.pdb")
                for k in ["pocket", "protein"]
                if os.path.exists(os.path.join(folder, f"{cid}_{k}.pdb"))
            ),
            None,
        )

        if not lig_path or not prot_path:
            raise FileNotFoundError(f"Missing required ligand/protein file for {cid}")

        ligand = _read_ligand(lig_path, self.include_hydrogens)
        protein = _read_protein(prot_path, self.include_hydrogens)

        if ligand is None or ligand.GetNumAtoms() == 0:
            raise ValueError(f"Failed to load ligand for {cid}")
        if protein is None:
            raise ValueError(f"Failed to load protein for {cid}")

        # ---------------- Extract node features and coordinates ----------------
        lig_pos = np.asarray(
            ligand.GetConformer().GetPositions(), dtype=np.float32
        )
        lig_x = get_ligand_feature_matrix(ligand)
        num_lig = ligand.GetNumAtoms()

        valid_prot_indices: List[int] = []
        prot_x_list: List[np.ndarray] = []
        residue_keys: List[tuple] = []

        for idx, atom in enumerate(protein.GetAtoms()):
            res_info = atom.GetPDBResidueInfo()
            res_name = (
                res_info.GetResidueName().strip().upper() if res_info else "UNK"
            )

            # Water molecule handling logic
            if res_name in {"HOH", "WAT", "H2O", "DOD"}:
                if self.water_policy == "remove":
                    continue
                elif self.water_policy == "interface":
                    atom_pos = np.asarray(
                        protein.GetConformer().GetAtomPosition(idx),
                        dtype=np.float32,
                    )
                    min_dist = np.linalg.norm(
                        lig_pos - atom_pos[None, :], axis=1
                    ).min()
                    if min_dist >= self.water_cutoff:
                        continue

            if not self.include_hydrogens and atom.GetAtomicNum() == 1:
                continue

            valid_prot_indices.append(idx)
            prot_x_list.append(get_protein_atom_features(atom))

            res_key = (
                (
                    res_info.GetChainId().strip(),
                    int(res_info.GetResidueNumber()),
                    res_info.GetInsertionCode().strip(),
                    res_name,
                )
                if res_info
                else ("", -1, "", "UNK")
            )
            residue_keys.append(res_key)

        if not valid_prot_indices:
            raise ValueError(f"No valid protein atoms after filtering for {cid}")

        prot_pos = np.asarray(
            protein.GetConformer().GetPositions()[valid_prot_indices],
            dtype=np.float32,
        )
        prot_x = np.stack(prot_x_list, axis=0).astype(np.float32)
        num_prot = len(valid_prot_indices)

        # Concatenate node attributes
        pos = torch.from_numpy(
            np.concatenate([lig_pos, prot_pos], axis=0)
        ).float()
        x = torch.from_numpy(
            np.concatenate([lig_x, prot_x], axis=0)
        ).float()
        lig_mask = torch.zeros(num_lig + num_prot, dtype=torch.bool)
        lig_mask[:num_lig] = True

        # Build residue‑id mapping
        res_map: Dict[tuple, int] = {}
        prot_res_ids: List[int] = []
        for k in residue_keys:
            if k not in res_map:
                res_map[k] = len(res_map)
            prot_res_ids.append(res_map[k])

        residue_id = torch.full((num_lig + num_prot,), -1, dtype=torch.long)
        residue_id[num_lig:] = torch.tensor(prot_res_ids, dtype=torch.long)

        # ---------------- Edge construction (four edge types) ----------------
        sources, targets, types, attrs = [], [], [], []
        added_pairs = set()

        def add_edge(src: int, tgt: int, edge_type: int, feature: np.ndarray):
            if (src, tgt) in added_pairs:
                return
            added_pairs.add((src, tgt))
            sources.append(src)
            targets.append(tgt)
            types.append(edge_type)
            attrs.append(feature)

        # (1) Ligand covalent edges
        for bond in ligand.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = get_bond_edge_features(bond, same_residue=False)
            add_edge(u, v, EDGE_LIGAND_COVALENT, feat)
            add_edge(v, u, EDGE_LIGAND_COVALENT, feat)

        # (2) Protein covalent edges
        prot_idx_map = {
            orig: num_lig + i for i, orig in enumerate(valid_prot_indices)
        }
        for bond in protein.GetBonds():
            u_orig, v_orig = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u_orig in prot_idx_map and v_orig in prot_idx_map:
                u, v = prot_idx_map[u_orig], prot_idx_map[v_orig]
                same_res = residue_id[u].item() == residue_id[v].item()
                feat = get_bond_edge_features(bond, same_residue=same_res)
                add_edge(u, v, EDGE_PROTEIN_COVALENT, feat)
                add_edge(v, u, EDGE_PROTEIN_COVALENT, feat)

        # (3) Protein non‑covalent spatial edges (skip existing covalent edges)
        spatial_pairs = _get_radius_pairs(
            prot_pos, prot_pos, self.protein_spatial_cutoff, self_pairs=True
        )
        empty_feat = get_bond_edge_features(None, same_residue=False)

        for i, j in spatial_pairs:
            u, v = num_lig + i, num_lig + j
            if (u, v) in added_pairs:
                continue
            same_res = residue_id[u].item() == residue_id[v].item()
            feat = get_bond_edge_features(None, same_residue=same_res)
            add_edge(u, v, EDGE_PROTEIN_SPATIAL, feat)
            add_edge(v, u, EDGE_PROTEIN_SPATIAL, feat)

        # (4) Ligand‑protein interface edges
        iface_pairs = _get_radius_pairs(
            lig_pos, prot_pos, self.interface_cutoff, self_pairs=False
        )
        if not iface_pairs:
            raise ValueError(f"No interface edges within {self.interface_cutoff} Å for {cid}")

        for i, j in iface_pairs:
            u, v = i, num_lig + j
            add_edge(u, v, EDGE_INTERFACE, empty_feat)
            add_edge(v, u, EDGE_INTERFACE, empty_feat)

        # ---------------- Wrap into ComplexData ----------------
        return ComplexData(
            x=x,
            pos=pos,
            edge_index=torch.tensor([sources, targets], dtype=torch.long),
            edge_attr=torch.from_numpy(np.stack(attrs, axis=0)).float(),
            edge_type=torch.tensor(types, dtype=torch.long),
            lig_mask=lig_mask,
            residue_id=residue_id,
        )

    def process(self):
        manifest_path = self.processed_paths[0]
        success_count, failures = 0, {}

        for cid in tqdm(self.complex_ids, desc=f"Building {self.split_name} graphs"):
            graph_path = self._graph_path(cid)
            if os.path.exists(graph_path):
                success_count += 1
                continue

            try:
                graph = self._build_graph(cid)
                tmp_path = graph_path + ".tmp"
                torch.save(graph, tmp_path)
                os.replace(tmp_path, graph_path)
                success_count += 1
            except Exception as e:
                failures[cid] = str(e)

        manifest = {
            "cache_key": self.cache_key,
            "split": self.split_name,
            "total": len(self.complex_ids),
            "success": success_count,
            "failed": len(failures),
            "failures": failures,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        self._update_valid_ids()