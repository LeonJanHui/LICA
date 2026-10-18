import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from rdkit import Chem


def get_dynamic_atom_labels(
    data: Any, 
    root_dir: Optional[str] = None,
    include_hydrogens: bool = False,
    water_policy: str = "remove"
) -> Tuple[List[str], List[str]]:
    """
    不重新建图的前提下，根据 data.complex_id 动态读取原始 SDF/PDB 文件，
    提取与图节点 1:1 对应的配体元素符号列表和蛋白质残基文本列表。
    """
    ligand_symbols = []
    protein_residues = []

    # 1. 尝试从 data 中读取 complex_id 或 pdb_id
    sample_id = None
    if hasattr(data, "complex_id"):
        cid = data.complex_id
        sample_id = str(cid[0] if isinstance(cid, (list, tuple)) else cid)
    elif hasattr(batch_or_data := data, "pdb_id"):
        pid = data.pdb_id
        sample_id = str(pid[0] if isinstance(pid, (list, tuple)) else pid)

    if not sample_id or not root_dir or not os.path.exists(root_dir):
        return ligand_symbols, protein_residues

    folder = os.path.join(root_dir, sample_id)
    if not os.path.exists(folder):
        return ligand_symbols, protein_residues

    # 2. 定位原始配体与口袋文件
    ligand_candidates = [
        os.path.join(folder, f"{sample_id}_ligand.sdf"),
        os.path.join(folder, f"{sample_id}_ligand.mol2"),
    ]
    protein_candidates = [
        os.path.join(folder, f"{sample_id}_pocket.pdb"),
        os.path.join(folder, f"{sample_id}_protein.pdb"),
    ]

    ligand_path = next((p for p in ligand_candidates if os.path.exists(p)), None)
    protein_path = next((p for p in protein_candidates if os.path.exists(p)), None)

    # 3. 动态解析配体元素 (与 dataset.py 的加氢/去氢逻辑对齐)
    if ligand_path:
        try:
            lower_path = ligand_path.lower()
            remove_hs = not include_hydrogens
            mol = None
            if lower_path.endswith(".sdf"):
                suppl = Chem.SDMolSupplier(ligand_path, sanitize=False, removeHs=remove_hs, strictParsing=False)
                mol = suppl[0] if len(suppl) else None
            elif lower_path.endswith(".mol2"):
                mol = Chem.MolFromMol2File(ligand_path, sanitize=False, removeHs=remove_hs, cleanupSubstructures=False)
            
            if mol:
                ligand_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        except Exception:
            ligand_symbols = []

    # 4. 动态解析蛋白口袋残基 (与 dataset.py 水分子/氢原子过滤对齐)
    if protein_path:
        try:
            prot_mol = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=not include_hydrogens)
            if prot_mol:
                for atom in prot_mol.GetAtoms():
                    info = atom.GetPDBResidueInfo()
                    if info:
                        resname = info.GetResidueName().strip().upper()
                        # 过滤水分子
                        if water_policy == "remove" and resname in {"HOH", "WAT", "H2O", "DOD"}:
                            continue
                        # 过滤氢原子
                        if not include_hydrogens and atom.GetAtomicNum() == 1:
                            continue
                        
                        chain = info.GetChainId().strip()
                        resnum = info.GetResidueNumber()
                        ins_code = info.GetInsertionCode().strip()
                        res_id_str = f"{chain}:{resname}{resnum}{ins_code}"
                        protein_residues.append(res_id_str)
        except Exception:
            protein_residues = []

    return ligand_symbols, protein_residues


class NativeAttentionExtractor:
    """
    针对 PureGeometricAffinityNetwork 的内生注意力提取与解析器
    """
    def __init__(self, model: torch.nn.Module, device: str = "cuda", root_dir: Optional[str] = None):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.root_dir = root_dir

    @torch.no_grad()
    def extract_from_data(
        self, 
        data: Any, 
        use_raw_scores: bool = False,
        root_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从单条图数据（PyG Data）中提取三个轨道的注意力指标
        """
        data = data.to(self.device)
        effective_root_dir = root_dir if root_dir is not None else self.root_dir

        # 1. 调用模型的前向传播，触发可解释性分支
        affinity, interp_dict = self.model(data, return_interpretability=True)
        
        # 强转为标量 float，消除格式化报错
        if isinstance(affinity, torch.Tensor):
            aff_val = float(affinity.detach().cpu().squeeze().item())
        else:
            aff_val = float(np.squeeze(affinity))

        results = {
            "predicted_affinity": aff_val,
            "tracks": {}
        }

        # 选用的得分类型：跨复合物对比用 raw_scores，单图可视化/相对贡献用 weights
        score_key = "raw_scores" if use_raw_scores else "weights"

        # 动态解析标签
        lig_symbols, prot_res_list = get_dynamic_atom_labels(data, root_dir=effective_root_dir)
        num_ligand_nodes = int(data.lig_mask.sum().item()) if hasattr(data, "lig_mask") else len(lig_symbols)

        # ----------------------------------------------------
        # 2. 解析 Track A: 配体内部原子注意力 (Pharmacophore Level)
        # ----------------------------------------------------
        if "ligand" in interp_dict:
            lig_info = interp_dict["ligand"]
            lig_indices = lig_info["node_indices"].cpu().numpy()
            lig_scores = lig_info[score_key].squeeze(-1).cpu().numpy()
            
            # 安全映射元素符号
            lig_elems = [
                lig_symbols[idx] if idx < len(lig_symbols) else "C"
                for idx in range(len(lig_indices))
            ]
            
            df_ligand = pd.DataFrame({
                "global_node_idx": lig_indices,
                "ligand_atom_idx": np.arange(len(lig_indices)),
                "ligand_element": lig_elems,                     # ➕ 配体元素 (如 'N', 'O')
                "attention_score": lig_scores
            }).sort_values(by="attention_score", ascending=False)
            
            results["tracks"]["track_a_ligand"] = df_ligand

        # ----------------------------------------------------
        # 3. 解析 Track B: 口袋节点注意力 (Hotspot Level)
        # ----------------------------------------------------
        if "pocket" in interp_dict:
            pocket_info = interp_dict["pocket"]
            pocket_indices = pocket_info["node_indices"].cpu().numpy()
            pocket_scores = pocket_info[score_key].squeeze(-1).cpu().numpy()
            
            df_pocket = pd.DataFrame({
                "global_node_idx": pocket_indices,
                "attention_score": pocket_scores
            })
            
            # 解析残基文本
            res_ids = []
            for global_idx in pocket_indices:
                local_prot_idx = global_idx - num_ligand_nodes
                if 0 <= local_prot_idx < len(prot_res_list):
                    res_ids.append(prot_res_list[local_prot_idx])
                elif hasattr(data, "residue_keys") and data.residue_keys is not None:
                    key = data.residue_keys[global_idx]
                    if key:
                        chain, resnum, ins_code, resname = key
                        res_ids.append(f"{chain}:{resname}{resnum}{ins_code if ins_code else ''}")
                    else:
                        res_ids.append(f"Atom_{global_idx}")
                else:
                    res_ids.append(f"Atom_{global_idx}")

            df_pocket["residue_id"] = res_ids
            # 按残基聚合并取最大注意力
            df_pocket_res = df_pocket.groupby("residue_id")["attention_score"].max().reset_index()
            df_pocket_res = df_pocket_res.sort_values(by="attention_score", ascending=False)
            results["tracks"]["track_b_pocket_residue"] = df_pocket_res

            df_pocket = df_pocket.sort_values(by="attention_score", ascending=False)
            results["tracks"]["track_b_pocket_atom"] = df_pocket

        # ----------------------------------------------------
        # 4. 解析 Track C: 显式界面原子对接触注意力 (Pairwise Interaction)
        # ----------------------------------------------------
        if "interface_pairs" in interp_dict:
            pair_info = interp_dict["interface_pairs"]
            
            lig_pair_nodes = pair_info["ligand_node_indices"].cpu().numpy()
            prot_pair_nodes = pair_info["protein_node_indices"].cpu().numpy()
            pair_scores = pair_info[score_key].squeeze(-1).cpu().numpy()
            distances = pair_info["distances"].cpu().numpy()
            distance_gates = pair_info["distance_gates"].cpu().numpy()

            # 映射元素符号与残基文本
            pair_lig_elems = [
                lig_symbols[idx] if idx < len(lig_symbols) else "C"
                for idx in lig_pair_nodes
            ]
            
            pair_prot_residues = []
            for global_idx in prot_pair_nodes:
                local_prot_idx = global_idx - num_ligand_nodes
                if 0 <= local_prot_idx < len(prot_res_list):
                    pair_prot_residues.append(prot_res_list[local_prot_idx])
                else:
                    pair_prot_residues.append(f"Prot_Atom_{global_idx}")

            df_pairs = pd.DataFrame({
                "ligand_node_idx": lig_pair_nodes,
                "ligand_element": pair_lig_elems,               # ➕ 配体元素 (如 'O')
                "protein_node_idx": prot_pair_nodes,
                "protein_residue": pair_prot_residues,           # ➕ 蛋白残基 (如 'A:LYS165')
                "attention_score": pair_scores,
                "distance_A": distances,
                "distance_gate": distance_gates
            }).sort_values(by="attention_score", ascending=False)
            
            results["tracks"]["track_c_pairs"] = df_pairs
            
            # 构建 M x N 的二维注意力热力图矩阵
            if "ligand" in interp_dict and "pocket" in interp_dict:
                num_lig_atoms = len(interp_dict["ligand"]["node_indices"])
                num_prot_atoms = len(interp_dict["pocket"]["node_indices"])
                
                lig_node_map = {idx: i for i, idx in enumerate(interp_dict["ligand"]["node_indices"].cpu().numpy())}
                prot_node_map = {idx: i for i, idx in enumerate(interp_dict["pocket"]["node_indices"].cpu().numpy())}
                
                matrix = np.zeros((num_lig_atoms, num_prot_atoms), dtype=np.float32)
                for l_idx, p_idx, score in zip(lig_pair_nodes, prot_pair_nodes, pair_scores):
                    if l_idx in lig_node_map and p_idx in prot_node_map:
                        matrix[lig_node_map[l_idx], prot_node_map[p_idx]] = score
                        
                results["tracks"]["track_c_matrix"] = matrix

        return results


# ==========================================
# 导出与报告生成工具函数
# ==========================================
def save_interpretability_report(results: Dict[str, Any], output_dir: str, sample_id: str = "sample"):
    """将提取的可解释结果保存为 CSV 和 NumPy 矩阵"""
    os.makedirs(output_dir, exist_ok=True)
    
    pred_aff = float(np.squeeze(results['predicted_affinity']))
    print(f"=== 样本 {sample_id} 预测亲和力: {pred_aff:.4f} ===")
    
    tracks = results["tracks"]
    
    # 保存 Track A
    if "track_a_ligand" in tracks:
        df_a = tracks["track_a_ligand"]
        df_a.to_csv(os.path.join(output_dir, f"{sample_id}_track_a_ligand.csv"), index=False)
 
        
    # 保存 Track B
    if "track_b_pocket_atom" in tracks:
        df_b = tracks["track_b_pocket_atom"]
        df_b.to_csv(os.path.join(output_dir, f"{sample_id}_track_b_pocket_atom.csv"), index=False)
        if "track_b_pocket_residue" in tracks:
            tracks["track_b_pocket_residue"].to_csv(
                os.path.join(output_dir, f"{sample_id}_track_b_pocket_residue.csv"), index=False
            )
           

    # 保存 Track C
    if "track_c_pairs" in tracks:
        df_c = tracks["track_c_pairs"]
        df_c.to_csv(os.path.join(output_dir, f"{sample_id}_track_c_pairs.csv"), index=False)
        print(f"\n Track C (界面原子对 Top 3 强接触):")
        print(df_c.head(3).to_string(index=False))
        
    if "track_c_matrix" in tracks:
        np.save(os.path.join(output_dir, f"{sample_id}_track_c_heatmap_matrix.npy"), tracks["track_c_matrix"])
       