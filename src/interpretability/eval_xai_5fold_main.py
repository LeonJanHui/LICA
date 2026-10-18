from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from torch_geometric.loader import DataLoader
import yaml
from tqdm import tqdm

# 1. 动态获取 src 目录路径并加入系统搜索路径，消除 Import 错误
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/interpretability
src_dir = os.path.abspath(os.path.join(current_dir, "..")) # src
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dataset import ComplexGraphDataset
from extract_attention import NativeAttentionExtractor, save_interpretability_report
from features import EDGE_STATIC_DIM, NODE_FEATURE_DIM, NUM_EDGE_TYPES
from model.network import PureGeometricAffinityNetwork


def load_config(path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_5fold_ensemble_evaluation(
    model_list: List[torch.nn.Module],
    test_loader: DataLoader,
    root_dir: str,
    output_dir: str = "./xai_5fold_ensemble_results",
    device: str = "cuda",
    save_individual_files: bool = True,
    use_raw_scores: bool = False
):
    """
    遍历整个数据集，5 个模型并行预测并取 Affinity 和 Track A/B/C 注意力得分的平均值 (Ensemble)
    """
    os.makedirs(output_dir, exist_ok=True)
    extractors = [NativeAttentionExtractor(m, device=device, root_dir=root_dir) for m in model_list]
    
    all_pred_affinities = []
    all_true_affinities = []
    summary_records = []

    print(f"🚀 开始执行 5-Fold 集成可解释性评估 (共 {len(model_list)} 个模型，评估 {len(test_loader)} 个样本)...")
    
    for idx, batch in enumerate(tqdm(test_loader, desc="5-Fold Ensemble Evaluation")):
        # 1. 获取样本标识符，防止 list/tensor 引发类型错误
        if hasattr(batch, "complex_id"):
            cid = batch.complex_id
            sample_id = str(cid[0] if isinstance(cid, (list, tuple)) else cid)
        elif hasattr(batch, "pdb_id"):
            pid = batch.pdb_id
            sample_id = str(pid[0] if isinstance(pid, (list, tuple)) else pid)
        else:
            sample_id = f"complex_{idx:04d}"

        # 2. 依次用 5 个模型提取预测值与注意力
        fold_affinities = []
        fold_results = []
        
        for ext in extractors:
            res = ext.extract_from_data(batch, use_raw_scores=use_raw_scores, root_dir=root_dir)
            fold_affinities.append(res["predicted_affinity"])
            fold_results.append(res)
            
        # 3. 对 5 个模型的亲和力预测值取均值 (Ensemble Prediction)
        ensemble_affinity = float(np.mean(fold_affinities))
        
        # 4. 对 5 个模型的 Track A/B/C 注意力得分进行跨模型聚合平均
        base_tracks = fold_results[0]["tracks"]
        
        # --- 集成 Track A (配体原子) ---
        if "track_a_ligand" in base_tracks:
            df_a = base_tracks["track_a_ligand"].copy()
            scores_a = [
                res["tracks"]["track_a_ligand"]["attention_score"].values 
                for res in fold_results if "track_a_ligand" in res["tracks"]
            ]
            if len(scores_a) == len(model_list):
                df_a["attention_score"] = np.mean(scores_a, axis=0)
                df_a = df_a.sort_values(by="attention_score", ascending=False)
                base_tracks["track_a_ligand"] = df_a

        # --- 集成 Track B (口袋残基) ---
        if "track_b_pocket_atom" in base_tracks:
            df_b_atom = base_tracks["track_b_pocket_atom"].copy()
            scores_b_atom = [
                res["tracks"]["track_b_pocket_atom"]["attention_score"].values 
                for res in fold_results if "track_b_pocket_atom" in res["tracks"]
            ]
            if len(scores_b_atom) == len(model_list):
                df_b_atom["attention_score"] = np.mean(scores_b_atom, axis=0)
                
                # 重新按平均注意力得分聚合残基 Hotspot
                if "residue_id" in df_b_atom.columns:
                    df_b_res = (
                        df_b_atom.groupby("residue_id")["attention_score"]
                        .max()
                        .reset_index()
                        .sort_values(by="attention_score", ascending=False)
                    )
                    base_tracks["track_b_pocket_residue"] = df_b_res
                    
                df_b_atom = df_b_atom.sort_values(by="attention_score", ascending=False)
                base_tracks["track_b_pocket_atom"] = df_b_atom

        # --- 集成 Track C (界面 Pair 与 2D 热力图矩阵) ---
        if "track_c_pairs" in base_tracks:
            df_c = base_tracks["track_c_pairs"].copy()
            scores_c = [
                res["tracks"]["track_c_pairs"]["attention_score"].values 
                for res in fold_results if "track_c_pairs" in res["tracks"]
            ]
            if len(scores_c) == len(model_list):
                df_c["attention_score"] = np.mean(scores_c, axis=0)
                df_c = df_c.sort_values(by="attention_score", ascending=False)
                base_tracks["track_c_pairs"] = df_c

        if "track_c_matrix" in base_tracks:
            matrices = [
                res["tracks"]["track_c_matrix"] 
                for res in fold_results if "track_c_matrix" in res["tracks"]
            ]
            if len(matrices) == len(model_list):
                base_tracks["track_c_matrix"] = np.mean(matrices, axis=0)

        # 组合集成后的单样本结果字典
        ensemble_results = {
            "predicted_affinity": ensemble_affinity,
            "tracks": base_tracks
        }

        # 5. 收集真实标签 (如果存在)
        true_affinity = None
        if hasattr(batch, "y") and batch.y is not None:
            true_affinity = float(batch.y.item() if batch.y.ndim == 0 else batch.y.cpu().numpy()[0])
            all_pred_affinities.append(ensemble_affinity)
            all_true_affinities.append(true_affinity)

        # 6. 提取统计指标保存至 Summary 汇总表
        num_lig_atoms = len(base_tracks.get("track_a_ligand", []))
        num_pocket_atoms = len(base_tracks.get("track_b_pocket_atom", []))
        num_pairs = len(base_tracks.get("track_c_pairs", []))
        
        top_hotspot_res = "N/A"
        if "track_b_pocket_residue" in base_tracks and not base_tracks["track_b_pocket_residue"].empty:
            top_hotspot_res = base_tracks["track_b_pocket_residue"].iloc[0]["residue_id"]

        top_pair_dist = np.nan
        top_pair_score = np.nan
        if "track_c_pairs" in base_tracks and not base_tracks["track_c_pairs"].empty:
            top_pair_dist = base_tracks["track_c_pairs"].iloc[0]["distance_A"]
            top_pair_score = base_tracks["track_c_pairs"].iloc[0]["attention_score"]

        summary_records.append({
            "sample_id": sample_id,
            "true_affinity": true_affinity,
            "pred_affinity": ensemble_affinity,
            "num_ligand_atoms": num_lig_atoms,
            "num_pocket_atoms": num_pocket_atoms,
            "num_interface_pairs": num_pairs,
            "top_hotspot_residue": top_hotspot_res,
            "top_pair_distance_A": top_pair_dist,
            "top_pair_attn_score": top_pair_score
        })

        # 7. 导出集成后的单样本 CSV / NPY 文件
        if save_individual_files:
            sample_out_dir = os.path.join(output_dir, "samples", sample_id)
            save_interpretability_report(ensemble_results, sample_out_dir, sample_id=sample_id)

    # 8. 生成全局数据集分析报告与评估指标
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(output_dir, "dataset_xai_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print("\n" + "="*60)
    print("📊 五折集成可解释性测评统计报告")
    print("="*60)
    print(f"✅ 已成功评估样本数: {len(summary_df)}")
    print(f"✅ 平均界面 Pair 接触数量: {summary_df['num_interface_pairs'].mean():.1f}")
    print(f"✅ 集成后 Top-1 强注意力 Pair 的平均空间距离: {summary_df['top_pair_distance_A'].mean():.2f} Å")

    if len(all_true_affinities) > 0:
        y_true = np.array(all_true_affinities)
        y_pred = np.array(all_pred_affinities)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        p_corr, _ = pearsonr(y_true, y_pred)
        s_corr, _ = spearmanr(y_true, y_pred)
        
        print(f"\n📈 五折集成后的亲和力预测回归性能:")
        print(f"   - Ensemble RMSE:        {rmse:.4f}")
        print(f"   - Ensemble Pearson R:   {p_corr:.4f}")
        print(f"   - Ensemble Spearman R:  {s_corr:.4f}")

    print(f"\n📁 所有结果与汇总文件已保存至: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(description="PureGeometricAffinityNetwork 五折集成可解释测评主程序")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径 (如 train_config.yaml)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="存放 5 个 best_model_fold_x.pt 权重的文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./xai_5fold_ensemble_results", help="结果导出目录")
    parser.add_argument("--use_raw_scores", action="store_true", help="是否使用未经过 Softmax 的 raw_scores")
    parser.add_argument("--no_individual_files", action="store_true", help="只输出全局汇总表，不导出单样本 CSV/NPY")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 自动定位 5 个 Fold 的 Checkpoint 路径
    ckpt_paths = [
        os.path.join(args.ckpt_dir, f"best_model_fold_{k}.pt") for k in range(1, 6)
    ]
    
    models = []
    m, d, a = cfg["model"], cfg["dataset"], cfg["ablation"]

    for path in ckpt_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 Checkpoint 文件: {path}")
            
        print(f"正在加载 Fold 权重: {path}")
        model = PureGeometricAffinityNetwork(
            node_in_dim=NODE_FEATURE_DIM,
            edge_static_dim=EDGE_STATIC_DIM,
            num_edge_types=NUM_EDGE_TYPES,
            hidden_dim=int(m["hidden_dim"]),
            num_layers=int(m["num_layers"]),
            rbf_dim=int(m["rbf_dim"]),
            interface_cutoff=float(d["interface_cutoff"]),
            protein_spatial_cutoff=float(d["protein_spatial_cutoff"]),
            dropout_rate=float(m["dropout_rate"]),
            coord_update=bool(a.get("coord_update", False)),
            max_coord_step=float(m.get("max_coord_step", 0.1)),
            use_track_a=bool(a.get("use_track_a", True)),
            use_track_b=bool(a.get("use_track_b", True)),
            use_track_c=bool(a.get("use_track_c", True)),
            param_match=bool(a.get("param_match", False)),
            param_match_track=a.get("param_match_track", None),
        ).to(device)
        
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models.append(model)

    # 2. 构建测试集 DataLoader (必须 batch_size=1)
    p, d = cfg["paths"], cfg["dataset"]
    test_dataset = ComplexGraphDataset(
        root_dir=p["root_dir"],
        ledger_json=p["ledger_json"],
        data_dict_json=p["data_dict_json"],
        save_dir=p["save_dir"],
        split_name=d["test_split"],
        interface_cutoff=float(d["interface_cutoff"]),
        protein_spatial_cutoff=float(d["protein_spatial_cutoff"]),
        label_col=d["label_col"],
        include_hydrogens=bool(d.get("include_hydrogens", False)),
        water_policy=d.get("water_policy", "remove"),
        water_cutoff=float(d.get("water_cutoff", 4.0)),
        strict_labels=bool(d.get("strict_labels", True)),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. 执行五折集成测评
    run_5fold_ensemble_evaluation(
        model_list=models,
        test_loader=test_loader,
        root_dir=p["root_dir"],
        output_dir=args.output_dir,
        device=device,
        save_individual_files=not args.no_individual_files,
        use_raw_scores=args.use_raw_scores
    )

if __name__ == "__main__":
    main()