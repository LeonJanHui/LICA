import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_ligand_residue_heatmap(
    sample_dir: str, 
    sample_id: str, 
    top_k_residues: int = 15,
    output_path: str = None
):
    """
    根据提取的 Track C Pairs 接触数据，自动生成 配体原子 vs. 蛋白质残基 交叉结合热力图
    """
    pairs_csv = os.path.join(sample_dir, f"{sample_id}_track_c_pairs.csv")
    pocket_csv = os.path.join(sample_dir, f"{sample_id}_track_b_pocket_atom.csv")
    
    if not os.path.exists(pairs_csv):
        raise FileNotFoundError(f"未找到界面 Pair 数据: {pairs_csv}")
        
    df_pairs = pd.read_csv(pairs_csv)
    
    # 1. 尝试关联口袋残基名称 (如果存在 track_b_pocket_atom.csv 且包含 residue_id)
    if os.path.exists(pocket_csv):
        df_pocket = pd.read_csv(pocket_csv)
        if "residue_id" in df_pocket.columns and "global_node_idx" in df_pocket.columns:
            node2res = dict(zip(df_pocket["global_node_idx"], df_pocket["residue_id"]))
            df_pairs["residue_id"] = df_pairs["protein_node_idx"].map(node2res)
        else:
            df_pairs["residue_id"] = df_pairs["protein_node_idx"].apply(lambda x: f"Prot_Atom_{x}")
    else:
        df_pairs["residue_id"] = df_pairs["protein_node_idx"].apply(lambda x: f"Prot_Atom_{x}")

    # 过滤无效残基空值
    df_pairs = df_pairs.dropna(subset=["residue_id"])

    # 2. 筛选出累积/最高注意力得分最高的 Top K 个结合残基
    top_res_list = (
        df_pairs.groupby("residue_id")["attention_score"]
        .max()
        .nlargest(top_k_residues)
        .index.tolist()
    )
    
    df_filtered = df_pairs[df_pairs["residue_id"].isin(top_res_list)]

    # 3. 构建 2D 透视表矩阵 (Ligand Atom Index vs. Residue ID)
    # 填充规则取 max (同一个残基内多个原子与配体原子的最大接触得分)
    pivot_matrix = df_filtered.pivot_table(
        index="ligand_node_idx", 
        columns="residue_id", 
        values="attention_score", 
        aggfunc="max",
        fill_value=0.0
    )

    # 重新按 Top-K 残基顺序排列列名，提升美观度
    ordered_cols = [r for r in top_res_list if r in pivot_matrix.columns]
    pivot_matrix = pivot_matrix[ordered_cols]
    
    # 格式化 Y 轴标签 (例如 Ligand_Atom_0, Ligand_Atom_1)
    pivot_matrix.index = [f"Ligand_Atom_{idx}" for idx in pivot_matrix.index]

    # 4. 绘制热力图
    sns.set_theme(style="white", font="sans-serif")
    plt.figure(figsize=(11, 7), dpi=300)

    ax = sns.heatmap(
        pivot_matrix,
        cmap="YlOrRd",
        annot=False,
        cbar_kws={'label': 'Interface Pair Attention Score'},
        linewidths=0.5,
        linecolor='white'
    )

    plt.title(f"Sample [{sample_id}] Ligand Atom vs. Pocket Residue Binding Heatmap", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Protein Pocket Residue", fontsize=11, fontweight='bold')
    plt.ylabel("Ligand Atom Index", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(sample_dir, f"{sample_id}_ligand_residue_heatmap.png")
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 配体-残基结合交叉热力图已成功保存至: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制配体与蛋白残基结合热力图")
    parser.add_argument("--sample_dir", type=str, required=True, help="单样本结果存放目录, 如 ./xai_results/samples/3u8n")
    parser.add_argument("--sample_id", type=str, required=True, help="样本 ID, 如 3u8n")
    parser.add_argument("--top_k", type=int, default=15, help="展示的 Top 口袋残基数量")
    args = parser.parse_args()

    generate_ligand_residue_heatmap(args.sample_dir, args.sample_id, top_k_residues=args.top_k)