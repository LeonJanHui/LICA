import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def plot_xai_dataset_summary(csv_path: str, output_image_path: str = "xai_dataset_summary_visualization.png"):
    """
    绘制数据集级 4 合 1 统计可视化图表
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到汇总文件: {csv_path}")

    df = pd.read_csv(csv_path)

    # 设置画图风格（针对 SCI 论文优化的配色与字体）
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)

    # ----------------------------------------------------
    # (A) True vs. Predicted Affinity (亲和力回归分析)
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    sns.regplot(
        data=df, x='true_affinity', y='pred_affinity', ax=ax1,
        scatter_kws={'alpha': 0.65, 'color': '#2b5c8f', 's': 35},
        line_kws={'color': '#e74c3c', 'linewidth': 2, 'label': 'Regression Line'}
    )
    
    # 1:1 对角线
    min_val = min(df['true_affinity'].min(), df['pred_affinity'].min()) - 0.5
    max_val = max(df['true_affinity'].max(), df['pred_affinity'].max()) + 0.5
    ax1.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.7, label='Ideal (1:1)')

    r_p, _ = pearsonr(df['true_affinity'], df['pred_affinity'])
    r_s, _ = spearmanr(df['true_affinity'], df['pred_affinity'])
    rmse = np.sqrt(((df['true_affinity'] - df['pred_affinity']) ** 2).mean())

    ax1.text(
        0.05, 0.80,
        f'Pearson R = {r_p:.3f}\nSpearman R = {r_s:.3f}\nRMSE = {rmse:.3f}',
        transform=ax1.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85),
        fontsize=10, fontweight='bold'
    )
    ax1.set_title('(A) True vs. Predicted Affinity', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('True Affinity (pK_a/pK_i)', fontsize=11)
    ax1.set_ylabel('Predicted Affinity (pK_a/pK_i)', fontsize=11)
    ax1.legend(loc='lower right', frameon=True)

    # ----------------------------------------------------
    # (B) Top-1 Pair Distance Distribution (强注意力接触的物理距离分布)
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    sns.histplot(df['top_pair_distance_A'], kde=True, ax=ax2, color='#27ae60', bins=20)
    mean_dist = df['top_pair_distance_A'].mean()
    ax2.axvline(
        mean_dist, color='#c0392b', linestyle='--', linewidth=2,
        label=f'Mean Distance = {mean_dist:.2f} Å'
    )
    ax2.set_title('(B) Top-1 Pair Distance Distribution (Å)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Spatial Distance (Å)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(loc='upper right', frameon=True)

    # ----------------------------------------------------
    # (C) Top-1 Attention Score vs. Distance (注意力与距离关联系数)
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    sns.scatterplot(
        data=df, x='top_pair_distance_A', y='top_pair_attn_score',
        hue='num_interface_pairs', palette='viridis', size='num_ligand_atoms',
        sizes=(20, 100), alpha=0.8, ax=ax3
    )
    ax3.set_title('(C) Top-1 Attention Score vs. Distance', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Top-1 Pair Distance (Å)', fontsize=11)
    ax3.set_ylabel('Attention Score', fontsize=11)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Scale/Pairs", fontsize=8)

    # ----------------------------------------------------
    # (D) Ligand Atoms vs. Interface Pairs (分子规模与界面连接扩展)
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    sns.scatterplot(
        data=df, x='num_ligand_atoms', y='num_interface_pairs',
        hue='num_pocket_atoms', palette='magma', alpha=0.8, ax=ax4, s=40
    )
    ax4.set_title('(D) Ligand Atoms vs. Interface Pair Connections', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('Number of Ligand Atoms', fontsize=11)
    ax4.set_ylabel('Number of Interface Pairs', fontsize=11)
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Pocket Atoms", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"✅ 宏观统计图表已成功保存至: {os.path.abspath(output_image_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./xai_results/dataset_xai_summary.csv")
    parser.add_argument("--output", type=str, default="xai_dataset_summary_visualization.png")
    args = parser.parse_args()
    plot_xai_dataset_summary(args.csv_path, args.output)