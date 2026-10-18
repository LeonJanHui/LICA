# /home/dldx/Boltz-EGNN-Affinity/src/interpretability/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class InterpretationVisualizer:
    """
    符合 JCIM / ACS 出版级视觉规范的高清全景图像渲染引擎 (内置全稳态空矩阵拦截盾)
    """
    def __init__(self):
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        sns.set_theme(style="ticks")
        
        # 莫兰迪高档科研色盘
        self.primary_color = "#2A629A"   # 深邃海蓝
        self.accent_color = "#C84B31"    # 砖红高亮

    def plot_figure_a_heatmap(self, res_df, save_path):
        """Figure A: 残基空间序列表现热图 (二维流形展开)"""
        # 🌟【刚性防御】：若遭遇零残基空账本，直接拦截，拒绝透传给 Seaborn，阻断 Numpy 崩溃
        if res_df is None or res_df.empty or len(res_df) == 0:
            print("⚠ [Visualization Warning] No active residues available. Skipping Figure A Heatmap.")
            return

        # 恢复按残基原位 ID 排序，展现序列景观
        df_sorted_id = res_df.sort_values(by='Residue_ID').reset_index(drop=True)
        
        scores_matrix = df_sorted_id['Attribution_Score'].values.reshape(1, -1)
        labels = [f"{name}\n{rid}" for name, rid in zip(df_sorted_id['Residue_Name'], df_sorted_id['Residue_ID'])]
        
        fig, ax = plt.subplots(figsize=(max(len(labels)*0.35, 8), 2.2), dpi=300)
        sns.heatmap(scores_matrix, annot=True, fmt=".3f", cmap="Blues", cbar=True,
                    xticklabels=labels, yticklabels=False, annot_kws={"size": 7, "weight": "bold"}, ax=ax)
        
        ax.set_title("Figure A: Spatial Attribution Heatmap Vector", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Residue Target Map", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

    def plot_figure_b_barplot(self, res_df, save_path, top_n=20):
        """Figure B: Top-20 核心热点残基贡献屏障图"""
        # 🌟【刚性防御】：空数据拦截
        if res_df is None or res_df.empty or len(res_df) == 0:
            print("⚠ [Visualization Warning] No active residues available. Skipping Figure B Barplot.")
            return

        top_df = res_df.head(top_n).copy()
        top_df['Label'] = [f"{name} ({rid})" for name, rid in zip(top_df['Residue_Name'], top_df['Residue_ID'])]
        
        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
        # 高亮前三名黄金残基 (如 ASP-189)
        colors = [self.accent_color if i < 3 else self.primary_color for i in range(len(top_df))]
        
        sns.barplot(x='Attribution_Score', y='Label', data=top_df, palette=colors, ax=ax, edgecolor='black', linewidth=0.5)
        
        # 注入原位数值刻度
        for i, v in enumerate(top_df['Attribution_Score']):
            ax.text(v + (v*0.02), i, f"{v:.4f}", va='center', fontsize=8, fontweight='bold', color='#333333')
            
        ax.set_title(f"Figure B: Top {top_n} Biophysical Hotspots", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Gradient × Attention Importance", fontsize=9, fontweight="bold")
        ax.set_ylabel("Target Residues", fontsize=9, fontweight="bold")
        sns.despine(ax=ax, trim=False)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

    def plot_figure_c_ranking(self, res_df, save_path):
        """Figure C: 全局残基能量贡献衰减与级联排序曲线"""
        # 🌟【刚性防御】：空数据拦截
        if res_df is None or res_df.empty or len(res_df) == 0:
            print("⚠ [Visualization Warning] No active residues available. Skipping Figure C Ranking.")
            return

        fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=300)
        
        ax.plot(res_df['Rank'].values, res_df['Attribution_Score'].values, 
                color=self.primary_color, linewidth=2.5, label="Attribution Trajectory")
        ax.fill_between(res_df['Rank'].values, res_df['Attribution_Score'].values, 
                        color=self.primary_color, alpha=0.15)
        
        # 捕捉拐点或高亮 Elite 集团
        highlight_count = min(3, len(res_df))
        ax.scatter(res_df['Rank'].values[:highlight_count], res_df['Attribution_Score'].values[:highlight_count], 
                   color=self.accent_color, s=40, zorder=5, edgecolors='black', label="Elite Hotspots")
        
        ax.set_title("Figure C: Global Residue Importance Ranking Curve", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Residue Rank (Decreasing)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Attribution Score Magnitude", fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3)
        sns.despine(ax=ax, trim=False)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()