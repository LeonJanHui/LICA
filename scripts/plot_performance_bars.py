# /home/dldx/Boltz-EGNN-Affinity/src/plot_metrics.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 🎨 SCI 印刷级样式配置 (严丝合缝对齐 JCIM / ACS 视觉规范)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_theme(style="ticks")

# 统一前后面板的学术高级配色 (净化尾部空格，高光高亮我们的 LICA-EGNN)
COLORS = {
    "GraphBar": "#D3D3D3",        # 浅灰色 (Baseline)
    "Pafnucy": "#A2B5CD",         # 钢蓝色 (经典 CNN)
    "SIGN": "#7F7F7F",            # 深灰色 (传统几何图)
    "Genscore": "#4682B4",        # 靛蓝色 (经典打分函数)
    "LICA-EGNN (Ours)": "#FF6A00", # 醒目的活力橙 (我们的完全体模型)
}


def main():
    # 1. 装载全新真实复现的全量盲测大考数据集
    data = {
        "Model": [
            "GraphBar",
            "Pafnucy",
            "SIGN",
            "Genscore",
            "LICA-EGNN (Ours)",
        ],
        "RMSE": [1.5168, 1.4789, 2.1722, 1.3916, 1.2568],
        "SD": [1.4647, 1.4740, 2.1184, 1.3945, 1.1827],
        "Pearson $R$": [0.7407, 0.7441, 0.2294, 0.7494, 0.8397],
        "Spearman $\\rho$": [0.7211, 0.7434, 0.2470, 0.7410, 0.8344],
    }
    df = pd.DataFrame(data)

    # 2. 初始化 2x2 的多面板高分辨率画布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    metrics = ["RMSE", "SD", "Pearson $R$", "Spearman $\\rho$"]
    sub_labels = ["(a)", "(b)", "(c)", "(d)"]

    # 3. 循环洗炼绘制每一个子面板
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 区分性能指标的演化收敛方向
        is_error = metric in ["RMSE", "SD"]
        direction_str = "($\\downarrow$ Lower is better)" if is_error else "($\\uparrow$ Higher is better)"

        # 绘制核心条形图
        bars = ax.bar(
            df["Model"],
            df[metric],
            color=[COLORS[m] for m in df["Model"]],
            edgecolor="black",
            linewidth=0.7,
            width=0.55,
            zorder=2,
        )

        # 🌟 核心增值设计：原位在柱状图顶部打上高精度字标，并对我们的模型实施加粗高亮
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            current_model = df["Model"].iloc[idx]
            is_ours = current_model == "LICA-EGNN (Ours)"
            
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),  # 向上垂直偏移 4 个点
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold" if is_ours else "normal",
                color="#D32F2F" if (is_ours and is_error) else ("#1B5E20" if is_ours else "black") 
                # 我们的模型：误差用红字，相关系数用绿字，极具学术视觉冲击力
            )

        # 美化轴标签与分面子标题
        ax.set_title(
            f"{sub_labels[i]} {metric} {direction_str}",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        ax.set_ylabel(metric, fontsize=10, fontweight="bold")

        # 针对不同维度的指标动态微调 Y 轴视窗边界，优雅留出顶部标签空间
        max_val = df[metric].max()
        ax.set_ylim(0, max_val * 1.18)

        # 轴线美化与饰面
        ax.set_xticklabels(
            df["Model"], rotation=15, ha="right", fontsize=9.5, fontweight="normal"
        )
        ax.grid(True, linestyle="--", alpha=0.3, axis="y", zorder=1)
        sns.despine(ax=ax, trim=False)

    # 4. 全局紧凑整饰并高清落盘
    plt.tight_layout()
    output_dir = "/home/dldx/Boltz-EGNN-Affinity/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Performance_Metrics_Bars.png")
    
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"📊 [SUCCESS] SCI-grade 2x2 multi-panel bar plot saved to: {output_path}")


if __name__ == "__main__":
    main()