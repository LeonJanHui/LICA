# /home/dldx/Boltz-EGNN-Affinity/src/plot_benchmark_comparison.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 📊 1. 结构化装填模型大盘真实对账数据
# ==========================================
models = ["LICA-EGNN", "GenScore", "Pafnucy", "GraphBAR"]

# Panel A: 亲和力相关系数大盘 (Higher is Better)
pearson_random = [0.8131, 0.7677, 0.7542, 0.7797]
pearson_purified = [0.8271, 0.7494, 0.7441, 0.7407]

# Panel B: 预测误差流形大盘 (Lower is Better)
rmse_random = [1.2880, 1.3412, 1.4593, 1.4711]
rmse_purified = [1.2763, 1.3916, 1.4789, 1.5168]

x = np.arange(len(models))
width = 0.35  # 分组并列柱状图的黄金宽度

# ==========================================
# 🎨 2. 初始化出版级高精双面板画布
# ==========================================
sns.set_theme(style="ticks")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=300)

# 定义 SCI 莫兰迪高冷对比色盘 (冷蓝代表常规随机划分，暖红代表防泄漏纯净划分)
COLOR_RANDOM = "#4A90E2"    # 随机划分：科技柔蓝
COLOR_PURIFIED = "#D0021B"  # 防泄漏划分：严谨深红

# ──────────────────────────────────────────
# 🚀 Panel A: Pearson R 性能泄露鸿沟对比 (Higher is Better)
# ──────────────────────────────────────────
rects1 = ax1.bar(x - width/2, pearson_random, width, label='Random Split', 
                 color=COLOR_RANDOM, edgecolor='black', linewidth=0.8, alpha=0.85)
rects2 = ax1.bar(x + width/2, pearson_purified, width, label='Tanimoto Split (Ours)', 
                 color=COLOR_PURIFIED, edgecolor='black', linewidth=0.8, alpha=0.9)

ax1.set_title("A  Linear Correlation Coefficient ($r$) Matrix", fontsize=13, fontweight='bold', pad=15, loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, fontweight='semibold')
ax1.set_ylabel("Pearson Correlation Coefficient", fontsize=11, fontweight='bold')
ax1.set_ylim(0.68, 0.96)  # 智能截断，无限放大“泄露降幅鸿沟”的视觉张力
ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# ──────────────────────────────────────────
# 🚀 Panel B: RMSE 预测误差回升对比 (Lower is Better)
# ──────────────────────────────────────────
rects3 = ax2.bar(x - width/2, rmse_random, width, label='Random Split', 
                 color=COLOR_RANDOM, edgecolor='black', linewidth=0.8, alpha=0.85)
rects4 = ax2.bar(x + width/2, rmse_purified, width, label='Tanimoto Split (Ours)', 
                 color=COLOR_PURIFIED, edgecolor='black', linewidth=0.8, alpha=0.9)

ax2.set_title("B  Absolute Predictive Error (RMSE) Profile", fontsize=13, fontweight='bold', pad=15, loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11, fontweight='semibold')
ax2.set_ylabel("Root-Mean-Square Error (RMSE)", fontsize=11, fontweight='bold')
ax2.set_ylim(0.80, 1.65)  # 智能截断，精准拉开误差飙升的对比
ax2.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# ==========================================
# 🏷️ 3. 自动化高密度精微数值标签注入
# ==========================================
def inject_labels(rects, ax, is_higher_better=True):
    for rect in rects:
        height = rect.get_height()
        # 根据指标是“越高越好”还是“越低越好”稍微调节数字悬浮的高度
        offset = 4 if is_higher_better else 4
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

inject_labels(rects1, ax1, is_higher_better=True)
inject_labels(rects2, ax1, is_higher_better=True)
inject_labels(rects3, ax2, is_higher_better=False)
inject_labels(rects4, ax2, is_higher_better=False)

# ==========================================
# 🧼 4. 极致去噪与 SCI 边框骨架精整
# ==========================================
for ax in [ax1, ax2]:
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    # 移除上方和右侧无用边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

# 🌟 破局点高亮：专门将你的模型（第0个标签）字体原位染红加粗，夺人眼球
for ax in [ax1, ax2]:
    labels = ax.get_xticklabels()
    labels[0].set_color("#D0021B")
    labels[0].set_fontweight("bold")
    labels[0].set_fontsize(12)

plt.tight_layout()

# 💾 一键导出最高印刷品质的图像资产
output_path = "/home/dldx/Boltz-EGNN-Affinity/results/LICA_EGNN_vs_Baselines_Leakage_Benchmark.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()

print(f"🎉 跨数据集划分的基准对比大图已完美输出至: {output_path}")