# /home/dldx/Boltz-EGNN-Affinity/src/plot_performance_scatter.py
import os
import sys
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 🎨 SCI 印刷级样式配置 (严丝合缝对齐 JCIM / ACS 视觉与出版规范)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_theme(style="ticks")

# ──────────────────────────────────────────────────────────────────────────
# 🎨 核心色彩控制台
# ──────────────────────────────────────────────────────────────────────────
COLOR_IDENTITY_LINE = "#888888"   # 1:1 理想参考线：内敛暗灰
COLOR_SCATTER_POINTS = "#2A629A"  # 复合物真实散点：高级深邃海蓝
COLOR_FITTING_LINE = "#C84B31"    # 最佳回归拟合线：典雅复古砖红
# ──────────────────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred):
    """严格物理指标计算核心 (集成 CASF 官方一元线性回归标准残差 SD)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 计算一元线性回归斜率与截距，进而解算标准残差 (SD)
    n = len(y_true)
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    y_fitted = slope * y_pred + intercept
    sd_val = np.sqrt(np.sum((y_true - y_fitted) ** 2) / (n - 2)) if n > 2 else 0.0
    
    r_pearson, _ = pearsonr(y_true, y_pred)
    r_spearman, _ = spearmanr(y_true, y_pred)
    
    return rmse, mae, sd_val, r_pearson, r_spearman, slope, intercept


def plot_fitting_scatter(y_true, y_pred, output_path):
    """绘制符合 JCIM 顶刊水准的高精散点拟合图"""
    rmse, mae, sd_val, r_p, r_s, slope, intercept = compute_metrics(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=300)
    
    # 🌟【本次原位修改点】：在图片顶部正中加冕学术标题 "Random Split"
    # 使用 pad=15 撑开安全空间，防止与内部指标看板和边缘发生空间踩踏
    ax.set_title("Random Split", fontsize=14, fontweight="bold", pad=15, loc="center")
    
    # 1. 绘制 1:1 理想对角参考线 (Identity Line)
    min_bound = min(y_true.min(), y_pred.min()) - 0.5
    max_bound = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([min_bound, max_bound], [min_bound, max_bound], 
            linestyle="--", color=COLOR_IDENTITY_LINE, linewidth=1.2, label="Ideal (y = x)", zorder=1)
    
    # 2. 绘制真实的线性校准回归线 (Linear Regression Fitting Line)
    x_fit = np.linspace(min_bound, max_bound, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color=COLOR_FITTING_LINE, linewidth=2.2, 
            label=f"Fitting (y = {slope:.2f}x + {intercept:.2f})", zorder=3)
    
    # 3. 绘制原子级亲和力复合物真实散点 (高级海蓝色)
    ax.scatter(y_pred, y_true, color=COLOR_SCATTER_POINTS, edgecolors="black", 
               linewidths=0.5, s=38, alpha=0.65, label="LICA-EGNN (CASF-2016)", zorder=2)
    
    ax.set_xlabel("Predicted Binding Affinity ($p K_d / p K_i$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Experimental Binding Affinity ($p K_d / p K_i$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlim(min_bound, max_bound)
    ax.set_ylim(min_bound, max_bound)
    
    # 4. 原位在左上角挂载高级半透明高精指标看板
    text_str = (
        f"$\\mathbf{{Pearson\\,R:}}$ {r_p:.4f}\n"
        f"$\\mathbf{{Spearman\\,\\rho:}}$ {r_s:.4f}\n"
        f"$\\mathbf{{RMSE:}}$ {rmse:.4f}\n"
        f"$\\mathbf{{MAE:}}$ {mae:.4f}\n"
        f"$\\mathbf{{SD:}}$ {sd_val:.4f}"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#D3D3D3", alpha=0.85)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props, style="normal")
    
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    sns.despine(ax=ax, trim=False)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"📈 [SUCCESS] LICA-EGNN performance scatter plot with 'Random Split' title saved to: {output_path}")


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # 📌 绝对物理路径自填区
    # ──────────────────────────────────────────────────────────────────────────
    ROOT_DIR = "/home/dldx/Boltz-EGNN-Affinity/data/raw/CASF-2016/coreset"          # CASF 晶体文件夹物理父目录
    LEDGER_JSON = "/home/dldx/Boltz-EGNN-Affinity/data/random/random.json"          # 划分账本 JSON 绝对路径
    DATA_DICT_JSON = "/home/dldx/Boltz-EGNN-Affinity/data/random/PDBbind_data_dict.json"  # 亲和力真实标签字典绝对路径
    SAVE_DIR = "/home/dldx/Boltz-EGNN-Affinity/data/processed/pyg_graphs_random"      # PyG 图数据的落盘/读取目录
    
    # 🎯 靶向锁定 Full Model 训练出的单核最优权重文件
    MODEL_WEIGHTS_PATH = "/home/dldx/Boltz-EGNN-Affinity/checkpoints/random/best_model.pt"
    OUTPUT_IMAGE_PATH = "/home/dldx/Boltz-EGNN-Affinity/results/LICA_EGNN_Full_Performance_Scatter.png"
    
    # 🧬 满血 Full Model 张量维度静态硬编码 (已防御 Unexpected key 报错)
    MODEL_CONFIG = {
        "node_in_dim": 32,
        "rbf_dim": 16,
        "num_edge_types": 3,   # 锁死为 3，完美匹配你的高性能最优权重文件
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout_rate": 0.15,
        "coord_update": True   # 激活 4 层等变空间诱导契合更新核心驱动
    }
    # ──────────────────────────────────────────────────────────────────────────

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[STATUS] Hardware Engine locked on: [{device}] (Premium Plotting Mode)")
    print(f"[STATUS] Target Weight Path: {MODEL_WEIGHTS_PATH}")
    
    from dataset import ComplexGraphDataset 
    from model.network import PureGeometricAffinityNetwork

    # 1. 载入独立测试集
    print("⏳ Loading CASF-2016 independent core test dataset...")
    test_dataset = ComplexGraphDataset(
        root_dir=ROOT_DIR, ledger_json=LEDGER_JSON, data_dict_json=DATA_DICT_JSON, 
        save_dir=SAVE_DIR, split_name='casf2016_indep', edge_cutoff=6.0, 
        label_col='log_kd_ki', ligand_ext='.sdf'
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_label_key = test_dataset.label_col
    
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ [CRITICAL ERROR] Target weights not found at: {MODEL_WEIGHTS_PATH}! Execution aborted.")
        return
        
    # 2. 满配实例化 LICA-EGNN 核心拓扑网络
    print(f"🔒 Instantiating Full-Config LICA-EGNN Paradigm...")
    model = PureGeometricAffinityNetwork(
        node_in_dim=MODEL_CONFIG["node_in_dim"], 
        rbf_dim=MODEL_CONFIG["rbf_dim"],
        num_edge_types=MODEL_CONFIG["num_edge_types"], 
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_layers=MODEL_CONFIG["num_layers"], 
        dropout_rate=MODEL_CONFIG["dropout_rate"],
        coord_update=MODEL_CONFIG["coord_update"]
    ).to(device)
    
    # 压入权重
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=False))
    model.eval()
    
    test_preds = []
    true_labels = []
    
    # 3. 推理数据流
    print("🚀 Running inference workflow on CASF-2016 test stream...")
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            test_preds.append(output.cpu().numpy())
            true_labels.append(batch_data[test_label_key].cpu().numpy())
            
    final_preds = np.concatenate(test_preds)
    final_trues = np.concatenate(true_labels)
    
    # 4. 生成图纸
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    plot_fitting_scatter(final_trues, final_preds, OUTPUT_IMAGE_PATH)


if __name__ == "__main__":
    main()