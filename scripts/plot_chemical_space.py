import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

warnings.filterwarnings("ignore")

# 🎨 1. LaTeX 1:1 纯正学术印刷规范配置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set_theme(style="ticks")

COLOR_TRAIN = "#36663E"         
COLOR_SCAFFOLD = "#FF9900"      
COLOR_RANDOM = "#FF6699"        


def main():
    SIM_DIR = "/home/dldx/Boltz-EGNN-Affinity/data/similarity"
    PSM_DIR = os.path.join(SIM_DIR, "pairwise_similarity_matrix")
    PSM_TANIMOTO = os.path.join(PSM_DIR, "pairwise_similarity_matrix_tanimoto.npy")
    PSM_COMPLEXES = os.path.join(PSM_DIR, "pairwise_similarity_complexes.json")
    ORIGINAL_JSON = os.path.join(SIM_DIR, "PDBbind_data_split_pdbbind.json")
    PURIFIED_JSON = os.path.join(SIM_DIR, "PDBbind_split_leakage_removed_c11_purified.json")
    OUTPUT_IMAGE = "../results/Chemical_Space_2x2_PrintReady.png"
    
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    
    mat_tanimoto = np.load(PSM_TANIMOTO)
    with open(PSM_COMPLEXES, 'r') as f: matrix_complexes = json.load(f)
    matrix_complexes = [str(c).strip().lower() for c in matrix_complexes]
    complex_to_idx = {comp: idx for idx, comp in enumerate(matrix_complexes)}
    
    with open(ORIGINAL_JSON, 'r') as f: orig_data = json.load(f)
    with open(PURIFIED_JSON, 'r') as f: pur_data = json.load(f)
        
    orig_train_idxs = [complex_to_idx[c] for c in orig_data.get('train', []) if str(c).strip().lower() in complex_to_idx]
    orig_test_idxs = [complex_to_idx[c] for c in orig_data.get('casf2016', []) if str(c).strip().lower() in complex_to_idx]
    pur_train_idxs = [complex_to_idx[c] for c in pur_data.get('train', []) if str(c).strip().lower() in complex_to_idx]
    pur_test_idxs = [complex_to_idx[c] for c in pur_data.get('casf2016_indep', []) if str(c).strip().lower() in complex_to_idx]
    
    pur_train_set = set([matrix_complexes[idx] for idx in pur_train_idxs])
    pur_test_set = set([matrix_complexes[idx] for idx in pur_test_idxs])

    max_original_sims = np.array([np.max(mat_tanimoto[t_idx, orig_train_idxs]) for t_idx in orig_test_idxs])
    max_purified_sims = np.array([np.max(mat_tanimoto[t_idx, pur_train_idxs]) for t_idx in pur_test_idxs])
    top10_original_sims = np.array([np.mean(np.sort(mat_tanimoto[t_idx, orig_train_idxs])[-10:]) for t_idx in orig_test_idxs])
    top10_purified_sims = np.array([np.mean(np.sort(mat_tanimoto[t_idx, pur_train_idxs])[-10:]) for t_idx in pur_test_idxs])
    
    thresholds = [0.5, 0.6, 0.7, 0.8]
    fraction_records = []
    for t in thresholds:
        fraction_records.append({'Threshold': f'>{t}', 'Split': 'Random Split', 'Fraction (%)': np.mean(max_original_sims > t) * 100.0})
        fraction_records.append({'Threshold': f'>{t}', 'Split': 'Tanimoto Split (Ours)', 'Fraction (%)': np.mean(max_purified_sims > t) * 100.0})
    df_fraction = pd.DataFrame(fraction_records)

    # 🌟 关键修改：按 LaTeX 真实版面尺寸 7.0 英寸定义画布，防止二次放缩失效！
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8), dpi=300)
    
    # --- Panel (a) ---
    dist_matrix = np.clip(1.0 - mat_tanimoto, 0.0, 1.0)
    reducer = umap.UMAP(metric='precomputed', n_neighbors=30, min_dist=0.1, random_state=42)
    umap_embedding = reducer.fit_transform(dist_matrix)
    
    umap_records = []
    for idx, comp in enumerate(matrix_complexes):
        if comp in pur_train_set: label = 'Train Set'
        elif comp in pur_test_set: label = 'CASF-2016 Core Set'
        else: continue
        umap_records.append({'UMAP_1': umap_embedding[idx, 0], 'UMAP_2': umap_embedding[idx, 1], 'Split': label})
    df_umap = pd.DataFrame(umap_records)
    
    ax_a = axes[0, 0]
    sns.scatterplot(data=df_umap[df_umap['Split'] == 'Train Set'], x='UMAP_1', y='UMAP_2', 
                    color=COLOR_TRAIN, alpha=0.35, s=6, label='Train Set', ax=ax_a, linewidth=0, zorder=1)
    sns.scatterplot(data=df_umap[df_umap['Split'] == 'CASF-2016 Core Set'], x='UMAP_1', y='UMAP_2', 
                    color=COLOR_SCAFFOLD, alpha=0.95, s=14, label='CASF-2016 Core Set', ax=ax_a, linewidth=0.2, edgecolor='white', zorder=2)
    
    ax_a.set_title("(a) UMAP Chemical Space", fontsize=9.5, fontweight='bold', pad=6)
    ax_a.set_xlabel("UMAP Dimension 1", fontsize=8.5); ax_a.set_ylabel("UMAP Dimension 2", fontsize=8.5)
    ax_a.tick_params(labelsize=7.5)
    ax_a.grid(True, linestyle="--", alpha=0.25); ax_a.legend(loc="upper right", frameon=True, fontsize=7)
    
    # --- Panel (b) ---
    ax_b = axes[0, 1]
    sns.ecdfplot(max_original_sims, color=COLOR_RANDOM, linewidth=1.6, label=f"Random Split (Mean = {max_original_sims.mean():.2f})", ax=ax_b)
    sns.ecdfplot(max_purified_sims, color=COLOR_SCAFFOLD, linewidth=1.6, label=f"Tanimoto Split (Mean = {max_purified_sims.mean():.2f})", ax=ax_b)
    ax_b.axvline(max_original_sims.mean(), color=COLOR_RANDOM, linestyle="--", linewidth=0.8, alpha=0.75)
    ax_b.axvline(max_purified_sims.mean(), color=COLOR_SCAFFOLD, linestyle="--", linewidth=0.8, alpha=0.75)
    
    ax_b.set_title("(b) ECDF of Top-1 Similarity", fontsize=9.5, fontweight='bold', pad=6)
    ax_b.set_xlabel("Max Tanimoto Coefficient", fontsize=8.5); ax_b.set_ylabel("Empirical CDF", fontsize=8.5)
    ax_b.set_xlim(0.0, 1.0); ax_b.tick_params(labelsize=7.5)
    ax_b.grid(True, linestyle="--", alpha=0.25); ax_b.legend(loc="lower right", fontsize=7, frameon=True)
    
    # --- Panel (c) ---
    ax_c = axes[1, 0]
    sns.kdeplot(top10_original_sims, color=COLOR_RANDOM, linewidth=1.5, fill=True, alpha=0.15, 
                label=f"Random Split (Mean = {top10_original_sims.mean():.2f})", ax=ax_c, clip=(0.0, 1.0))
    sns.kdeplot(top10_purified_sims, color=COLOR_SCAFFOLD, linewidth=1.5, fill=True, alpha=0.15, 
                label=f"Tanimoto Split (Mean = {top10_purified_sims.mean():.2f})", ax=ax_c, clip=(0.0, 1.0))
    ax_c.axvline(top10_original_sims.mean(), color=COLOR_RANDOM, linestyle="--", linewidth=0.8, alpha=0.75)
    ax_c.axvline(top10_purified_sims.mean(), color=COLOR_SCAFFOLD, linestyle="--", linewidth=0.8, alpha=0.75)
    
    ax_c.set_title("(c) Top-10 Cluster Density", fontsize=9.5, fontweight='bold', pad=6)
    ax_c.set_xlabel("Average Top-10 Tanimoto", fontsize=8.5); ax_c.set_ylabel("Density", fontsize=8.5)
    ax_c.set_xlim(0.0, 1.0); ax_c.tick_params(labelsize=7.5)
    ax_c.grid(True, linestyle="--", alpha=0.25); ax_c.legend(loc="upper right", fontsize=7, frameon=True)
    
    # --- Panel (d) ---
    ax_d = axes[1, 1]
    palette_dict = {'Random Split': COLOR_RANDOM, 'Tanimoto Split (Ours)': COLOR_SCAFFOLD}
    sns.barplot(data=df_fraction, x='Threshold', y='Fraction (%)', hue='Split', palette=palette_dict, 
                ax=ax_d, edgecolor='#1C2833', linewidth=0.5, alpha=0.9)
    
    for p in ax_d.patches:
        val = p.get_height()
        if val > 0:
            ax_d.annotate(f"{val:.1f}%", (p.get_x() + p.get_width() / 2., val + 1.2), 
                        ha='center', va='center', fontsize=7, color='#1C2833', fontweight='semibold')
            
    ax_d.set_title("(d) High-Similarity Fraction", fontsize=9.5, fontweight='bold', pad=6)
    ax_d.set_xlabel("Tanimoto Boundary", fontsize=8.5); ax_d.set_ylabel("Test Ligands Fraction (%)", fontsize=8.5)
    ax_d.set_ylim(0.0, max(df_fraction['Fraction (%)']) + 14.0); ax_d.tick_params(labelsize=7.5)
    ax_d.grid(True, axis='y', linestyle="--", alpha=0.25); ax_d.legend(loc="upper right", fontsize=6.8, frameon=True)
    
    sns.despine(trim=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, bbox_inches='tight', dpi=300)
    print(f"\n✅ [OPTIMIZED 2x2 PRINT-READY EXPORT COMPLETE]：{OUTPUT_IMAGE}")

if __name__ == '__main__': main()