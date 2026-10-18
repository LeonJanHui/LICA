# /home/dldx/Boltz-EGNN-Affinity/src/preprocess/cluster_based_split.py
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

warnings.filterwarnings("ignore")

# ========================================================================
# 1. Top‑journal‑level rigid hyperparameter definition
# ========================================================================
CLUSTER_THRESHOLD = 0.75   # Rigid connected‑component similarity threshold (0.7‑0.8 is strict standard for top journals)
RUN_ID = "c11"

SIM_DIR = "/home/dldx/Boltz-EGNN-Affinity/data/similarity"
INPUT_SPLIT_FILE = os.path.join(SIM_DIR, "PDBbind_data_split_pdbbind.json")
OUTPUT_SPLIT_FILE = os.path.join(SIM_DIR, f"PDBbind_split_leakage_removed_{RUN_ID}_purified.json")

PSM_DIR = os.path.join(SIM_DIR, "pairwise_similarity_matrix")
PSM_TANIMOTO = os.path.join(PSM_DIR, "pairwise_similarity_matrix_tanimoto.npy")
PSM_COMPLEXES = os.path.join(PSM_DIR, "pairwise_similarity_complexes.json")
DATA_DICT_PATH = os.path.join(SIM_DIR, "PDBbind_data_dict.json")

def main():
    if not os.path.exists(INPUT_SPLIT_FILE):
        raise FileNotFoundError(f"CRITICAL: Base split missing at {INPUT_SPLIT_FILE}")

    # Load global dataset assets
    with open(INPUT_SPLIT_FILE, 'r') as f: orig_splits = json.load(f)
    with open(DATA_DICT_PATH, 'r') as f: affinity_data = json.load(f)
    with open(PSM_COMPLEXES, 'r') as f: matrix_complexes = json.load(f)
    
    matrix_complexes = [str(c).strip().lower() for c in matrix_complexes]
    complex_to_idx = {comp: idx for idx, comp in enumerate(matrix_complexes)}
    num_complexes = len(matrix_complexes)
    
    mat_tanimoto = np.load(PSM_TANIMOTO)
    casf2016_valid = set([str(c).strip().lower() for c in orig_splits['casf2016'] if str(c).strip().lower() in complex_to_idx])

    # ──────────────────────────────────────────────────────────────
    # 🚀 Core Step 1: Fast single‑linkage clustering via SciPy sparse graph (equivalent to Union‑Find)
    # ──────────────────────────────────────────────────────────────
    print(f"STATUS: Compiling adjacency graph with Tanimoto boundary > {CLUSTER_THRESHOLD}...")
    # An undirected edge is built between two nodes whenever ECFP6 similarity exceeds the threshold
    adj_matrix = mat_tanimoto >= CLUSTER_THRESHOLD
    np.fill_diagonal(adj_matrix, False)  # Remove self‑loops
    
    print("STATUS: Dissecting molecular chemical universe into connected components...")
    # Invoke high‑speed underlying C++ connected‑component solver
    n_components, labels = connected_components(csr_matrix(adj_matrix), directed=False)
    print(f"✅ Graph decomposition completed! The full molecular dataset is partitioned into {n_components} independent chemical clusters.")

    # ──────────────────────────────────────────────────────────────
    # 🚀 Core Step 2: Multi‑modal meta‑feature extraction (for intra‑cluster tie‑breaking regression)
    # ──────────────────────────────────────────────────────────────
    print("STATUS: Ingesting crystal quality features for intra‑cluster tie‑breaking...")
    refined = np.zeros(num_complexes, dtype=np.int32)
    resolutions = np.zeros(num_complexes, dtype=np.float32)
    
    for idx, comp in enumerate(matrix_complexes):
        meta = affinity_data.get(comp, {})
        if 'refined' in meta.get('dataset', []):
            refined[idx] = 1
        res_str = meta.get('resolution', '3.0')
        if res_str == 'NMR' or res_str is None:
            resolutions[idx] = 3.0
        else:
            try:
                resolutions[idx] = float(res_str)
            except:
                resolutions[idx] = 3.0

    # ──────────────────────────────────────────────────────────────
    # 🚀 Core Step 3: Cluster‑level leakage isolation & intra‑cluster redundancy pruning
    # ──────────────────────────────────────────────────────────────
    print("STATUS: Executing cluster‑level leakage isolation and intra‑cluster skeleton pruning...")
    
    # Pre‑build index list for nodes belonging to each cluster
    cluster_to_node_idxs = {c_id: [] for c_id in range(n_components)}
    for idx, c_id in enumerate(labels):
        cluster_to_node_idxs[c_id].append(idx)
        
    final_train_dataset = []
    leaking_clusters_count = 0
    redundant_pruned_count = 0
    
    for c_id, node_idxs in cluster_to_node_idxs.items():
        node_idxs = np.array(node_idxs, dtype=np.int32)
        cluster_complexes = [matrix_complexes[idx] for idx in node_idxs]
        
        # 1. Hard leakage prevention rule: cluster‑wise exclusion
        # If any molecule within the cluster belongs to the test set, all training molecules in this cluster are discarded entirely!
        has_test_leak = any(comp in casf2016_valid for comp in cluster_complexes)
        if has_test_leak:
            leaking_clusters_count += 1
            continue  # Drop the whole cluster
            
        # 2. Intra‑cluster redundancy sampling: for leakage‑free training‑only clusters, perform representative sampling
        # Guarantee one high‑quality representative per scaffold cluster in the final training set
        if len(node_idxs) == 1:
            # Isolated singleton cluster, keep directly
            final_train_dataset.append(cluster_complexes[0])
        else:
            # Dense homolog cluster: select best representative via multi‑criteria tie‑breaking
            # Priority A: prefer samples from the Refined set
            cluster_refined = refined[node_idxs]
            best_candidates = node_idxs[cluster_refined == np.max(cluster_refined)]
            
            # Priority B: best crystallographic resolution (smaller value indicates higher quality)
            cluster_resolutions = resolutions[best_candidates]
            best_idx = best_candidates[np.argmin(cluster_resolutions)]
            
            final_train_dataset.append(matrix_complexes[best_idx])
            redundant_pruned_count += (len(node_idxs) - 1)

    print("\n" + "="*60)
    print("(SUMMARY)")
    print(f"METRICS: Total Extracted Clusters        N = {n_components}")
    print(f"METRICS: Test‑Leaking Purged Clusters    N = {leaking_clusters_count}")
    print(f"METRICS: Intra‑cluster Pruned Redundancy N = {redundant_pruned_count}")
    print(f"🔥 METRICS: Definitive Purified Train Pool N = {len(final_train_dataset)}")
    print(f"METRICS: Retained Independent Test (CASF) N = {len(casf2016_valid)}")
    print("="*60)

    # Assemble and dump output split dictionary
    split_dict = {
        'casf2016': list(casf2016_valid),
        'train': final_train_dataset,
        'casf2016_indep': list(casf2016_valid)
    }
    
    with open(OUTPUT_SPLIT_FILE, 'w', encoding='utf-8') as f:
        json.dump(split_dict, f, indent=4)
    print(f"STATUS: Ultimate cluster‑purified split successfully written to {OUTPUT_SPLIT_FILE}")


if __name__ == '__main__':
    main()