# /home/dldx/Boltz-EGNN-Affinity/src/preprocess/compute_pairwise_tanimoto.py
import os
import sys
import json
import logging
import warnings
import argparse
from time import time
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
log_file = 'pairwise_similarity_tanimoto_error.log'
rotating_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024, backupCount=3)
rotating_handler.setFormatter(log_formatter)
rotating_handler.setLevel(logging.ERROR)
logger = logging.getLogger('root')
logger.setLevel(logging.ERROR)
logger.addHandler(rotating_handler)


def parse_curated_sdf_files(root_dir, complexes, ligand_ext=".sdf"):
    """Parses ligand SDF files from hierarchical subdirectories safely."""
    parsed_molecules = {}
    for comp_id in complexes:
        sdf_path = os.path.join(root_dir, comp_id, f"{comp_id}_ligand{ligand_ext}")
        try:
            mol = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)[0]
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
            parsed_molecules[comp_id] = mol
        except Exception:
            parsed_molecules[comp_id] = None
    return parsed_molecules


def save_matrices_to_npy(similarity_matrix, save_path):
    """Enforces symmetry, fills identity diagonal, and flushes matrix to disk."""
    symmetric_matrix = np.maximum(similarity_matrix, similarity_matrix.T)
    np.fill_diagonal(symmetric_matrix, 1.0)
    np.save(save_path, symmetric_matrix.astype(np.float32))
    print(f"INFO: ECFP6 Tanimoto similarity matrix successfully saved to {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compute pure pairwise Tanimoto similarity matrix using standard ECFP6.")
    parser.add_argument('--root_dir', type=str, default="/home/dldx/Boltz-EGNN-Affinity/data/curated_pdbbind")
    parser.add_argument('--index_file', type=str, default="/home/dldx/Boltz-EGNN-Affinity/data/curated_pdbbind/curated_indices.csv")
    parser.add_argument('--save_dir', type=str, default="/home/dldx/Boltz-EGNN-Affinity/data/similarity/pairwise_similarity_matrix")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(args.index_file):
        print(f"CRITICAL ERROR: Curated index ledger not found at {args.index_file}")
        sys.exit(1)

    df_index = pd.read_csv(args.index_file)
    complexes = sorted(df_index['pdb_id'].str.strip().str.lower().tolist())
    num_complexes = len(complexes)
    print(f"INFO: Identified {num_complexes} curated complexes from ledger table.", flush=True)

    complexes_json_path = os.path.join(args.save_dir, 'pairwise_similarity_complexes.json')
    with open(complexes_json_path, 'w') as f:
        json.dump(complexes, f)

    PSM_tanimoto_file = os.path.join(args.save_dir, "pairwise_similarity_matrix_tanimoto.npy")
    similarity_matrix_tanimoto = np.zeros((num_complexes, num_complexes), dtype=np.float32)

    print("STATUS: Parsing ligand topologies into virtual memory space...", flush=True)
    parsed_molecules = parse_curated_sdf_files(args.root_dir, complexes)
    print("STATUS: Ligand parsing complete.", flush=True)

    # ──────────────────────────────────────────────────────────────
    # 2048-bit ECFP6 (Morgan Radius=3)
    # ──────────────────────────────────────────────────────────────
    print("STATUS: Pre-computing standard 2048-bit Morgan Bit Fingerprints (ECFP6)...", flush=True)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)  # 🎯 Radius=3 代表 ECFP6
    fps = []
    for comp_id in complexes:
        mol = parsed_molecules.get(comp_id)
        if mol is not None:
            fps.append(mfpgen.GetFingerprint(mol))
        else:
            fps.append(None)
    print("STATUS: ECFP6 Bit Fingerprint generation complete. Commencing vectorized matrix computation...", flush=True)

    global_tic = time()
    for i in range(num_complexes):
        fp_i = fps[i]
        if fp_i is None:
            continue
            
        target_fps = []
        target_js = []
        for j in range(i + 1, num_complexes):
            if fps[j] is not None:
                target_fps.append(fps[j])
                target_js.append(j)
                
        if not target_fps:
            continue
            
        try:
            scores = DataStructs.BulkTanimotoSimilarity(fp_i, target_fps)
            for j_col, score in zip(target_js, scores):
                similarity_matrix_tanimoto[i, j_col] = score
                similarity_matrix_tanimoto[j_col, i] = score
                
        except Exception as e:
            logger.error(f"Error matrix row bulk computation at index {i}: {str(e)}")

        if (i + 1) % 1000 == 0:
            print(f"Progress node check: [{i+1}/{num_complexes}] successfully compiled.", flush=True)
            save_matrices_to_npy(similarity_matrix_tanimoto, PSM_tanimoto_file)

    save_matrices_to_npy(similarity_matrix_tanimoto, PSM_tanimoto_file)
    print(f"STATUS: Matrix compilation complete. Elapsed time: {time() - global_tic:.2f} seconds.", flush=True)


if __name__ == '__main__':
    main()