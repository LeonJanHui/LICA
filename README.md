# LICA
Source code for "Multi-View Relation-Aware Equivariant Graph Neural Network for protein-ligand binding affinity prediction" 

## 1. Environment
conda env create -f environment.yaml
conda activate lica_env

## 2. Pre-trained Weights
Ensemble 5-fold weights available in Release v1.0

## 3. Data Preprocessing
If you need to reprocess the dataset from scratch, run the following scripts in sequence:
1. curate_pdbbind.py
2. compute_pairwise_tanimoto.py
3. cluster_based_split.py

If you skip the full preprocessing pipeline, the pre-filtered and split dataset whitelist is provided at:
data/similarity/PDBbind_split_leakage_removed_c11_purified.json

## 4. Run
python train.py

