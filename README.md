# LICA
Source code for "Multi-View Relation-Aware Equivariant Graph Neural Network for protein-ligand binding affinity prediction" 

## 1. Environment
conda env create -f environment.yaml
conda activate lica_env

## 2. Pre-trained Weights
Ensemble 5-fold weights available in Release v1.0

## 3. Dataset
PDBbind v2020, filtering & split rules in manuscript Methods section

## 4. Run
python train.py
