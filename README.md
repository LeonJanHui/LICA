# Boltz-EGNN-Affinity

Relation-aware equivariant graph network for protein–ligand binding affinity (pK) on PDBbind, evaluated on CASF-2016.

The model builds a complex graph and passes messages with a relation-aware EGNN. Three readouts are fused for the final prediction:

- **Track A** pools the ligand.
- **Track B** pools the pocket, conditioned on the ligand.
- **Track C** attends over ligand–pocket atom pairs inside 4.5 Å.

## Layout

| Path | Role |
|---|---|
| `src/train.py` | Training and evaluation |
| `src/model/` | EGNN layers and the three-track network |
| `src/dataset.py` | Complex graph construction |
| `src/preprocess/` | PDBbind curation, ECFP6 Tanimoto matrix, connected-component split |
| `configs/ablation/` | Full model and Track A/B/C ablations, including parameter-matched controls |
| `configs/train_config.yaml` | Random-split training config |
| `scripts/` | Ablation table, error analysis, and Track C interpretability |
| `docs/ABLATION_PROTOCOL.md` | What “remove a track” means |

`data/`, `checkpoints/`, and `baselines/` stay on disk for local runs and are not part of this source repository.

## Data protocol

Curated complexes are filtered in `src/preprocess/curate_pdbbind.py` (resolution ≤ 3.0 Å, ligand heavy atoms ≥ 5, molecular weight ≥ 100 Da). The main results use the Tanimoto split in `data/similarity/PDBbind_split_mydata.json`: ligands are connected when ECFP6 Tanimoto similarity is at least 0.75, and any connected component that touches CASF-2016 is removed from development. A random split of the same curated pool is available for comparison.

## Train

From the repository root, with the project environment active:

```bash
python src/train.py --config configs/ablation/00_full.yaml
```

Ablation configs live in `configs/ablation/`. The reported protocol uses AdamW, learning rate `5e-4`, batch size 32, 4 EGNN layers, hidden size 256, and 3 seeds (`42`, `43`, `44`). Ablation runs stop after 30 epochs without a validation-RMSE improvement (`patience: 30`).

## Environment

Training used PyTorch 2.4.0+cu124 and PyTorch Geometric 2.7.0. See `requirements.txt`. A full conda export is in `src/environment.yaml` (environment name `lica-egnn`).
