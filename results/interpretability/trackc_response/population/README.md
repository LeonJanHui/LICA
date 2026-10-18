# CASF-2016 population: Full vs w/o Track C attention

## Design

- **Full**: `checkpoints/mydata_repeats` (3-seed mean), Track C → protein-atom / residue ranks
- **w/o-C**: `checkpoints/ablation_wo_track_c` (3-seed mean), Track B pocket atoms
- Not inference-time masking of Track C in the Full model.
- Contacts: PLIP residues on curated complex PDBs.

## Cohort

- CASF-2016 graphs: **285**
- Successful metrics: **285**
- With PLIP contacts (n_plip>0): **280**

## Key results

- Spearman(Full-C atom, w/o-C Track B) median = **-0.155** (mean -0.144; fraction ρ<0.3 = 1.00)
- PLIP overlap@5: Full Track C median = **0.600** vs w/o-C Track B = **0.200** (median Δ = **0.400**)
- Wilcoxon signed-rank (Full > w/o-C overlap@5): p = **1.59e-43**
- Top-10 Track C pair distance median = **3.40 Å**

## Figures

- `casf2016_full_vs_woc_population.png` — Spearman / overlap@5 / Δoverlap
- `casf2016_trackc_top10_distance.png` — short-range check

## Suggested Results sentence

> Across CASF-2016, protein-atom attention from independently trained Full (Track C) and w/o-Track-C (Track B) models is largely uncorrelated (median Spearman ρ ≈ -0.16), indicating redistribution of pocket focus. Track C residue ranks agree better with PLIP contacts than w/o-C Track B (median overlap@5 0.60 vs 0.20; median Δ ≈ 0.40), and Top-10 Track C pairs remain short-range (median ≈ 3.4 Å).

## Re-run

```bash
/home/dldx/miniconda3/envs/boltz-egnn/bin/python \
  scripts/xai_trackc_response/run_casf2016_population.py --device cuda
```

