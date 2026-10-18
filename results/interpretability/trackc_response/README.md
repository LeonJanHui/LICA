# Track C interpretability — reviewer response

## What this answers

1. **3D mapping**: Track C (Full) / Track A+B (w/o C) attention → atom B-factors on the complex PDB.
2. **Case selection** (explicit, locked below).
3. **Outlier**: included to show failure modes.
4. **Full vs w/o Track C**: same complexes, side-by-side maps + PLIP overlap table.

## Case selection criteria

| PDB | Role | Criterion |
|-----|------|-----------|
| **4j3l** | lowest_error | Smallest Full-model abs error on CASF-2016 (abs_err≈0.008). Also carries π–π contacts (n_pi_pi=2), serving as an accurate exemplar. |
| **2qbr** | interaction_type_Hbond_saltbridge | Low abs error (≈0.041) with a chemically typical polar interface: n_hbonds=7 and n_salt_bridges=2. Chosen to illustrate H-bond/salt-bridge recognition. |
| **4tmn** | outlier_high_error | High-error outlier (abs_err≈3.40) with a flexible, H-bond-rich ligand. Included to expose model limitations on attention maps. |

## Models

- Full (+Track C): `checkpoints/mydata_repeats` (3-seed mean)
- w/o Track C: `checkpoints/ablation_wo_track_c` (3-seed mean)

## Key quantitative result

See `full_vs_woc_summary.csv`.

- mean PLIP overlap@5: **Track C (Full) = 0.667** vs **w/o-C Track B = 0.133** (Δ=+0.533)
- mean Spearman(protein-atom attention Full vs w/o-C) = -0.159 (≪1 ⇒ redistribution)
- mean Top-10 Track C pair distance = 3.39 Å

## Suggested Results sentence

> Mapping Track C attention onto complex atoms shows short-range interface contacts (~3.4 Å) that agree with PLIP residues (overlap@5 ≈ 0.67). Removing Track C lowers this agreement and reorders pocket attention (Spearman ≈ -0.16), indicating Track C changes the learned interaction representation.

## Render (local PyMOL)

```bash
pymol -cq results/interpretability/trackc_response/2qbr/full/2qbr_attention.pml
pymol -cq results/interpretability/trackc_response/2qbr/wo_track_c/2qbr_attention.pml
```

## Re-run

```bash
/home/dldx/miniconda3/envs/boltz-egnn/bin/python scripts/xai_trackc_response/run_all.py
```

## Caveats

- Weights are within-complex relative attentions, **not** binding free energies.
- w/o Track C has no interface-pair ω_ij; comparison uses Track A/B atom maps.
- CASF-2016 case studies; mechanistic evidence, not a population claim.
