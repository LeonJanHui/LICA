# Step 3 — Chemical / structural profile of High-Δ complexes

High-Δ = top quartile of $\Delta_{abs}=|e_T|-|e_R|$ (n=72, $\Delta \ge 0.275$).

## Headline

Significant continuous shifts (High-Δ vs rest, MWU p<0.05):
- **Rotatable bonds**: higher (mean 7.43 vs 5.42, Cliff's δ=+0.28, p=0.000369)
- **H-bonds**: higher (mean 4.54 vs 3.45, Cliff's δ=+0.19, p=0.0129)
- **Max ligand Tc (Tanimoto train)**: higher (mean 0.568 vs 0.525, Cliff's δ=+0.18, p=0.0217)

Significant binary enrichments (Fisher p<0.05):
- **Flexible (≥8 rotbonds)**: enriched (40% vs 23%, OR=2.3, p=0.00589)
- **Has T-stacking**: depleted (8% vs 19%, OR=0.39, p=0.0414)
- **Has hydrophobic**: depleted (86% vs 95%, OR=0.34, p=0.0198)

## Files

| File | Content |
|------|---------|
| `high_delta_complexes.csv` | High-Δ rows + chem/struct columns |
| `continuous_feature_tests.csv` | MWU + Cliff's δ vs rest / vs Low-Δ |
| `binary_enrichment.csv` | Rates + Fisher OR |
| `delta_by_chem_bins.csv` | mean Δ by MW / flex / TPSA / pK / pocket / leakage |
| `spearman_delta_vs_features.csv` | Rank correlation of features with Δ |
| `fig_highdelta_feature_cliffs.*` | Effect-size forest (continuous) |
| `fig_highdelta_binary_enrichment.*` | Rate-diff enrichment |
| `fig_delta_by_chem_bins.*` | mean Δ bars by bins |
| `fig_highdelta_boxplots.*` | Distributions by Δ quartile |

## Re-run

```bash
/home/dldx/miniconda3/envs/boltz-egnn/bin/python scripts/analyze_high_delta_chemotype.py
```

## Top continuous (by |Cliff's δ|)

- Rotatable bonds: mean_diff=+2.01, δ=+0.28, p=0.000369
- H-bonds: mean_diff=+1.1, δ=+0.19, p=0.0129
- Max ligand Tc (Tanimoto train): mean_diff=+0.0436, δ=+0.18, p=0.0217
- cLogP: mean_diff=-0.889, δ=-0.15, p=0.0566
- H-bond donors: mean_diff=+0.551, δ=+0.15, p=0.0526
- H-bond acceptors: mean_diff=+0.845, δ=+0.14, p=0.0795
- Max ligand Tc (Random train): mean_diff=+0.0586, δ=+0.14, p=0.0808
- TPSA: mean_diff=+20.4, δ=+0.14, p=0.0852

## Top binary (by |rate diff|)

- Flexible (≥8 rotbonds): 40% vs 23% (OR=2.3, p=0.00589)
- Has T-stacking: 8% vs 19% (OR=0.39, p=0.0414)
- Has salt bridge: 35% vs 25% (OR=1.6, p=0.13)
- Has hydrophobic: 86% vs 95% (OR=0.34, p=0.0198)
- Large ligand (MW≥500): 19% vs 12% (OR=1.7, p=0.168)
- Leakish (max_R≥0.7 & max_T<0.7): 38% vs 31% (OR=1.3, p=0.313)
- High pocket TM (≥0.8): 92% vs 97% (OR=0.32, p=0.0811)
- Has π–π: 19% vs 15% (OR=1.4, p=0.351)

## Bins with largest mean Δ

- **mw_bin**: highest mean Δ in `<300` (Δ=+0.071, n=113); lowest in `400–500` (Δ=-0.004, n=53)
- **flex_bin**: highest mean Δ in `≥13` (Δ=+0.228, n=22); lowest in `0–3` (Δ=-0.083, n=85)
- **tpsa_bin**: highest mean Δ in `≥140` (Δ=+0.149, n=46); lowest in `90–140` (Δ=+0.004, n=98)
- **pk_bin**: highest mean Δ in `7–9` (Δ=+0.066, n=83); lowest in `5–7` (Δ=+0.010, n=95)
- **leakish_flag**: highest mean Δ in `Leakish` (Δ=+0.099, n=93); lowest in `Not leakish` (Δ=-0.001, n=192)
- **chemotype**: highest mean Δ in `Nucleotide-like` (Δ=+0.125, n=4); lowest in `Other` (Δ=+0.024, n=253)

## Strongest Spearman(feature, Δ)

- Salt bridges: ρ=+0.155, p=0.00898
- Rotatable bonds: ρ=+0.132, p=0.0254
- Max ligand Tc (Tanimoto train): ρ=+0.125, p=0.0356
- H-bonds: ρ=+0.120, p=0.0429
- Max ligand Tc (Random train): ρ=+0.102, p=0.0861
- cLogP: ρ=-0.098, p=0.0984
