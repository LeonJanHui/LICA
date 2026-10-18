# \(M_R\) vs \(M_T\) per-complex Δ (CASF-2016)

Definition:

\[
\Delta_{\mathrm{abs}} = |y-\hat y_T| - |y-\hat y_R|
\]

Positive Δ ⇒ the complex is **harder under the Tanimoto-trained model** than under the Random-trained model.

Predictions are the mean over 3 seeds for each protocol. Same 285 CASF complexes.

## Key numbers

| Model | RMSE | MAE | PCC |
|-------|------|-----|-----|
| \(M_R\) (Random) | 1.178 | 0.900 | 0.849 |
| \(M_T\) (Tanimoto) | 1.233 | 0.932 | 0.831 |

- mean Δ_abs ≈ **+0.032**
- fraction with Δ>0 ≈ **52%**
- mean Δ when max Tanimoto to Random-train ≥0.7 ≈ **+0.111**
- mean Δ when max_R <0.5 ≈ **−0.034**

## Files

| File | Content |
|------|---------|
| `per_complex_delta.csv` | Per-complex predictions, Δ, similarity, scaffold |
| `strata_summary.csv` | MAE/RMSE/Δ by similarity & scaffold strata |
| `top20_tanimoto_hurt.csv` | Largest positive Δ |
| `top20_tanimoto_helped.csv` | Largest negative Δ |
| `fig_delta_vs_random_similarity.png` | Δ vs max_random + bin boxplot |
| `fig_mae_by_stratum.png` | \(M_R\) vs \(M_T\) MAE by stratum |
| `chemotype/` | Step 3: chemical / structural profile of High-Δ |

## Re-run

```bash
python scripts/analyze_random_vs_tanimoto_delta.py
# reuse cached preds:
python scripts/analyze_random_vs_tanimoto_delta.py --skip-predict
```
