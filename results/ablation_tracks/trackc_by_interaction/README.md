# Track C affinity gain vs PLIP interaction type

## Design

- Full: `checkpoints/mydata_repeats` (3-seed mean), Tanimoto/mydata split.
- w/o Track C: `checkpoints/ablation_wo_track_c` (3-seed mean), same split.
- Per complex: $\Delta|\varepsilon| = |y-\hat y_{w/o C}| - |y-\hat y_{Full}|$. Positive means Track C lowers absolute pK error.
- Interaction counts: `results/T-split/test_interactions.csv` (PLIP).
- This is a **prediction-error** stratification. It is separate from attention–PLIP overlap.

## Cohort

- n = 285
- mean Δ|ε| = 0.062 (median 0.047)
- fraction with Δ|ε| > 0 (Track C helps) = 0.551
- mean |ε| Full = 0.932; w/o C = 0.993

## Spearman (Δ|ε| vs count)

          feature                     label   n  spearman_delta_abs        p
         n_hbonds                   H-bonds 285           -0.089246 0.132835
   n_salt_bridges              Salt bridges 285           -0.050362 0.396993
    n_hydrophobic               Hydrophobic 285           -0.154234 0.009109
          n_pi_pi                       π–π 285           -0.044307 0.456231
     n_t_stacking                T-stacking 285           -0.115063 0.052332
      n_cation_pi                  Cation–π 285            0.130326 0.027815
        n_halogen                   Halogen 285            0.057844 0.330531
  n_water_bridges             Water bridges 285            0.001436 0.980742
n_metal_complexes                     Metal 285                 NaN      NaN
          n_polar       Polar (H-bond+salt) 285           -0.097035 0.102091
       n_aromatic Aromatic (π–π+T+cation–π) 285           -0.076619 0.197158
       n_contacts      All counted contacts 285           -0.181268 0.002125

## Present vs absent

         feature              label  n_present  n_absent  mean_delta_present  mean_delta_absent  median_delta_present  median_delta_absent  mean_abs_full_present  mean_abs_woc_present  mean_abs_full_absent  mean_abs_woc_absent  p_mannwhitney
       has_hbond             H-bond        267        18            0.062624           0.045138              0.043944             0.068777               0.941009              1.003633              0.794385             0.839523       0.789160
 has_salt_bridge        Salt bridge         79       206            0.008043           0.082028              0.000457             0.066347               0.959434              0.967476              0.921131             1.003159       0.292558
 has_hydrophobic        Hydrophobic        264        21            0.051163           0.191718              0.043876             0.270403               0.917467              0.968630              1.111276             1.302995       0.175454
    has_pi_stack π-stack (π–π or T)         83       202           -0.007188           0.089751             -0.036273             0.073154               1.009858              1.002670              0.899654             0.989405       0.048261
   has_cation_pi           Cation–π         13       272            0.330412           0.048668              0.309716             0.040152               0.537429              0.867841              0.950594             0.999262       0.028092
     has_halogen            Halogen         19       266            0.130057           0.056624              0.100787             0.042926               0.754495              0.884552              0.944409             1.001033       0.338045
has_water_bridge       Water bridge        210        75            0.055384           0.078700              0.049007             0.043808               0.952322              1.007706              0.874141             0.952841       0.995442
       has_metal              Metal          0       285                 NaN           0.061520                   NaN             0.047229                    NaN                   NaN              0.931748             0.993268            NaN

