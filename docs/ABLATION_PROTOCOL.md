# Track Ablation Protocol (Table SX)

## Design

When Track A / B / C is removed, **the entire corresponding readout branch is deleted**. The fusion module concatenates embeddings from the **remaining two tracks only**.

Everything else is held fixed:

- EGNN depth, graph construction, node/edge encoders
- Fusion MLP (same architecture; input width shrinks with fewer tracks)
- Regression head
- Learning rate, batch size, epochs, optimizer, cutoff radii, random seeds

The **only** intentional change in a plain ablation is removal of one readout branch, which reduces the number of trainable parameters.

Equivariant / coordinate-update ablation is **not** included in this suite.

## Parameter-matched controls (critical)

Reviewers may argue that similar drops across track removals are due to parameter count rather than function. For each removed track we therefore train a control:

| Config | Change |
|--------|--------|
| w/o Track A + param-matched | Track A removed; MLP placeholder with ~matched param count |
| w/o Track B + param-matched | Track B removed; MLP placeholder |
| w/o Track C + param-matched | Track C removed; MLP placeholder |

The placeholder occupies the removed track’s fusion slot so readout width matches the full model (3×`hidden_dim`). Target size equals the removed attention + projector parameter count.

## Training protocol

| Item | Value |
|------|-------|
| Split | Same mydata ledger (`train` / `valid` / `casf2016`) |
| Repeats | 3 seeds (`seed`, `seed+1`, `seed+2`) |
| Early stopping | **patience = 30** (all ablation configs) |
| Other hyperparameters | Identical to `configs/train_config.yaml` |

Configs live in `configs/ablation/`:

- `00_full.yaml`
- `01_wo_track_a.yaml` … `03_wo_track_c.yaml`
- `04_wo_track_a_param_match.yaml` … `06_wo_track_c_param_match.yaml`

## Run

```bash
# Full model already trained at checkpoints/mydata_repeats — skip 00_full:
python scripts/run_ablation_suite.py --only 01_wo_track_a,02_wo_track_b,03_wo_track_c,04_wo_track_a_param_match,05_wo_track_b_param_match,06_wo_track_c_param_match
python scripts/make_ablation_table.py
```

## Reporting

- **Table SX** (SI): full vs each ablation vs each param-matched control — RMSE, MAE, PCC (mean±std), parameter count, ΔPCC, paired t-test / Wilcoxon p-values on per-seed PCC.
- Main text: brief summary; detailed numbers in SI.

### Interpretation

- If **w/o Track C + param-matched** remains significantly worse than the full model, Track C’s contribution is functional, not a capacity artifact.
- If the gap disappears after parameter matching, the earlier drop should be attributed primarily to parameter count and reinterpreted accordingly.
