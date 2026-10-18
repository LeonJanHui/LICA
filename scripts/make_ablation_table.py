#!/usr/bin/env python3
"""Build Table SX: full vs track ablations vs param-matched controls.

Reads ``mydata_test_metrics.csv`` under each ablation checkpoint directory and
writes a markdown/CSV/LaTeX summary with RMSE/MAE/PCC (mean±std), parameter
counts, ΔPCC vs full, and paired Wilcoxon / t-test p-values on per-seed PCC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:  # pragma: no cover
    ttest_rel = wilcoxon = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = [
    ("Full model", "00_full"),
    ("w/o Track A", "01_wo_track_a"),
    ("w/o Track B", "02_wo_track_b"),
    ("w/o Track C", "03_wo_track_c"),
    ("w/o Track A + param-matched", "04_wo_track_a_param_match"),
    ("w/o Track B + param-matched", "05_wo_track_b_param_match"),
    ("w/o Track C + param-matched", "06_wo_track_c_param_match"),
]


def _load_cfg(stem: str) -> dict:
    path = PROJECT_ROOT / "configs" / "ablation" / f"{stem}.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _metrics_path(stem: str) -> Path:
    cfg = _load_cfg(stem)
    return Path(cfg["paths"]["save_model_dir"]) / "mydata_test_metrics.csv"


def _count_params(stem: str) -> int | None:
    """Instantiate the model on CPU and count trainable parameters."""
    import sys

    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from features import EDGE_STATIC_DIM, NODE_FEATURE_DIM, NUM_EDGE_TYPES
    from model.network import PureGeometricAffinityNetwork

    cfg = _load_cfg(stem)
    m, d, a = cfg["model"], cfg["dataset"], cfg["ablation"]
    model = PureGeometricAffinityNetwork(
        node_in_dim=NODE_FEATURE_DIM,
        edge_static_dim=EDGE_STATIC_DIM,
        num_edge_types=NUM_EDGE_TYPES,
        hidden_dim=int(m["hidden_dim"]),
        num_layers=int(m["num_layers"]),
        rbf_dim=int(m["rbf_dim"]),
        interface_cutoff=float(d["interface_cutoff"]),
        protein_spatial_cutoff=float(d["protein_spatial_cutoff"]),
        dropout_rate=float(m["dropout_rate"]),
        coord_update=bool(a.get("coord_update", False)),
        max_coord_step=float(m.get("max_coord_step", 0.1)),
        use_track_a=bool(a.get("use_track_a", True)),
        use_track_b=bool(a.get("use_track_b", True)),
        use_track_c=bool(a.get("use_track_c", True)),
        param_match=bool(a.get("param_match", False)),
        param_match_track=a.get("param_match_track", None),
    )
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _load_seed_rows(stem: str) -> pd.DataFrame | None:
    path = _metrics_path(stem)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[~df["repeat"].astype(str).isin(["mean", "std"])].copy()
    df["PCC"] = df["PCC"].astype(float)
    df["RMSE"] = df["RMSE"].astype(float)
    df["MAE"] = df["MAE"].astype(float)
    return df


def _fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def _pvalues(full_pcc: np.ndarray, other_pcc: np.ndarray) -> tuple[float | None, float | None]:
    if len(full_pcc) < 2 or len(other_pcc) < 2 or len(full_pcc) != len(other_pcc):
        return None, None
    if ttest_rel is None:
        return None, None
    t_p = float(ttest_rel(full_pcc, other_pcc).pvalue)
    # Wilcoxon needs non-zero differences; fall back if all equal.
    diff = full_pcc - other_pcc
    if np.allclose(diff, 0):
        w_p = 1.0
    else:
        try:
            w_p = float(wilcoxon(full_pcc, other_pcc).pvalue)
        except ValueError:
            w_p = None
    return t_p, w_p


def build_table(out_dir: Path) -> pd.DataFrame:
    rows = []
    full_df = _load_seed_rows("00_full")
    full_pcc = (
        full_df["PCC"].to_numpy(dtype=np.float64) if full_df is not None else None
    )
    full_mean_pcc = float(full_pcc.mean()) if full_pcc is not None else None

    for label, stem in VARIANTS:
        df = _load_seed_rows(stem)
        try:
            n_params = _count_params(stem)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] param count failed for {stem}: {exc}")
            n_params = None

        if df is None:
            rows.append(
                {
                    "Model": label,
                    "config": stem,
                    "n_params": n_params,
                    "RMSE": "—",
                    "MAE": "—",
                    "PCC": "—",
                    "ΔPCC": "—",
                    "p_ttest": "—",
                    "p_wilcoxon": "—",
                    "status": "missing",
                }
            )
            continue

        rmse_m, rmse_s = df["RMSE"].mean(), df["RMSE"].std(ddof=1)
        mae_m, mae_s = df["MAE"].mean(), df["MAE"].std(ddof=1)
        pcc_m, pcc_s = df["PCC"].mean(), df["PCC"].std(ddof=1)
        pcc = df["PCC"].to_numpy(dtype=np.float64)

        if stem == "00_full" or full_mean_pcc is None:
            delta = 0.0
            t_p = w_p = None
        else:
            delta = float(pcc_m - full_mean_pcc)
            t_p, w_p = _pvalues(full_pcc, pcc)

        # Prefer CSV n_params if present and consistent.
        if "n_params" in df.columns and df["n_params"].notna().any():
            n_params = int(df["n_params"].dropna().iloc[0])

        rows.append(
            {
                "Model": label,
                "config": stem,
                "n_params": n_params,
                "RMSE": _fmt_pm(rmse_m, rmse_s if len(df) > 1 else 0.0),
                "MAE": _fmt_pm(mae_m, mae_s if len(df) > 1 else 0.0),
                "PCC": _fmt_pm(pcc_m, pcc_s if len(df) > 1 else 0.0),
                "ΔPCC": f"{delta:+.4f}" if stem != "00_full" else "—",
                "p_ttest": f"{t_p:.4g}" if t_p is not None else ("—" if stem == "00_full" else "n/a"),
                "p_wilcoxon": f"{w_p:.4g}" if w_p is not None else ("—" if stem == "00_full" else "n/a"),
                "status": "ok",
                "RMSE_mean": rmse_m,
                "MAE_mean": mae_m,
                "PCC_mean": pcc_m,
                "PCC_std": pcc_s if len(df) > 1 else 0.0,
                "delta_pcc": delta if stem != "00_full" else 0.0,
                "p_ttest_raw": t_p,
                "p_wilcoxon_raw": w_p,
            }
        )

    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    display_cols = [
        "Model",
        "n_params",
        "RMSE",
        "MAE",
        "PCC",
        "ΔPCC",
        "p_ttest",
        "p_wilcoxon",
    ]
    display = table[display_cols].copy()
    csv_path = out_dir / "table_SX_ablation.csv"
    md_path = out_dir / "table_SX_ablation.md"
    tex_path = out_dir / "table_SX_ablation.tex"
    display.to_csv(csv_path, index=False)

    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    md_lines = [
        "# Table SX. Track ablation and parameter-matched controls",
        "",
        "Metrics on CASF-2016 (test), mean±std over 3 seeds. "
        "ΔPCC and p-values are vs the full model (paired on seeds).",
        "",
        header,
        sep,
        *body,
        "",
        "Notes: plain w/o Track X deletes the readout branch and fuses the "
        "remaining two embeddings; +param-matched replaces the removed branch "
        "with an MLP of comparable size. Patience=30 for all rows. "
        "Equivariant coord-update ablation is excluded from this table.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Compact LaTeX for SI.
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Track ablation and parameter-matched controls (CASF-2016, 3 seeds).}",
        r"\label{tab:SX_ablation}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Params & RMSE & MAE & PCC & $\Delta$PCC & $p$ \\",
        r"\midrule",
    ]
    for _, r in display.iterrows():
        params = f"{int(r['n_params']):,}" if pd.notna(r["n_params"]) else "—"
        pval = r["p_wilcoxon"] if r["p_wilcoxon"] not in ("—", "n/a") else r["p_ttest"]
        tex.append(
            f"{r['Model']} & {params} & {r['RMSE']} & {r['MAE']} & "
            f"{r['PCC']} & {r['ΔPCC']} & {pval} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"[OK] {csv_path}")
    print(f"[OK] {md_path}")
    print(f"[OK] {tex_path}")
    print(display.to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "ablation_tracks"),
    )
    args = parser.parse_args()
    build_table(Path(args.out_dir))


if __name__ == "__main__":
    main()
