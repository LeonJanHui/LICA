"""Evaluate 3-repeat checkpoints on the configured test split; write metrics CSV + scatter."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train import (  # noqa: E402
    build_dataset,
    build_model,
    load_checkpoint,
    load_config,
    make_loader,
    metrics,
    predict,
    write_metrics_csv,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_theme(style="ticks")


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    n = len(y_true)
    slope = intercept = 0.0
    sd_val = 0.0
    if n > 2 and np.std(y_pred) > 0:
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        y_fitted = slope * y_pred + intercept
        sd_val = float(np.sqrt(np.sum((y_true - y_fitted) ** 2) / (n - 2)))

    r_pearson = r_spearman = 0.0
    if n > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r_pearson = float(pearsonr(y_true, y_pred)[0])
        r_spearman = float(spearmanr(y_true, y_pred)[0])

    return rmse, mae, sd_val, r_pearson, r_spearman, slope, intercept


def plot_fitting_scatter(y_true, y_pred, output_path, title="Test set"):
    rmse, mae, sd_val, r_p, r_s, slope, intercept = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=300)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, loc="center")

    min_bound = min(y_true.min(), y_pred.min()) - 0.5
    max_bound = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot(
        [min_bound, max_bound],
        [min_bound, max_bound],
        linestyle="--",
        color="#7F7F7F",
        linewidth=1.2,
        label="Ideal (y = x)",
        zorder=1,
    )

    x_fit = np.linspace(min_bound, max_bound, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(
        x_fit,
        y_fit,
        color="#D32F2F",
        linewidth=2.0,
        label=f"Fitting (y = {slope:.2f}x + {intercept:.2f})",
        zorder=3,
    )

    ax.scatter(
        y_pred,
        y_true,
        color="#FF6A00",
        edgecolors="black",
        linewidths=0.4,
        s=35,
        alpha=0.6,
        label="EGNN (mean over repeats)",
        zorder=2,
    )

    ax.set_xlabel(
        "Predicted Binding Affinity ($p K_d / p K_i$)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Experimental Binding Affinity ($p K_d / p K_i$)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_xlim(min_bound, max_bound)
    ax.set_ylim(min_bound, max_bound)

    text_str = (
        f"$\\mathbf{{Pearson\\,R:}}$ {r_p:.4f}\n"
        f"$\\mathbf{{Spearman\\,\\rho:}}$ {r_s:.4f}\n"
        f"$\\mathbf{{RMSE:}}$ {rmse:.4f}\n"
        f"$\\mathbf{{MAE:}}$ {mae:.4f}\n"
        f"$\\mathbf{{SD:}}$ {sd_val:.4f}"
    )
    props = dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        edgecolor="#D3D3D3",
        alpha=0.85,
    )
    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=9.5,
    )
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    sns.despine(ax=ax, trim=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[PLOT] saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Default: <save_model_dir>/mydata_test_metrics.csv",
    )
    parser.add_argument(
        "--output_image",
        default=str(PROJECT_ROOT / "results" / "mydata_repeat_scatter.png"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    t = cfg["training"]
    d = cfg["dataset"]
    save_dir = Path(cfg["paths"]["save_model_dir"])
    num_repeats = int(t.get("num_repeats", 3))
    base_seed = int(t.get("seed", 42))

    test_set = build_dataset(cfg, d["test_split"])
    test_loader = make_loader(test_set, cfg, False)

    rows = []
    all_preds = []
    labels = None

    for repeat in range(num_repeats):
        model_path = save_dir / f"repeat{repeat}" / "best.pt"
        if not model_path.is_file():
            print(f"[WARNING] missing checkpoint: {model_path}")
            continue

        seed = base_seed + repeat
        model = build_model(cfg, device)
        load_checkpoint(model_path, model, device)
        y_true, y_pred = predict(model, test_loader, device, f"Test R{repeat}")
        labels = y_true if labels is None else labels
        all_preds.append(y_pred)

        rmse, mae, sd, pcc, spearman = metrics(y_true, y_pred)
        print(
            f"repeat{repeat}: RMSE={rmse:.6f}  MAE={mae:.6f}  "
            f"PCC={pcc:.6f}  SD={sd:.6f}"
        )
        rows.append(
            {
                "repeat": repeat,
                "seed": seed,
                "checkpoint": str(model_path),
                "n_test": len(test_set),
                "ValRMSE": "",
                "RMSE": float(rmse),
                "MAE": float(mae),
                "PCC": float(pcc),
                "Spearman": float(spearman),
                "SD": float(sd),
            }
        )

    if not rows:
        print(f"[ERROR] no checkpoints found under {save_dir}/repeat*/best.pt")
        return

    csv_path = Path(args.output_csv) if args.output_csv else save_dir / "mydata_test_metrics.csv"
    write_metrics_csv(csv_path, rows)
    print(f"[METRICS] saved -> {csv_path}")

    arr = np.array([[x["RMSE"], x["MAE"], x["PCC"]] for x in rows], dtype=np.float64)
    mean = arr.mean(0)
    std = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(3)
    print("------- mean ± std over repeats -------")
    print(
        f"RMSE={mean[0]:.4f}±{std[0]:.4f} | "
        f"MAE={mean[1]:.4f}±{std[1]:.4f} | "
        f"PCC={mean[2]:.4f}±{std[2]:.4f}"
    )

    mean_pred = np.mean(np.stack(all_preds, axis=0), axis=0)
    plot_fitting_scatter(
        labels,
        mean_pred,
        args.output_image,
        title=f"{d['test_split']} (mean of {len(rows)} repeats)",
    )


if __name__ == "__main__":
    main()
