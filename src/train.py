from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
warnings.filterwarnings("ignore", category=UserWarning)

from dataset import ComplexGraphDataset
from features import EDGE_STATIC_DIM, NODE_FEATURE_DIM, NUM_EDGE_TYPES
from model.network import PureGeometricAffinityNetwork


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    sd = 0.0
    if len(y_true) > 2 and np.std(y_pred) > 0:
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        residual = y_true - (slope * y_pred + intercept)
        sd = float(np.sqrt(np.sum(residual**2) / (len(y_true) - 2)))

    pearson = spearman = 0.0
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(pearsonr(y_true, y_pred)[0])
        spearman = float(spearmanr(y_true, y_pred)[0])

    return rmse, mae, sd, pearson, spearman


def build_dataset(cfg, split):
    p, d = cfg["paths"], cfg["dataset"]
    dataset = ComplexGraphDataset(
        root_dir=p["root_dir"],
        ledger_json=p["ledger_json"],
        data_dict_json=p["data_dict_json"],
        save_dir=p["save_dir"],
        split_name=split,
        interface_cutoff=float(d["interface_cutoff"]),
        protein_spatial_cutoff=float(d["protein_spatial_cutoff"]),
        label_col=d["label_col"],
        include_hydrogens=bool(d.get("include_hydrogens", False)),
        water_policy=d.get("water_policy", "remove"),
        water_cutoff=float(d.get("water_cutoff", 4.0)),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No valid graphs in split: {split}")
    print(f"[DATA] {split}: {len(dataset)} graphs")
    return dataset


def build_model(cfg, device):
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
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[MODEL] tracks A/B/C="
        f"{int(model.use_track_a)}/{int(model.use_track_b)}/{int(model.use_track_c)} "
        f"param_match={model.param_match}"
        f"{(':' + model.param_match_track) if model.param_match else ''} "
        f"n_params={n_params:,}"
    )
    return model


def make_loader(dataset, cfg, shuffle):
    t = cfg["training"]
    workers = int(t.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(t["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(t.get("pin_memory", True)),
        persistent_workers=bool(t.get("persistent_workers", False)) and workers > 0,
    )


def train_epoch(model, loader, optimizer, device, clip_norm, epoch, epochs):
    model.train()
    loss_sum = sample_count = 0
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{epochs:03d}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        target = batch.y.view(-1)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch).view(-1)
        loss = nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        n = target.numel()
        loss_sum += loss.item() * n
        sample_count += n
        progress.set_postfix(mse=f"{loss.item():.4f}")

    return loss_sum / sample_count


@torch.inference_mode()
def predict(model, loader, device, desc):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        batch = batch.to(device, non_blocking=True)
        y_true.append(batch.y.view(-1).cpu().numpy())
        y_pred.append(model(batch).view(-1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def evaluate(model, loader, device, desc):
    return metrics(*predict(model, loader, device, desc))


def save_checkpoint(path, model, epoch, best_val_rmse):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_rmse": best_val_rmse,
        },
        path,
    )


def load_checkpoint(path, model, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{path} does not match current model parameters. Delete old checkpoint and retrain."
        ) from exc


def train_repeat(cfg, train_set, valid_set, test_set, repeat, seed, device, run_dir):
    t = cfg["training"]
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "best.pt"
    done_path = run_dir / "DONE"

    train_loader = make_loader(train_set, cfg, True)
    valid_loader = make_loader(valid_set, cfg, False)
    test_loader = make_loader(test_set, cfg, False)
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if bool(t.get("resume_existing", False)) and done_path.exists() and model_path.exists():
        load_checkpoint(model_path, model, device)
        v_rmse, v_mae, v_sd, v_p, v_s = evaluate(model, valid_loader, device, f"R{repeat} Valid")
        t_rmse, t_mae, t_sd, t_p, t_s = evaluate(model, test_loader, device, f"R{repeat} Test")
        print(
            f"[LOADED] Repeat {repeat} | ValRMSE={v_rmse:.4f} | "
            f"Test RMSE={t_rmse:.4f} MAE={t_mae:.4f} PCC={t_p:.4f}"
        )
        return {
            "repeat": repeat,
            "seed": seed,
            "checkpoint": str(model_path),
            "n_test": len(test_set),
            "n_params": n_params,
            "ValRMSE": v_rmse,
            "RMSE": t_rmse,
            "MAE": t_mae,
            "PCC": t_p,
            "Spearman": t_s,
            "SD": t_sd,
        }

    set_seed(seed)
    base_lr = float(t["learning_rate"])
    warmup = int(t.get("warmup_epochs", 0))
    epochs = int(t["epochs"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(t["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup, 1),
        eta_min=float(t.get("min_learning_rate", 1e-6)),
    )

    best_val_rmse, bad_epochs = float("inf"), 0
    best_test = None
    for epoch in range(1, epochs + 1):
        if warmup and epoch <= warmup:
            for group in optimizer.param_groups:
                group["lr"] = base_lr * epoch / warmup

        train_mse = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            float(t.get("gradient_clip_norm", 1.0)),
            epoch,
            epochs,
        )
        v_rmse, v_mae, v_sd, v_p, v_s = evaluate(model, valid_loader, device, "Valid")
        t_rmse, t_mae, t_sd, t_p, t_s = evaluate(model, test_loader, device, "Test")
        if epoch > warmup:
            scheduler.step()

        print(
            f"Repeat {repeat} | Epoch {epoch:03d} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"Train MSE={train_mse:.4f} | "
            f"Val RMSE={v_rmse:.4f} PCC={v_p:.4f} | "
            f"Test RMSE={t_rmse:.4f} MAE={t_mae:.4f} PCC={t_p:.4f}"
        )

        if v_rmse < best_val_rmse:
            best_val_rmse, bad_epochs = v_rmse, 0
            save_checkpoint(model_path, model, epoch, best_val_rmse)
            best_test = (t_rmse, t_mae, t_sd, t_p, t_s)
        else:
            bad_epochs += 1
            if bad_epochs >= int(t["patience"]):
                print(
                    f"[EARLY STOP] Repeat {repeat}, epoch {epoch} "
                    f"(Best Val RMSE: {best_val_rmse:.4f})"
                )
                break

    load_checkpoint(model_path, model, device)
    v_rmse, v_mae, v_sd, v_p, v_s = evaluate(model, valid_loader, device, f"R{repeat} Valid*")
    t_rmse, t_mae, t_sd, t_p, t_s = evaluate(model, test_loader, device, f"R{repeat} Test*")
    if best_test is None:
        best_test = (t_rmse, t_mae, t_sd, t_p, t_s)

    summary = (
        f"Best(val) - ValRMSE: {v_rmse:.6f} | "
        f"Test RMSE: {t_rmse:.6f}, MAE: {t_mae:.6f}, SD: {t_sd:.6f}, "
        f"PCC: {t_p:.6f}, Spearman: {t_s:.6f}\n"
    )
    print(summary, end="")
    done_path.write_text(summary, encoding="utf-8")

    return {
        "repeat": repeat,
        "seed": seed,
        "checkpoint": str(model_path),
        "n_test": len(test_set),
        "n_params": n_params,
        "ValRMSE": float(v_rmse),
        "RMSE": float(t_rmse),
        "MAE": float(t_mae),
        "PCC": float(t_p),
        "Spearman": float(t_s),
        "SD": float(t_sd),
    }


def write_metrics_csv(path, rows):
    fieldnames = [
        "repeat",
        "seed",
        "checkpoint",
        "n_test",
        "n_params",
        "ValRMSE",
        "RMSE",
        "MAE",
        "PCC",
        "Spearman",
        "SD",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        if rows:
            arr = np.array(
                [[x["RMSE"], x["MAE"], x["PCC"]] for x in rows], dtype=np.float64
            )
            mean = arr.mean(0)
            std = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(3)
            writer.writerow(
                {
                    "repeat": "mean",
                    "seed": "",
                    "checkpoint": "",
                    "n_test": rows[0]["n_test"],
                    "n_params": rows[0].get("n_params", ""),
                    "ValRMSE": "",
                    "RMSE": mean[0],
                    "MAE": mean[1],
                    "PCC": mean[2],
                    "Spearman": "",
                    "SD": "",
                }
            )
            writer.writerow(
                {
                    "repeat": "std",
                    "seed": "",
                    "checkpoint": "",
                    "n_test": rows[0]["n_test"],
                    "n_params": "",
                    "ValRMSE": "",
                    "RMSE": std[0],
                    "MAE": std[1],
                    "PCC": std[2],
                    "Spearman": "",
                    "SD": "",
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
    )
    cfg = load_config(parser.parse_args().config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(
        f"[SCHEMA] node={NODE_FEATURE_DIM}, "
        f"edge_attr={EDGE_STATIC_DIM}, edge_types={NUM_EDGE_TYPES}"
    )
    t = cfg["training"]
    print(
        f"[TRAIN] repeats={t.get('num_repeats', 3)} epochs={t['epochs']} "
        f"patience={t['patience']} lr={t['learning_rate']}"
    )

    save_dir = Path(cfg["paths"]["save_model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    d = cfg["dataset"]
    train_set = build_dataset(cfg, d["train_split"])
    valid_set = build_dataset(cfg, d.get("valid_split", "valid"))
    test_set = build_dataset(cfg, d["test_split"])

    rows = []
    num_repeats = int(t.get("num_repeats", 3))
    base_seed = int(t["seed"])
    for repeat in range(num_repeats):
        seed = base_seed + repeat
        run_dir = save_dir / f"repeat{repeat}"
        print(f"\n{'=' * 18} REPEAT {repeat}/{num_repeats - 1} (seed={seed}) {'=' * 18}")
        rows.append(
            train_repeat(
                cfg, train_set, valid_set, test_set, repeat, seed, device, run_dir
            )
        )

    csv_path = save_dir / "mydata_test_metrics.csv"
    write_metrics_csv(csv_path, rows)
    print(f"\n[METRICS] saved -> {csv_path}")
    if rows:
        arr = np.array([[x["RMSE"], x["MAE"], x["PCC"]] for x in rows], dtype=np.float64)
        mean, std = arr.mean(0), arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(3)
        print(
            f"[TEST] RMSE={mean[0]:.4f}±{std[0]:.4f} | "
            f"MAE={mean[1]:.4f}±{std[1]:.4f} | "
            f"PCC={mean[2]:.4f}±{std[2]:.4f}"
        )


if __name__ == "__main__":
    main()
