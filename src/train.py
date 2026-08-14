from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from torch.utils.data import Subset
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

    sd = 0.0
    if len(y_true) > 2 and np.std(y_pred) > 0:
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        residual = y_true - (slope * y_pred + intercept)
        sd = float(np.sqrt(np.sum(residual**2) / (len(y_true) - 2)))

    pearson = spearman = 0.0
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(pearsonr(y_true, y_pred)[0])
        spearman = float(spearmanr(y_true, y_pred)[0])

    return rmse, sd, pearson, spearman


def build_dataset(cfg, split):
    p, d = cfg["paths"], cfg["dataset"]
    # 已删除未使用的 strict_labels 参数，防止 TypeError 报错
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
    return PureGeometricAffinityNetwork(
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
    ).to(device)


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


def save_checkpoint(path, model, epoch, best_rmse):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_rmse": best_rmse,
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


def train_fold(cfg, dataset, train_idx, val_idx, fold, device, save_dir):
    t = cfg["training"]
    train_loader = make_loader(Subset(dataset, train_idx), cfg, True)
    val_loader = make_loader(Subset(dataset, val_idx), cfg, False)
    model_path = save_dir / f"best_model_fold_{fold}.pt"
    model = build_model(cfg, device)

    if bool(t.get("resume_existing", True)) and model_path.exists():
        load_checkpoint(model_path, model, device)
        result = evaluate(model, val_loader, device, f"Fold {fold} Valid")
        print(f"[LOADED] Fold {fold} | RMSE={result[0]:.4f} | Pearson={result[2]:.4f}")
        return result[0]

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

    best_rmse, bad_epochs = float("inf"), 0
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
        rmse, sd, pearson, spearman = evaluate(model, val_loader, device, "Valid")
        if epoch > warmup:
            scheduler.step()

        print(
            f"Fold {fold} | Epoch {epoch:03d} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"Train MSE={train_mse:.4f} | Val RMSE={rmse:.4f} | "
            f"Pearson={pearson:.4f}"
        )

        if rmse < best_rmse:
            best_rmse, bad_epochs = rmse, 0
            save_checkpoint(model_path, model, epoch, best_rmse)
        else:
            bad_epochs += 1
            if bad_epochs >= int(t["patience"]):
                print(f"[EARLY STOP] Fold {fold}, epoch {epoch}")
                break

    return best_rmse


def ensemble_test(cfg, device, save_dir):
    folds = int(cfg["training"]["num_folds"])
    dataset = build_dataset(cfg, cfg["dataset"]["test_split"])
    loader = make_loader(dataset, cfg, False)
    predictions, labels = [], None

    for fold in range(1, folds + 1):
        path = save_dir / f"best_model_fold_{fold}.pt"
        if not path.exists():
            print(f"[WARNING] Missing checkpoint: {path}")
            continue

        model = build_model(cfg, device)
        load_checkpoint(path, model, device)
        fold_labels, fold_predictions = predict(
            model, loader, device, f"Test Fold {fold}"
        )
        labels = fold_labels if labels is None else labels
        predictions.append(fold_predictions)

    if not predictions:
        print("[WARNING] No checkpoints found for ensemble test")
        return

    result = metrics(labels, np.mean(np.stack(predictions), axis=0))
    print("\n" + "=" * 54)
    print("CASF-2016 ENSEMBLE")
    print("=" * 54)
    print(
        f"RMSE={result[0]:.4f} | SD={result[1]:.4f} | "
        f"Pearson={result[2]:.4f} | Spearman={result[3]:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
    )
    cfg = load_config(parser.parse_args().config)
    set_seed(int(cfg["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(
        f"[SCHEMA] node={NODE_FEATURE_DIM}, "
        f"edge_attr={EDGE_STATIC_DIM}, edge_types={NUM_EDGE_TYPES}"
    )

    save_dir = Path(cfg["paths"]["save_model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg, cfg["dataset"]["train_split"])
    folds = int(cfg["training"]["num_folds"])
    if len(dataset) < folds:
        raise ValueError(f"{len(dataset)} samples cannot support {folds}-fold CV")

    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(cfg["training"]["seed"]),
    )
    fold_rmse = []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(np.arange(len(dataset))), start=1
    ):
        print(f"\n{'=' * 18} FOLD {fold}/{folds} {'=' * 18}")
        fold_rmse.append(
            train_fold(
                cfg, dataset, train_idx, val_idx, fold, device, save_dir
            )
        )

    print(
        f"\n[CV] RMSE={np.mean(fold_rmse):.4f} "
        f"± {np.std(fold_rmse):.4f}"
    )
    ensemble_test(cfg, device, save_dir)


if __name__ == "__main__":
    main()