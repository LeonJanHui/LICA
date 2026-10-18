#!/usr/bin/env python3
"""Force-build missing PyG graphs for splits listed in a train config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import ComplexGraphDataset  # noqa: E402


def build(cfg: dict, split: str) -> int:
    p, d = cfg["paths"], cfg["dataset"]
    print(f"\n===== Building split={split} =====", flush=True)
    ds = ComplexGraphDataset(
        root_dir=p["root_dir"],
        ledger_json=p["ledger_json"],
        data_dict_json=p["data_dict_json"],
        save_dir=p["save_dir"],
        split_name=split,
        interface_cutoff=float(d["interface_cutoff"]),
        protein_spatial_cutoff=float(d["protein_spatial_cutoff"]),
        label_col=d["label_col"],
        include_hydrogens=bool(d["include_hydrogens"]),
        water_policy=d["water_policy"],
        water_cutoff=float(d["water_cutoff"]),
    )
    print(f"[DONE] {split}: usable graphs={len(ds)}", flush=True)
    return len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d = cfg["dataset"]
    n_tr = build(cfg, d["train_split"])
    n_va = build(cfg, d.get("valid_split", "valid"))
    n_te = build(cfg, d["test_split"])
    print(f"[SUMMARY] train={n_tr} valid={n_va} test={n_te}", flush=True)
    if n_tr < 14000 or n_va < 1000:
        raise SystemExit(
            f"[ERROR] graph counts too low (train={n_tr}, valid={n_va}); "
            "inspect manifests under data/curated_pdbbind/processed/"
        )


if __name__ == "__main__":
    main()
