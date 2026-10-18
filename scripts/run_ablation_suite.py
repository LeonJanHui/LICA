#!/usr/bin/env python3
"""Train the Track A/B/C ablation suite (incl. param-matched controls).

Protocol
--------
- Removing Track A/B/C deletes that entire readout branch; fusion concatenates
  only the remaining track embeddings.
- Param-matched controls replace the removed branch with an MLP of comparable
  parameter count so fusion width matches the full model.
- All other components and hyperparameters are held fixed.
- Early stopping patience = 30 for every config in this suite.
- Equivariant / coord-update ablation is NOT part of this suite.

Usage
-----
  python scripts/run_ablation_suite.py
  python scripts/run_ablation_suite.py --only 00_full,03_wo_track_c
  python scripts/run_ablation_suite.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = PROJECT_ROOT / "configs" / "ablation"
TRAIN_PY = PROJECT_ROOT / "src" / "train.py"

DEFAULT_ORDER = [
    "00_full",
    "01_wo_track_a",
    "02_wo_track_b",
    "03_wo_track_c",
    "04_wo_track_a_param_match",
    "05_wo_track_b_param_match",
    "06_wo_track_c_param_match",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated config stems to run (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training.",
    )
    parser.add_argument(
        "--python",
        default="",
        help="Python interpreter used to launch train.py "
        "(default: boltz-egnn env if present, else sys.executable).",
    )
    args = parser.parse_args()
    if not args.python:
        candidates = [
            Path("/home/dldx/miniconda3/envs/boltz-egnn/bin/python"),
            Path.home() / "miniconda3" / "envs" / "boltz-egnn" / "bin" / "python",
            Path(sys.executable),
        ]
        args.python = str(next(p for p in candidates if p.exists()))

    stems = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only
        else list(DEFAULT_ORDER)
    )

    for stem in stems:
        cfg = ABLATION_DIR / f"{stem}.yaml"
        if not cfg.exists():
            raise FileNotFoundError(cfg)
        cmd = [args.python, str(TRAIN_PY), "--config", str(cfg)]
        print("=" * 72)
        print("[RUN]", " ".join(cmd))
        print("=" * 72)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    print("\n[DONE] Ablation suite finished. Build Table SX with:")
    print("  python scripts/make_ablation_table.py")


if __name__ == "__main__":
    main()
