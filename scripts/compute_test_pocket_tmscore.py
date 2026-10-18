#!/usr/bin/env python3
"""Compute max pocket TM-score from each CASF test complex to Tanimoto-split train.

Pipeline per test pocket:
  1) AA-composition + length prefilter -> Top-K train candidates
  2) TM-align on candidates (multiprocess over test set)
  3) Record max TM-score and nearest train PDB

Also reports a fast max pocket sequence-identity proxy from the prefilter pool.

Outputs:
  results/T-split/test_pocket_similarity.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path("/home/dldx/Boltz-EGNN-Affinity")
CURATED = PROJECT_ROOT / "data" / "curated_pdbbind"
TANIMOTO_SPLIT = PROJECT_ROOT / "data" / "similarity" / "PDBbind_split_leakage_removed_c11_purified.json"
MYDATA_SPLIT = PROJECT_ROOT / "data" / "similarity" / "PDBbind_split_mydata.json"
OUT_DIR = PROJECT_ROOT / "results" / "T-split"
TMALIGN = Path("/home/dldx/bin/TMalign")

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H",
}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
TM_RE = re.compile(
    r"TM-score=\s*([0-9.]+)\s*\(if normalized by length of Chain_([12])",
    re.IGNORECASE,
)


def load_ids(path: Path, keys):
    data = json.load(open(path))
    for k in keys:
        if k in data:
            return [str(x).strip().lower() for x in data[k]]
    raise KeyError(f"none of {keys} in {path}")


def pocket_path(cid: str) -> Path:
    return CURATED / cid / f"{cid}_pocket.pdb"


def parse_pocket_ca_seq(pdb_path: Path) -> str:
    """CA-trace one-letter sequence (unique residue id order)."""
    seq = []
    seen = set()
    with open(pdb_path) as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            resname = line[17:20].strip().upper()
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26]
            key = (chain, resseq, icode, resname)
            if key in seen:
                continue
            seen.add(key)
            aa = AA3_TO_1.get(resname)
            if aa:
                seq.append(aa)
    return "".join(seq)


def aa_composition(seq: str) -> np.ndarray:
    vec = np.zeros(20, dtype=np.float64)
    if not seq:
        return vec
    for ch in seq:
        i = AA_ORDER.find(ch)
        if i >= 0:
            vec[i] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec


def seq_identity_proxy(a: str, b: str) -> float:
    """Fast identity proxy: max over ungapped shifts of matching fraction."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    best = 0
    la, lb = len(a), len(b)
    for shift in range(lb - la + 1):
        matches = sum(1 for i in range(la) if a[i] == b[shift + i])
        best = max(best, matches)
    # also try reverse length roles already handled; normalize by shorter length
    return best / max(min(la, lb), 1)


def run_tmalign(path_a: Path, path_b: Path, tmalign: str, timeout: float = 30.0):
    try:
        proc = subprocess.run(
            [tmalign, str(path_a), str(path_b)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return np.nan, np.nan
    tm1 = tm2 = np.nan
    for m in TM_RE.finditer(proc.stdout):
        score = float(m.group(1))
        if m.group(2) == "1":
            tm1 = score
        else:
            tm2 = score
    return tm1, tm2


WORKER_STATE: dict | None = None


def init_worker(state: dict):
    global WORKER_STATE
    WORKER_STATE = state


def process_one_test(test_cid: str) -> dict:
    state = WORKER_STATE
    assert state is not None
    train_ids = state["train_ids"]
    train_comp = state["train_comp"]  # (N, 20)
    train_lens = state["train_lens"]
    train_seqs = state["train_seqs"]
    top_k = state["top_k"]
    tmalign = state["tmalign"]
    prefilter_pool = state["prefilter_pool"]

    test_pdb = pocket_path(test_cid)
    test_seq = parse_pocket_ca_seq(test_pdb)
    test_vec = aa_composition(test_seq)
    tlen = max(len(test_seq), 1)

    # composition cosine
    norms = np.linalg.norm(train_comp, axis=1) * (np.linalg.norm(test_vec) + 1e-12)
    cos = (train_comp @ test_vec) / np.maximum(norms, 1e-12)

    # length ratio penalty
    ratio = np.minimum(train_lens, tlen) / np.maximum(train_lens, tlen)
    score = cos * ratio

    # exclude self if present
    order = np.argsort(-score)
    candidates = []
    for idx in order:
        cid = train_ids[idx]
        if cid == test_cid:
            continue
        candidates.append(int(idx))
        if len(candidates) >= prefilter_pool:
            break

    # sequence-identity proxy on prefilter pool
    idents = []
    for idx in candidates:
        idents.append(seq_identity_proxy(test_seq, train_seqs[idx]))
    if idents:
        best_ident_i = int(np.argmax(idents))
        max_identity = float(idents[best_ident_i])
        nearest_ident_pdb = train_ids[candidates[best_ident_i]]
    else:
        max_identity = float("nan")
        nearest_ident_pdb = ""

    # re-rank by identity within pool, take top_k for TM-align
    cand_order = np.argsort(-np.asarray(idents, dtype=np.float64))[:top_k]
    tm_candidates = [candidates[i] for i in cand_order]

    best_tm = -1.0
    best_tm1 = best_tm2 = float("nan")
    nearest_tm = ""
    n_tm = 0
    for idx in tm_candidates:
        cid = train_ids[idx]
        tm1, tm2 = run_tmalign(test_pdb, pocket_path(cid), tmalign)
        if not np.isfinite(tm1) and not np.isfinite(tm2):
            continue
        n_tm += 1
        vals = [v for v in (tm1, tm2) if np.isfinite(v)]
        tm = max(vals) if vals else float("nan")
        if np.isfinite(tm) and tm > best_tm:
            best_tm = float(tm)
            best_tm1 = float(tm1) if np.isfinite(tm1) else float("nan")
            best_tm2 = float(tm2) if np.isfinite(tm2) else float("nan")
            nearest_tm = cid

    return {
        "pdb_id": test_cid,
        "max_pocket_tmscore": best_tm if best_tm >= 0 else float("nan"),
        "tmscore_norm_by_test": best_tm1,
        "tmscore_norm_by_train": best_tm2,
        "nearest_train_pdb_pocket": nearest_tm,
        "max_pocket_identity": max_identity,
        "nearest_train_pdb_identity": nearest_ident_pdb,
        "test_pocket_length": len(test_seq),
        "n_tmalign": n_tm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmalign", default=str(TMALIGN))
    ap.add_argument("--out_csv", default=str(OUT_DIR / "test_pocket_similarity.csv"))
    ap.add_argument("--top_k", type=int, default=100, help="TM-align candidates per test")
    ap.add_argument(
        "--prefilter_pool",
        type=int,
        default=500,
        help="Composition prefilter pool before identity re-rank",
    )
    ap.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="If >0, randomly sample this many train pockets (0=all Tanimoto train)",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--train_split",
        default=str(TANIMOTO_SPLIT),
        help="Must be Tanimoto purified split JSON",
    )
    ap.add_argument("--test_split", default=str(MYDATA_SPLIT))
    args = ap.parse_args()

    if not Path(args.tmalign).is_file():
        raise FileNotFoundError(f"TMalign not found: {args.tmalign}")

    test_ids = load_ids(Path(args.test_split), ["casf2016", "test", "casf2016_indep"])
    train_ids = load_ids(Path(args.train_split), ["train"])
    train_ids = [c for c in train_ids if pocket_path(c).is_file()]
    test_ids = [c for c in test_ids if pocket_path(c).is_file()]

    if args.max_train and args.max_train < len(train_ids):
        rng = np.random.default_rng(args.seed)
        train_ids = sorted(rng.choice(train_ids, size=args.max_train, replace=False).tolist())
        print(f"[TRAIN] sampled {len(train_ids)} pockets from Tanimoto train")
    else:
        print(f"[TRAIN] using full Tanimoto train pockets: {len(train_ids)}")
    print(f"[TEST] {len(test_ids)} pockets | workers={args.workers} top_k={args.top_k}")

    print("[CACHE] parsing train pocket sequences...")
    train_seqs = []
    train_comp = []
    train_lens = []
    keep_train = []
    for cid in tqdm(train_ids, desc="train pockets"):
        seq = parse_pocket_ca_seq(pocket_path(cid))
        if len(seq) < 5:
            continue
        keep_train.append(cid)
        train_seqs.append(seq)
        train_comp.append(aa_composition(seq))
        train_lens.append(len(seq))
    train_ids = keep_train
    train_comp = np.asarray(train_comp, dtype=np.float64)
    train_lens = np.asarray(train_lens, dtype=np.float64)
    print(f"[CACHE] usable train pockets: {len(train_ids)}")

    state = {
        "train_ids": train_ids,
        "train_comp": train_comp,
        "train_lens": train_lens,
        "train_seqs": train_seqs,
        "top_k": int(args.top_k),
        "prefilter_pool": int(args.prefilter_pool),
        "tmalign": str(args.tmalign),
    }

    rows = []
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=init_worker,
        initargs=(state,),
    ) as ex:
        futs = {ex.submit(process_one_test, cid): cid for cid in test_ids}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="TM-align tests"):
            rows.append(fut.result())
            if len(rows) % 25 == 0:
                pd.DataFrame(rows).sort_values("pdb_id").to_csv(out_csv, index=False)

    df = pd.DataFrame(rows).sort_values("pdb_id").reset_index(drop=True)
    # exact columns requested + extras
    cols = [
        "pdb_id",
        "max_pocket_tmscore",
        "nearest_train_pdb_pocket",
        "max_pocket_identity",
        "nearest_train_pdb_identity",
        "tmscore_norm_by_test",
        "tmscore_norm_by_train",
        "test_pocket_length",
        "n_tmalign",
    ]
    df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f"[DONE] {out_csv}  n={len(df)}")
    print(
        df[["max_pocket_tmscore", "max_pocket_identity"]]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()
