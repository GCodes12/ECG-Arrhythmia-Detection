"""Verify that app/inference.py reproduces the original V4 scalograms exactly.

Regenerates beats from the raw WFDB records with the same loop as
v4_Preprocessing.ipynb (cell 21) and compares every regenerated scalogram and
annotation symbol with the images stored in the original V4 batch files
(train_v4_batch_N.h5 / val_v4_batch_N.h5 that contain a ``patient_id`` dataset).

This script only reads data. It requires the original (non-distributed) files:

    python scripts/verify_preprocessing_parity.py \
        --raw-dir /path/to/datasets/raw \
        --batch-dir /path/to/datasets/v4_database \
        --split train --batches 0-8

Raw record folders expected inside --raw-dir: MIT-BIH/ and INCART/
(the folder the original code called "INCART"; see docs/DATASETS.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.inference import beat_to_scalogram  # noqa: E402


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def record_beats(raw_dir: Path, rec: str):
    """Yield (symbol, beat) exactly as the cell-21 generator does."""
    folder = "INCART" if rec.startswith("3") else "MIT-BIH"
    path = raw_dir / folder / rec
    record = wfdb.rdrecord(str(path))
    ann = wfdb.rdann(str(path), "atr")
    signal = record.p_signal[:, 0]
    for i in range(1, len(ann.sample) - 1):
        start, end = ann.sample[i] - 150, ann.sample[i] + 150
        if start < 0 or end > len(signal):
            continue
        yield ann.symbol[i], signal[start:end]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--batches", default="0-8", help="contiguous batch range starting at 0, e.g. 0-8")
    args = ap.parse_args()

    batch_ids = parse_range(args.batches)
    if batch_ids != list(range(batch_ids[0], batch_ids[-1] + 1)) or batch_ids[0] != 0:
        print("Batches must be a contiguous range starting at 0 (beats are aligned sequentially).")
        return 2

    generators: dict[str, object] = {}
    total = exact = label_ok = 0
    max_abs_diff = 0
    for b in batch_ids:
        fn = args.batch_dir / f"{args.split}_v4_batch_{b}.h5"
        with h5py.File(fn, "r") as h:
            if "patient_id" not in h:
                print(f"{fn.name}: no patient_id dataset; cannot align, stopping.")
                break
            images = h["images"][:]
            labels = [x.decode() for x in h["labels"][:]]
            pids = [x.decode() for x in h["patient_id"][:]]
        batch_exact = 0
        for img, lab, pid in zip(images, labels, pids):
            if pid not in generators:
                generators[pid] = record_beats(args.raw_dir, pid)
            sym, beat = next(generators[pid])
            regen = beat_to_scalogram(beat)
            diff = int(np.abs(regen.astype(np.int16) - img.astype(np.int16)).max())
            max_abs_diff = max(max_abs_diff, diff)
            total += 1
            exact += diff == 0
            batch_exact += diff == 0
            label_ok += sym == lab
        print(f"{fn.name}: records {sorted(set(pids))}  pixel-exact {batch_exact}/{len(images)}")

    print("-" * 70)
    print(f"beats compared        : {total}")
    print(f"pixel-exact matches   : {exact}  ({exact / max(total, 1):.4%})")
    print(f"annotation symbol match: {label_ok}  ({label_ok / max(total, 1):.4%})")
    print(f"max |pixel difference|: {max_abs_diff}")
    return 0 if total and exact == total and label_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
