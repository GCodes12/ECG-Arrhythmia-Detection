"""Export the small example inputs shipped with the demo, with full provenance.

1. Scalogram PNGs copied (unchanged pixels) from the V4 balanced dataset
   ``v4.2_balanced.h5``. Two per class are drawn with ``numpy.random.default_rng(42)``
   (not selected by model correctness). Every image is traced to its exact source
   record by finding the identical image in the original V4 batch files that carry
   a ``patient_id`` dataset. A draw that cannot be traced (its batch is unavailable
   or lacks ``patient_id``) is replaced by a draw from ``default_rng(43)``, so the
   other classes' draws are unchanged.
2. Single-beat CSVs cut from MIT-BIH Arrhythmia Database records with exactly the
   V4 window (annotated R peak +/- 150 samples at 360 Hz).

Requires the original, non-distributed data files:

    python scripts/export_examples.py \
        --balanced-h5 /path/to/v4.2_balanced.h5 \
        --batch-dir   /path/to/datasets/v4_database \
        --mitbih-dir  /path/to/datasets/raw/MIT-BIH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import wfdb
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CODES = ("N", "S", "V", "F", "Q")
SF_DATALESS = 0x40000000  # macOS flag for cloud-only (not downloaded) files

DATABASES = {
    "mitdb": {
        "name": "MIT-BIH Arrhythmia Database",
        "version": "1.0.0",
        "doi": "https://doi.org/10.13026/C2F305",
        "license": "ODC-By-1.0",
    },
    "stdb": {
        "name": "MIT-BIH ST Change Database",
        "version": "1.0.0",
        "doi": "https://doi.org/10.13026/C2ZW2H",
        "license": "ODC-By-1.0",
    },
}

# One MIT-BIH record and annotation symbol per AAMI class (first matching beat).
BEAT_SOURCES = {
    "N": ("100", "N"),
    "S": ("232", "A"),
    "V": ("119", "V"),
    "F": ("208", "F"),
    "Q": ("107", "/"),
}


def database_of(record: str) -> str:
    # MIT-BIH Arrhythmia records are 100-234; records 300-327 are MIT-BIH ST Change
    # Database records (stored in the original project's folder named "INCART";
    # identity verified by byte-identical .hea headers, see docs/DATASETS.md).
    return "stdb" if 300 <= int(record) <= 327 else "mitdb"


def _digest(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def build_batch_index(batch_dir: Path) -> dict[str, dict]:
    """Map image hash -> provenance for every locally available batch with patient_id."""
    index: dict[str, dict] = {}
    for fn in sorted(os.listdir(batch_dir)):
        if not re.fullmatch(r"(train|val)_v4_batch_\d+\.h5", fn):
            continue
        path = batch_dir / fn
        if os.stat(path).st_flags & SF_DATALESS:
            continue  # never trigger a cloud download
        with h5py.File(path, "r") as h:
            if "patient_id" not in h:
                continue
            images = h["images"][:]
            labels = [x.decode() for x in h["labels"][:]]
            pids = [x.decode() for x in h["patient_id"][:]]
        for pos, (img, sym, pid) in enumerate(zip(images, labels, pids)):
            index.setdefault(
                _digest(img), {"record": pid, "annotation_symbol": sym, "batch_file": fn, "batch_position": pos}
            )
    return index


def export_scalograms(h5_path: Path, batch_dir: Path, out_dir: Path, per_class: int = 2) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    index = build_batch_index(batch_dir)
    rng, fallback = np.random.default_rng(42), np.random.default_rng(43)
    manifest = []
    with h5py.File(h5_path, "r") as h:
        labels = h["labels"][:]
        draws = {cls: list(rng.choice(np.flatnonzero(labels == cls), per_class, replace=False)) for cls in range(5)}
        for cls, code in enumerate(CODES):
            pool = np.flatnonzero(labels == cls)
            chosen: list[tuple[int, dict, str]] = []
            for i in draws[cls]:
                prov = index.get(_digest(h["images"][int(i)]))
                if prov:
                    chosen.append((int(i), prov, "default_rng(42)"))
            used = {int(i) for i in draws[cls]}
            while len(chosen) < per_class:
                i = int(fallback.choice(pool))
                if i in used:
                    continue
                used.add(i)
                prov = index.get(_digest(h["images"][i]))
                if prov:
                    chosen.append((i, prov, "default_rng(43) replacement"))
            for i, prov, how in sorted(chosen):
                name = f"{code}_idx{i}.png"
                Image.fromarray(h["images"][i], mode="L").save(out_dir / name)
                db = database_of(prov["record"])
                manifest.append(
                    {
                        "file": name,
                        "label": code,
                        "balanced_h5_index": i,
                        "selection": how,
                        "source_database": db,
                        "record": prov["record"],
                        "annotation_symbol": prov["annotation_symbol"],
                        "batch_file": prov["batch_file"],
                        "batch_position": prov["batch_position"],
                        "license": DATABASES[db]["license"],
                    }
                )
    return manifest


def export_beats(mitbih_dir: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for code, (rec, symbol) in BEAT_SOURCES.items():
        path = str(mitbih_dir / rec)
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        assert record.fs == 360, record.fs
        signal = record.p_signal[:, 0]
        for i in range(1, len(ann.sample) - 1):
            if ann.symbol[i] != symbol:
                continue
            start, end = ann.sample[i] - 150, ann.sample[i] + 150
            if start < 0 or end > len(signal):
                continue
            beat = signal[start:end]
            name = f"{code}_mitbih_{rec}_sample{int(ann.sample[i])}.csv"
            with open(out_dir / name, "w") as f:
                f.write(f"ecg_{record.units[0]}\n")
                f.writelines(f"{v:.6f}\n" for v in beat)
            manifest.append(
                {
                    "file": name,
                    "label": code,
                    "annotation_symbol": symbol,
                    "source_database": "mitdb",
                    "record": rec,
                    "lead": record.sig_name[0],
                    "r_peak_sample": int(ann.sample[i]),
                    "fs_hz": 360,
                    "n_samples": 300,
                    "license": DATABASES["mitdb"]["license"],
                }
            )
            break
    return manifest


def export_colour_example(scalogram_manifest: list[dict], out_path: Path) -> dict:
    """Jet-colormap rendering of the first N example scalogram.

    Used only to demonstrate that colour images are rejected by the app; it is
    never a valid model input.
    """
    import cv2

    src = next(e for e in scalogram_manifest if e["label"] == "N")
    gray = np.asarray(Image.open(ROOT / "examples" / "scalograms" / src["file"]))
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mode="RGB").save(out_path)
    return {
        "file": str(out_path.relative_to(ROOT)),
        "derived_from": f"examples/scalograms/{src['file']}",
        "source_database": src["source_database"],
        "record": src["record"],
        "purpose": "rejected-input demonstration (colour image)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--balanced-h5", type=Path, required=True)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--mitbih-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = {
        "databases": DATABASES,
        "license_uri": "https://opendatacommons.org/licenses/by/1-0/",
        "scalograms": export_scalograms(args.balanced_h5, args.batch_dir, ROOT / "examples" / "scalograms"),
        "beats": export_beats(args.mitbih_dir, ROOT / "examples" / "beats"),
    }
    manifest["colour_example"] = export_colour_example(
        manifest["scalograms"], ROOT / "docs" / "assets" / "colour_scalogram_example.png"
    )
    (ROOT / "examples" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
