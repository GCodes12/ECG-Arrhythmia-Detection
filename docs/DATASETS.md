# Datasets and provenance

None of the datasets below are distributed in this repository. This document records what the V4 model was
built from, as established by inspecting the original files.

## Source recordings

| Folder in the original project | Records | What the headers show |
|---|---|---|
| `datasets/raw/MIT-BIH/` | 48 (100–234) | MIT-BIH Arrhythmia Database: 360 Hz, 2 leads, 30 min, format 212. The pipeline uses channel 0, which is MLII except in records 102 and 104 (V5) |
| `datasets/raw/INCART/` | 27 (301–327) | **MIT-BIH ST Change Database** (`stdb`): 360 Hz, 1 or 2 channels named `ECG`, 13–67 min, format 212 |

**About the "INCART" folder.** The original code names this folder INCART, but it contains **MIT-BIH ST Change
Database** records. Verification: all 27 local `.hea` files are **byte-identical** to the official `stdb` v1.0.0 headers
(<https://physionet.org/files/stdb/1.0.0/>), including the per-signal checksum fields, which are computed from the
signal data. The official database has 28 records (300–327); record 300 is not in the project. The records do not
match the St Petersburg INCART 12-lead Arrhythmia Database (75 records I01–I75, 12 leads, 257 Hz, 30 minutes). True
INCART records were downloaded by an earlier project version but were **not** used by V4.

(Header identity was verified. The local `.dat` signal files were not compared byte-for-byte with the official files,
but the header checksums are computed from them.)

Official sources:
- MIT-BIH Arrhythmia Database: <https://physionet.org/content/mitdb/1.0.0/>
- MIT-BIH ST Change Database: <https://physionet.org/content/stdb/1.0.0/>
- St Petersburg INCART 12-lead Arrhythmia Database: <https://physionet.org/content/incartdb/1.0.0/>

## Generated data (V4)

1. **Batch files** (`v4_database/`, 185 HDF5 files, about 4.2 GB): 150 `train_v4_batch_N.h5` + 35 `val_v4_batch_N.h5`,
   1,000 beats each. Keys: `images` (uint8 256×256), `labels` (raw annotation symbol). 161 files also contain `patient_id`;
   val batches 19–34 do not, because they were written by a resumed run whose code did not save it.
2. **Balanced dataset** (`v4.2_balanced.h5`, 1.12 GB): `images` (39982, 256, 256) uint8, gzip; `labels` (39982,) int32,
   with 0 = N, 1 = S, 2 = V, 3 = F, 4 = Q. Counts: N 20,000 · S 3,594 · V 7,550 · F 802 · Q 8,036. No patient or record IDs.

## Record-level split (recovered)

The preprocessing split records with `train_test_split(sorted(MIT-BIH) + sorted(INCART-folder), test_size=0.2,
random_state=42)`. Recomputing this reproduces the record order found in the `patient_id` values of all 161
batch files that have them.

- **Train (60):** 124 105 213 309 319 221 222 117 231 232 209 107 311 234 210 306 303 121 315 202 327 114 307 103 118
  220 108 325 106 217 321 308 233 326 116 205 223 203 301 201 230 320 310 112 212 323 312 318 219 208 101 305 123 102
  200 122 313 324 115 304
- **Validation (15):** 104 316 111 100 215 314 207 113 *(verified from `patient_id`)*, then 322 317 214 228 302 119 109
  *(inferred: these batches have no `patient_id`; supported by the generator order and by annotation-symbol counts)*.

**This split was not used for the deployed model.** The balanced dataset merged train and validation batches
before training, and training used beat-level random splits.

## AAMI mapping

| Class | Label | MIT-BIH symbols |
|---|---|---|
| N | 0 | N, L, R, e, j |
| S | 1 | A, a, J, S |
| V | 2 | V, E |
| F | 3 | F |
| Q | 4 | /, f, Q |

Other annotation symbols (`+ ~ | " ! x [ ]` …) are not beat classes and were dropped.
