# Example inputs

All example files were produced by `scripts/export_examples.py`. The provenance of every file (source database,
record, annotation symbol and, for scalograms, the original batch file and position) is in `manifest.json`.

**Contains information from the MIT-BIH Arrhythmia Database, which is made available under the ODC Attribution
License** (<https://opendatacommons.org/licenses/by/1-0/>). Source: Moody GB, Mark RG, *MIT-BIH Arrhythmia Database*
(version 1.0.0), PhysioNet, <https://doi.org/10.13026/C2F305>. See `../NOTICE.md` for full citations.

## `scalograms/` (10 PNG, 256×256, 8-bit grayscale)

- Pixel-exact copies of images from the V4 balanced dataset `v4.2_balanced.h5` (index in the file name).
- Two per class, drawn with `numpy.random.default_rng(42)`. They were **not** selected by whether the model classifies
  them correctly.
- Each image was traced to its exact source record by finding the identical image in the original V4 batch files
  that store `patient_id`. Two draws (`V_idx35068`, `Q_idx34987`) could not be traced, because their batch files were not
  available locally, so they were replaced by draws from `default_rng(43)` (`V_idx31028`, `Q_idx34315`).
  All 10 come from MIT-BIH Arrhythmia Database records.
- **These images are part of the data the model was trained on.** Predictions on them show what the model does;
  they are not evidence of generalisation.

## `beats/` (5 CSV, one beat each)

- One header line (`ecg_mV`) followed by 300 values in mV: lead MLII, 360 Hz, annotated R peak ± 150 samples
  (the V4 window). The values are identical to the raw record (verified against the WFDB files).
- Cut from MIT-BIH Arrhythmia Database records 100 (N), 232 (A → S), 119 (V), 208 (F) and 107 (paced `/` → Q). For each,
  the first annotated beat of that type with a full window was taken.
- All five records were part of the V4 training pool.

## Provenance and observed model outputs

Recorded with the shipped checkpoint (TF 2.20.0 / Keras 3.13.2). Nothing was filtered for correctness. These 15
training-pool samples are an illustration, **not** a performance measure.

| File | Source record | Reference label (symbol) | Model top class | Softmax |
|---|---|---|---|---|
| `scalograms/N_idx1875.png` | mitdb 105 | N (`N`) | N | 0.955 |
| `scalograms/N_idx16473.png` | mitdb 221 | N (`N`) | N | 0.928 |
| `scalograms/S_idx22635.png` | mitdb 232 | S (`A`) | S | 0.999 |
| `scalograms/S_idx22655.png` | mitdb 232 | S (`A`) | S | 0.999 |
| `scalograms/V_idx16892.png` | mitdb 221 | V (`V`) | V | 0.983 |
| `scalograms/V_idx31028.png` | mitdb 201 | V (`V`) | V | 0.981 |
| `scalograms/F_idx4602.png` | mitdb 213 | F (`F`) | V ✗ | 0.917 |
| `scalograms/F_idx31215.png` | mitdb 201 | F (`F`) | V ✗ | 0.769 |
| `scalograms/Q_idx34315.png` | mitdb 102 | Q (`/`) | Q | 0.998 |
| `scalograms/Q_idx36419.png` | mitdb 104 | Q (`/`) | Q | 1.000 |
| `beats/N_mitbih_100_sample370.csv` | mitdb 100 | N (`N`) | S ✗ | 0.676 |
| `beats/S_mitbih_232_sample737.csv` | mitdb 232 | S (`A`) | S | 0.999 |
| `beats/V_mitbih_119_sample503.csv` | mitdb 119 | V (`V`) | V | 0.980 |
| `beats/F_mitbih_208_sample697.csv` | mitdb 208 | F (`F`) | V ✗ | 0.696 |
| `beats/Q_mitbih_107_sample272.csv` | mitdb 107 | Q (`/`) | Q | 0.998 |

11 of 15 agree with the reference annotation. All three fusion examples are predicted as V, consistent with F
being the weakest class historically.
