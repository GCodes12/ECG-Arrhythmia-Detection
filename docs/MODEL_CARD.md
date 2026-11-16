# Model card: V4 "fast mode" CWT-CNN

| | |
|---|---|
| File | `models/v4_custom_fast_mode.keras` (1,608,778 bytes, SHA-256 `a45d4a21946b30fc…`) |
| Origin | Best checkpoint of `v4_2_Model.ipynb` cell 25 (epoch 19, saved 2026-04-16 14:28:30). Byte-identical to the original `godzilla_v4_fast_mode.keras` |
| Framework | Keras 3.13.2 on TensorFlow (trained with TF 2.20.0, mixed precision `mixed_float16`) |
| Task | Single-beat classification into 5 AAMI classes: N, S, V, F, Q |
| Input | (256, 256, 3) float32 in [0, 1]: a grayscale CWT scalogram repeated on 3 channels |
| Output | Softmax over (N, S, V, F, Q) |
| Parameters | 129,477 (129,285 trainable) |

## Architecture

Conv2D(32, 5×5, stride 2, same, ReLU) → MaxPool(2) → BatchNorm → Conv2D(64, 3×3, stride 2, same, ReLU) → MaxPool(2) →
BatchNorm → Conv2D(128, 3×3, same, ReLU) → GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.5) → Dense(5, softmax).

## Input pipeline (must be matched exactly)

300-sample beat at 360 Hz centred on the annotated R peak → `pywt.cwt(beat, scales=1..128, 'mexh')` → `abs` →
`cv2.resize(…, (256, 256))` → `cv2.normalize(…, 0, 255, NORM_MINMAX).astype('uint8')` → repeat to 3 channels → ÷255.
No filtering is applied. See `app/inference.py`.

## Training data and procedure

39,982 beats (`docs/DATASETS.md`). Adam (lr 1e-4 with ReduceLROnPlateau), categorical cross-entropy, class weights
`{N: 1, S: 12, V: 5, F: 5, Q: 5}`, batch size 128. Training used **beat-level random 80/20 splits that were reshuffled
without a seed** in each Colab session.

## Evaluation

| Metric (historical, from the original notebook) | Value |
|---|---|
| Accuracy on a seed-42 subset of 7,997 beats | 94.64% |
| Macro F1 | 0.87 |
| F1 per class: N / S / V / F / Q | 0.96 / 0.83 / 0.95 / 0.60 / 0.99 |

**These numbers are not independently validated and do not measure generalisation:**
- The subset is drawn from the same 39,982-beat pool, and most of it was probably used for training (unseeded,
  changing training splits).
- Beats from the same patient/record are on both sides of the split.
- No record-wise or external evaluation of this model exists.

## Intended use and limitations

- Educational and portfolio demonstration of a CWT + CNN pipeline. **Not a medical device; not for clinical use.**
- Works only on pre-segmented, R-peak-centred 360 Hz single beats (or their V4 scalograms). It does not detect beats,
  check signal quality, or analyse rhythm.
- F (fusion) is the weakest class in the historical results (F1 0.60).
- Softmax outputs are not calibrated probabilities.
