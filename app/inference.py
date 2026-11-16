"""Inference utilities for the V4 ECG CWT-CNN demo.

This module contains everything the Gradio app needs that is not UI code:
model loading, input validation, the V4 preprocessing, and prediction.

The preprocessing reproduces the original V4 dataset-generation code
(``v4_Preprocessing.ipynb``, cell 21) line for line:

    start, end = ann.sample[i] - 150, ann.sample[i] + 150
    scales = np.arange(1, 129)
    coef, _ = pywt.cwt(signal[start:end], scales, 'mexh')
    rescaled = cv2.resize(np.abs(coef), (256, 256))
    norm_img = cv2.normalize(rescaled, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

and the model input conversion used during training (``v4_2_Model.ipynb``,
cells 23/25):

    x = np.repeat(x[..., np.newaxis], 3, axis=-1) / 255.0

Nothing here changes the trained model or the preprocessing methodology.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pywt
from PIL import Image

# --------------------------------------------------------------------------
# Constants taken from the original V4 code
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "v4_custom_fast_mode.keras"
MODEL_PATH = Path(os.environ.get("ECG_MODEL_PATH", DEFAULT_MODEL_PATH))

# Label order 0..4 as encoded in v4_2_Model.ipynb (cell 19 AAMI mapping).
CLASS_CODES: tuple[str, ...] = ("N", "S", "V", "F", "Q")
CLASS_NAMES: tuple[str, ...] = (
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)",
)

BEAT_LENGTH = 300  # samples: R-peak annotation +/- 150 samples
SAMPLING_RATE_HZ = 360  # all records used by the V4 pipeline are 360 Hz
CWT_SCALES = np.arange(1, 129)  # scales 1..128
CWT_WAVELET = "mexh"  # Mexican hat
IMAGE_SIZE = 256  # model input is 256 x 256 (x 3 identical channels)

MAX_CSV_BYTES = 200_000
GRAYSCALE_TOLERANCE = 2  # max per-pixel channel difference still treated as grayscale
R_PEAK_TOLERANCE = 30  # samples; heuristic check only (produces a warning)


class InputValidationError(ValueError):
    """Raised for user input that the model cannot meaningfully process."""


class ModelUnavailableError(RuntimeError):
    """Raised when the model checkpoint cannot be loaded."""


@dataclass
class Prediction:
    probabilities: dict[str, float]
    top_class: str
    top_probability: float
    scalogram: np.ndarray  # uint8 (256, 256): the exact image the model saw
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    """Load the trained V4 checkpoint once (read-only, no recompilation)."""
    model_path = Path(path) if path else MODEL_PATH
    if not model_path.is_file():
        raise ModelUnavailableError(
            f"Model file not found: {model_path.name}. "
            "Place v4_custom_fast_mode.keras in the models/ directory."
        )
    try:
        import keras  # imported lazily so the UI module can start quickly

        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ModelUnavailableError(
            "The model could not be loaded. It was saved with Keras 3.13.2; "
            "install the pinned versions from requirements.txt. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    expected_in = (None, IMAGE_SIZE, IMAGE_SIZE, 3)
    expected_out = (None, len(CLASS_CODES))
    if tuple(model.input_shape) != expected_in or tuple(model.output_shape) != expected_out:
        raise ModelUnavailableError(
            f"Unexpected model signature {model.input_shape} -> {model.output_shape}; "
            f"expected {expected_in} -> {expected_out}."
        )
    return model


def predict_probabilities(model_input: np.ndarray, model=None) -> np.ndarray:
    """Run the trained model on a (1, 256, 256, 3) float32 array in [0, 1]."""
    model = model or load_model()
    probs = np.asarray(model.predict(model_input, verbose=0)[0], dtype=np.float64)
    return probs


# --------------------------------------------------------------------------
# V4 preprocessing (reproduced from the original notebooks)
# --------------------------------------------------------------------------
def beat_to_scalogram(beat: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert one 300-sample, 360 Hz ECG beat to the V4 uint8 scalogram.

    Identical operations to v4_Preprocessing.ipynb cell 21.
    """
    signal = np.asarray(beat, dtype=np.float64)
    coef, _ = pywt.cwt(signal, CWT_SCALES, CWT_WAVELET)
    rescaled = cv2.resize(np.abs(coef), (IMAGE_SIZE, IMAGE_SIZE))
    norm_img = cv2.normalize(rescaled, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    return norm_img


def scalogram_to_model_input(scalogram: np.ndarray) -> np.ndarray:
    """Grayscale uint8 (256, 256) -> float32 (1, 256, 256, 3) in [0, 1].

    Identical to the training generator in v4_2_Model.ipynb (cells 23/25).
    """
    if scalogram.shape != (IMAGE_SIZE, IMAGE_SIZE) or scalogram.dtype != np.uint8:
        raise InputValidationError(
            f"Internal error: expected uint8 {IMAGE_SIZE}x{IMAGE_SIZE} scalogram, "
            f"got {scalogram.dtype} {scalogram.shape}."
        )
    x = np.repeat(scalogram[..., np.newaxis], 3, axis=-1) / 255.0
    return x[np.newaxis].astype(np.float32)


# --------------------------------------------------------------------------
# Input parsing / validation
# --------------------------------------------------------------------------
def image_to_scalogram(image: Image.Image) -> tuple[np.ndarray, list[str]]:
    """Validate an uploaded scalogram image and return it as uint8 grayscale.

    The model was trained only on grayscale |CWT| scalograms. Colour images
    (for example jet-colormap plots) are rejected instead of being silently
    converted, because predictions on them would not be meaningful.
    """
    if image is None:
        raise InputValidationError("Please upload a scalogram image.")
    warnings: list[str] = []

    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = np.asarray(image.convert("RGBA"))
        if rgba[..., 3].min() < 255:
            raise InputValidationError(
                "The image contains transparent pixels. Upload an opaque grayscale scalogram."
            )

    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    channel_spread = int((rgb.max(axis=-1) - rgb.min(axis=-1)).max())
    if channel_spread > GRAYSCALE_TOLERANCE:
        raise InputValidationError(
            "This looks like a colour image. The V4 model was trained only on "
            "grayscale |CWT| scalograms (see the examples). Colour plots such as "
            "jet-colormap scalograms are not a supported input."
        )

    width, height = image.size
    if min(width, height) < 64:
        raise InputValidationError(
            f"Image is too small ({width}x{height}). Expected a 256x256 scalogram."
        )
    gray = image.convert("L")
    if (width, height) != (IMAGE_SIZE, IMAGE_SIZE):
        if width != height:
            raise InputValidationError(
                f"Image is not square ({width}x{height}). Expected a 256x256 scalogram."
            )
        # Matches the original app behaviour (PIL resize), but flagged because
        # all training images were exactly 256x256.
        gray = gray.resize((IMAGE_SIZE, IMAGE_SIZE))
        warnings.append(
            f"Image was resized from {width}x{height} to 256x256. Training images were "
            "exactly 256x256, so resized inputs may not match the training distribution."
        )

    scalogram = np.asarray(gray, dtype=np.uint8)
    if int(scalogram.max()) - int(scalogram.min()) == 0:
        raise InputValidationError("The image is blank (a single uniform value).")
    if scalogram.min() > 10 or scalogram.max() < 245:
        warnings.append(
            "Pixel range is "
            f"{scalogram.min()}-{scalogram.max()}. V4 scalograms are min-max normalised to "
            "0-255, so this image may not have been produced by the V4 pipeline."
        )
    return scalogram, warnings


def parse_beat_csv(data: bytes | str, sampling_rate_hz: float | None) -> tuple[np.ndarray, list[str]]:
    """Parse and validate a single-beat CSV.

    Accepted format: exactly 300 numeric values (one column or one row),
    optionally preceded by a single non-numeric header line. Values are the
    ECG amplitude (e.g. mV) sampled at 360 Hz, centred on the R peak.
    Signals are never resampled.
    """
    if sampling_rate_hz is None:
        raise InputValidationError("Please state the sampling frequency of the beat (must be 360 Hz).")
    try:
        fs = float(sampling_rate_hz)
    except (TypeError, ValueError):
        raise InputValidationError("Sampling frequency must be a number.") from None
    if not np.isfinite(fs) or abs(fs - SAMPLING_RATE_HZ) > 1e-9:
        raise InputValidationError(
            f"Sampling frequency must be exactly {SAMPLING_RATE_HZ} Hz (got {fs:g} Hz). "
            "Signals are not resampled; please provide a 360 Hz beat."
        )

    if isinstance(data, bytes):
        if len(data) > MAX_CSV_BYTES:
            raise InputValidationError("CSV file is too large for a single 300-sample beat.")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise InputValidationError("CSV must be a UTF-8 text file.") from None
    else:
        text = data

    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise InputValidationError("The CSV file is empty.")

    def split(line: str) -> list[str]:
        for sep in (",", ";", "\t"):
            if sep in line:
                return [tok.strip() for tok in line.split(sep)]
        return line.split()

    rows = [split(ln) for ln in lines]

    def is_number(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    if rows and not all(is_number(t) for t in rows[0] if t != ""):
        rows = rows[1:]  # one header line is allowed
        if not rows:
            raise InputValidationError("The CSV file contains a header but no values.")

    widths = {len(r) for r in rows}
    if len(rows) > 1 and widths != {1}:
        raise InputValidationError(
            "CSV must contain a single column (one value per line) or a single row; "
            f"found {len(rows)} rows with {sorted(widths)} columns."
        )
    tokens = [t for r in rows for t in r]

    values = []
    for pos, tok in enumerate(tokens, start=1):
        if tok == "" or tok.lower() in {"nan", "na", "null", "none"}:
            raise InputValidationError(f"Missing value at position {pos}.")
        try:
            val = float(tok)
        except ValueError:
            raise InputValidationError(f"Non-numeric value {tok!r} at position {pos}.") from None
        if not np.isfinite(val):
            raise InputValidationError(f"Invalid (non-finite) value at position {pos}.")
        values.append(val)

    if len(values) != BEAT_LENGTH:
        raise InputValidationError(
            f"Expected exactly {BEAT_LENGTH} samples (R peak +/- 150 samples at 360 Hz); "
            f"got {len(values)}. Signals are not padded, trimmed or resampled."
        )
    beat = np.asarray(values, dtype=np.float64)
    if np.ptp(beat) == 0:
        raise InputValidationError("The beat is a constant signal; there is nothing to analyse.")

    warnings: list[str] = []
    peak = int(np.argmax(np.abs(beat - np.median(beat))))
    if abs(peak - BEAT_LENGTH // 2) > R_PEAK_TOLERANCE:
        warnings.append(
            f"The largest deflection is at sample {peak}, not near the centre (sample 150). "
            "V4 training beats were centred on the annotated R peak; an off-centre beat "
            "may not match the training distribution. (Heuristic check only.)"
        )
    return beat, warnings


# --------------------------------------------------------------------------
# End-to-end helpers used by the UI
# --------------------------------------------------------------------------
def _build_prediction(scalogram: np.ndarray, warnings: list[str], model=None) -> Prediction:
    probs = predict_probabilities(scalogram_to_model_input(scalogram), model=model)
    probabilities = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    top = int(np.argmax(probs))
    return Prediction(
        probabilities=probabilities,
        top_class=CLASS_NAMES[top],
        top_probability=float(probs[top]),
        scalogram=scalogram,
        warnings=warnings,
    )


def predict_from_image(image: Image.Image, model=None) -> Prediction:
    scalogram, warnings = image_to_scalogram(image)
    return _build_prediction(scalogram, warnings, model=model)


def predict_from_beat_csv(data: bytes | str, sampling_rate_hz: float | None, model=None) -> tuple[Prediction, np.ndarray]:
    beat, warnings = parse_beat_csv(data, sampling_rate_hz)
    scalogram = beat_to_scalogram(beat)
    return _build_prediction(scalogram, warnings, model=model), beat


def read_image_bytes(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        raise InputValidationError("The file could not be read as an image.") from None
