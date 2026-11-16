"""Tests for app/inference.py: model loading, V4 preprocessing, validation, prediction.

Run from the repository root:  python -m pytest -q
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import pywt
from PIL import Image

from app import inference as inf

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
MANIFEST = json.loads((EXAMPLES / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# Reference: the original cell-21 code from v4_Preprocessing.ipynb, verbatim
# ---------------------------------------------------------------------------
def original_cell21_scalogram(segment: np.ndarray) -> np.ndarray:
    scales = np.arange(1, 129)
    coef, _ = pywt.cwt(segment, scales, 'mexh')
    rescaled = cv2.resize(np.abs(coef), (256, 256))
    norm_img = cv2.normalize(rescaled, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return norm_img


def original_generator_input(x: np.ndarray) -> np.ndarray:
    # v4_2_Model.ipynb, GodzillaGenerator.__getitem__ (single image)
    return np.repeat(x[np.newaxis][..., np.newaxis], 3, axis=-1) / 255.0


def csv_bytes(values, header: str | None = "ecg_mV") -> bytes:
    lines = ([header] if header else []) + [f"{v:.6f}" for v in values]
    return ("\n".join(lines) + "\n").encode()


def synthetic_beat(n: int = 300) -> np.ndarray:
    t = np.arange(n)
    return 1.2 * np.exp(-0.5 * ((t - 150) / 4.0) ** 2) - 0.1 + 0.15 * np.exp(-0.5 * ((t - 230) / 18.0) ** 2)


def png(arr: np.ndarray, mode: str) -> Image.Image:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf)


@pytest.fixture(scope="session")
def model():
    return inf.load_model()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_model_loads_with_expected_signature(model):
    assert tuple(model.input_shape) == (None, 256, 256, 3)
    assert tuple(model.output_shape) == (None, 5)
    assert model.count_params() == 129_477


def test_model_path_is_inside_repository():
    assert inf.DEFAULT_MODEL_PATH == ROOT / "models" / "v4_custom_fast_mode.keras"
    assert inf.DEFAULT_MODEL_PATH.is_file()


def test_missing_model_raises_clear_error(tmp_path):
    with pytest.raises(inf.ModelUnavailableError):
        inf.load_model.__wrapped__(str(tmp_path / "nope.keras"))


# ---------------------------------------------------------------------------
# Preprocessing parity with the original notebook code
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_beat_to_scalogram_matches_original_cell21(seed):
    beat = synthetic_beat() + np.random.default_rng(seed).normal(0, 0.02, 300)
    ours = inf.beat_to_scalogram(beat)
    ref = original_cell21_scalogram(beat)
    assert ours.dtype == np.uint8 and ours.shape == (256, 256)
    assert np.array_equal(ours, ref)


@pytest.mark.parametrize("entry", MANIFEST["beats"], ids=lambda e: e["file"])
def test_example_beats_match_original_cell21(entry):
    data = (EXAMPLES / "beats" / entry["file"]).read_bytes()
    beat, _ = inf.parse_beat_csv(data, 360)
    assert np.array_equal(inf.beat_to_scalogram(beat), original_cell21_scalogram(beat))


def test_model_input_matches_training_generator():
    s = inf.beat_to_scalogram(synthetic_beat())
    x = inf.scalogram_to_model_input(s)
    assert x.shape == (1, 256, 256, 3) and x.dtype == np.float32
    np.testing.assert_allclose(x, original_generator_input(s), rtol=0, atol=1e-7)
    assert 0.0 <= x.min() and x.max() <= 1.0


# ---------------------------------------------------------------------------
# Prediction with the real checkpoint
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entry", MANIFEST["scalograms"], ids=lambda e: e["file"])
def test_example_scalograms_are_model_compatible(entry, model):
    img = Image.open(EXAMPLES / "scalograms" / entry["file"])
    assert img.mode == "L" and img.size == (256, 256)
    pred = inf.predict_from_image(img, model=model)
    assert pred.warnings == []
    # pixels reach the model unchanged (PNG is lossless)
    assert np.array_equal(pred.scalogram, np.asarray(img))
    probs = np.array(list(pred.probabilities.values()))
    assert probs.shape == (5,) and np.all(probs >= 0) and abs(probs.sum() - 1) < 1e-4
    assert pred.top_class in inf.CLASS_NAMES


def test_rgb_copy_of_grayscale_is_accepted_identically(model):
    gray = np.asarray(Image.open(EXAMPLES / "scalograms" / MANIFEST["scalograms"][0]["file"]))
    rgb = png(np.repeat(gray[..., None], 3, -1), "RGB")
    a = inf.predict_from_image(rgb, model=model)
    b = inf.predict_from_image(png(gray, "L"), model=model)
    assert np.array_equal(a.scalogram, b.scalogram)
    assert a.probabilities == pytest.approx(b.probabilities)


@pytest.mark.parametrize("entry", MANIFEST["beats"], ids=lambda e: e["file"])
def test_example_beats_predict(entry, model):
    pred, beat = inf.predict_from_beat_csv((EXAMPLES / "beats" / entry["file"]).read_bytes(), 360, model=model)
    assert beat.shape == (300,)
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-4


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------
def test_csv_accepts_single_row_and_no_header():
    beat = synthetic_beat()
    row = (",".join(f"{v:.6f}" for v in beat)).encode()
    parsed, _ = inf.parse_beat_csv(row, 360)
    np.testing.assert_allclose(parsed, np.round(beat, 6))
    parsed2, _ = inf.parse_beat_csv(csv_bytes(beat, header=None), 360.0)
    np.testing.assert_allclose(parsed2, parsed)


@pytest.mark.parametrize(
    "payload, fs, match",
    [
        (csv_bytes(synthetic_beat(299)), 360, "exactly 300"),
        (csv_bytes(synthetic_beat(301)), 360, "exactly 300"),
        (csv_bytes(synthetic_beat()), 250, "360 Hz"),
        (csv_bytes(synthetic_beat()), 720, "360 Hz"),
        (csv_bytes(synthetic_beat()), None, "sampling frequency"),
        (csv_bytes(synthetic_beat()), "abc", "number"),
        (csv_bytes(synthetic_beat()), float("nan"), "360 Hz"),
        (b"", 360, "empty"),
        (b"ecg_mV\n", 360, "header but no values"),
        (b"ecg\n" + b"\n".join([b"0.1"] * 150 + [b"NaN"] + [b"0.2"] * 149), 360, "Missing value"),
        (b",".join([b"0.1"] * 150 + [b""] + [b"0.2"] * 149), 360, "Missing value"),
        (b"ecg\n" + b"\n".join([b"0.1"] * 150 + [b"abc"] + [b"0.2"] * 149), 360, "Non-numeric"),
        (b"ecg\n" + b"\n".join([b"0.1"] * 150 + [b"inf"] + [b"0.2"] * 149), 360, "non-finite"),
        (b"ecg\n" + b"\n".join([b"0.1"] * 150 + [b""] + [b"0.2"] * 149), 360, "exactly 300"),
        (csv_bytes([0.5] * 300), 360, "constant"),
        (b"a,b\n" + b"\n".join([b"0.1,0.2"] * 300), 360, "single column"),
        (b"\xff\xfe\x00\x01", 360, "UTF-8"),
        (b"1\n" * 100_001, 360, "too large"),
    ],
)
def test_csv_rejects_invalid_input(payload, fs, match):
    with pytest.raises(inf.InputValidationError, match=f"(?i){match}"):
        inf.parse_beat_csv(payload, fs)


def test_csv_off_centre_beat_warns_but_is_accepted():
    beat = np.roll(synthetic_beat(), 90)
    _, warnings = inf.parse_beat_csv(csv_bytes(beat), 360)
    assert any("centre" in w for w in warnings)


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------
def test_colour_image_is_rejected():
    jet = ROOT / "docs" / "assets" / "colour_scalogram_example.png"
    with pytest.raises(inf.InputValidationError, match="colour"):
        inf.image_to_scalogram(Image.open(jet))


def test_transparent_image_is_rejected():
    rgba = np.zeros((256, 256, 4), np.uint8)
    rgba[..., :3] = np.linspace(0, 255, 256, dtype=np.uint8)[None, :, None]
    rgba[..., 3] = 255
    rgba[0, 0, 3] = 0
    with pytest.raises(inf.InputValidationError, match="transparent"):
        inf.image_to_scalogram(png(rgba, "RGBA"))


@pytest.mark.parametrize(
    "arr, match",
    [
        (np.full((256, 256), 7, np.uint8), "blank"),
        (np.tile(np.arange(256, dtype=np.uint8), (128, 1)), "not square"),
        (np.tile(np.arange(0, 256, 8, dtype=np.uint8), (32, 1)), "too small"),
    ],
)
def test_bad_grayscale_images_are_rejected(arr, match):
    with pytest.raises(inf.InputValidationError, match=match):
        inf.image_to_scalogram(png(arr, "L"))


def test_none_image_is_rejected():
    with pytest.raises(inf.InputValidationError):
        inf.image_to_scalogram(None)


def test_non_256_square_grayscale_is_resized_with_warning():
    small = np.tile(np.linspace(0, 255, 128).astype(np.uint8), (128, 1))
    scalogram, warnings = inf.image_to_scalogram(png(small, "L"))
    assert scalogram.shape == (256, 256)
    assert any("resized" in w for w in warnings)


def test_unreadable_image_bytes():
    with pytest.raises(inf.InputValidationError):
        inf.read_image_bytes(b"not an image")
