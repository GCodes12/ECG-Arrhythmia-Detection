"""Gradio demo for the V4 ECG arrhythmia classifier (CWT scalogram + CNN).

Run from the repository root:

    python app/app.py

All predictions come from the trained checkpoint in models/. Inference logic
lives in app/inference.py; this file only builds the user interface.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import inference as inf  # noqa: E402

ROOT = inf.PROJECT_ROOT
EXAMPLES_DIR = ROOT / "examples"
CLASS_COLOURS = {"N": "#34d399", "S": "#fbbf24", "V": "#f87171", "F": "#a78bfa", "Q": "#94a3b8"}


# ---------------------------------------------------------------------------
# Examples (shipped in examples/, see examples/README.md for provenance)
# ---------------------------------------------------------------------------
def _load_manifest() -> dict:
    path = EXAMPLES_DIR / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else {"scalograms": [], "beats": []}


MANIFEST = _load_manifest()
CODE_TO_NAME = dict(zip(inf.CLASS_CODES, inf.CLASS_NAMES))
IMAGE_EXAMPLES = [
    str(EXAMPLES_DIR / "scalograms" / e["file"])
    for e in MANIFEST["scalograms"]
    if (EXAMPLES_DIR / "scalograms" / e["file"]).is_file()
]
IMAGE_EXAMPLE_LABELS = [
    f"{CODE_TO_NAME[e['label']]} · dataset idx {e['balanced_h5_index']}" for e in MANIFEST["scalograms"]
]
BEAT_EXAMPLES = [
    [str(EXAMPLES_DIR / "beats" / e["file"]), 360]
    for e in MANIFEST["beats"]
    if (EXAMPLES_DIR / "beats" / e["file"]).is_file()
]
BEAT_EXAMPLE_LABELS = [
    f"{CODE_TO_NAME[e['label']]} · MIT-BIH {e['record']} ('{e['annotation_symbol']}')" for e in MANIFEST["beats"]
]


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _card(body: str, kind: str = "neutral") -> str:
    return f'<div class="result-card {kind}">{body}</div>'


EMPTY_CARD = _card(
    '<div class="rc-eyebrow">Awaiting input</div>'
    '<div class="rc-title">No prediction yet</div>'
    '<p class="rc-text">Upload a grayscale V4 scalogram or a single 300-sample ECG beat, '
    "or pick one of the examples, then press <b>Analyse</b>.</p>"
)


def _error_card(title: str, message: str) -> str:
    return _card(
        f'<div class="rc-eyebrow">Input not accepted</div>'
        f'<div class="rc-title">{html.escape(title)}</div>'
        f'<p class="rc-text">{html.escape(message)}</p>',
        "error",
    )


def _result_card(pred: inf.Prediction, source: str) -> str:
    code = inf.CLASS_CODES[inf.CLASS_NAMES.index(pred.top_class)]
    colour = CLASS_COLOURS[code]
    warn_html = "".join(f"<li>{html.escape(w)}</li>" for w in pred.warnings)
    warn_block = f'<ul class="rc-warnings">{warn_html}</ul>' if warn_html else ""
    return _card(
        f'<div class="rc-eyebrow">Model output · {html.escape(source)}</div>'
        f'<div class="rc-title"><span class="rc-dot" style="background:{colour}"></span>'
        f"{html.escape(pred.top_class)}</div>"
        f'<div class="rc-metric"><span class="rc-value">{pred.top_probability * 100:.1f}%</span>'
        '<span class="rc-unit">softmax probability</span></div>'
        '<p class="rc-text">This is the class with the highest softmax output of the V4 CNN. '
        "Softmax values are not calibrated confidence and are not a diagnosis.</p>"
        f"{warn_block}",
        "ok" if not pred.warnings else "warn",
    )


def _beat_frame(beat: np.ndarray | None) -> pd.DataFrame:
    if beat is None:
        return pd.DataFrame({"sample": [], "amplitude": []})
    return pd.DataFrame({"sample": np.arange(len(beat)), "amplitude": beat})


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def analyse_image(image):
    try:
        pred = inf.predict_from_image(image)
    except inf.InputValidationError as exc:
        return _error_card("Unsupported image", str(exc)), None, None, gr.update(visible=False)
    except inf.ModelUnavailableError as exc:
        return _error_card("Model unavailable", str(exc)), None, None, gr.update(visible=False)
    return (
        _result_card(pred, "scalogram image"),
        pred.probabilities,
        pred.scalogram,
        gr.update(visible=False),
    )


def analyse_beat(file_path, sampling_rate):
    if not file_path:
        return _error_card("No file", "Please upload a CSV file containing one beat."), None, None, gr.update(visible=False)
    try:
        data = Path(file_path).read_bytes()
        pred, beat = inf.predict_from_beat_csv(data, sampling_rate)
    except inf.InputValidationError as exc:
        return _error_card("Invalid ECG beat", str(exc)), None, None, gr.update(visible=False)
    except inf.ModelUnavailableError as exc:
        return _error_card("Model unavailable", str(exc)), None, None, gr.update(visible=False)
    return (
        _result_card(pred, "ECG beat CSV → CWT"),
        pred.probabilities,
        pred.scalogram,
        gr.update(value=_beat_frame(beat), visible=True),
    )


def clear_outputs():
    return EMPTY_CARD, None, None, gr.update(value=_beat_frame(None), visible=False)


# ---------------------------------------------------------------------------
# Static content
# ---------------------------------------------------------------------------
HEADER = """
<div class="app-header">
  <div>
    <div class="app-kicker">Research demo · V4 model</div>
    <h1 class="app-title">ECG Arrhythmia Classification</h1>
    <p class="app-sub">Continuous wavelet transform (Mexican hat) scalograms of single heartbeats,
    classified into five AAMI beat classes by a compact convolutional neural network.</p>
  </div>
  <div class="app-badges">
    <span class="badge">5 AAMI classes</span>
    <span class="badge">256×256 CWT input</span>
    <span class="badge">129,477 parameters</span>
    <span class="badge badge-warn">Not a medical device</span>
  </div>
</div>
"""

IMAGE_HELP = """
**Accepted:** a grayscale V4 scalogram (|CWT| magnitude, 256×256, values min–max scaled to 0–255),
such as the examples below. These are the images the model was trained on.

**Rejected:** colour images (e.g. jet-colormap plots), transparent, blank or non-square images.
"""

BEAT_HELP = """
**Format:** a `.csv` (or `.txt`) file with **exactly 300 numeric values**: one per line, or a single row.
One non-numeric header line (e.g. `ecg_mV`) is allowed.

- The beat must be sampled at **360 Hz** and centred on the annotated R peak (150 samples before, 150 after).
- The signal is **not** resampled, padded or trimmed. Other lengths or sampling rates are rejected.
- This accepts one pre-segmented beat only; it is not a detector for full ECG recordings.

The beat goes through the documented V4 preprocessing: Mexican-hat CWT (scales 1–128) → absolute value →
resize to 256×256 → min–max to 0–255 (uint8) → 3 identical channels → ÷255.
"""

ABOUT_MD = """
### What this demo does
It runs the original **V4** convolutional network (`models/v4_custom_fast_mode.keras`) on a CWT scalogram
of a single heartbeat and shows its softmax output for five AAMI classes:
**N** (normal and bundle-branch beats), **S** (supraventricular ectopic), **V** (ventricular ectopic),
**F** (fusion) and **Q** (paced / unclassifiable).

### Model
| | |
|---|---|
| Architecture | Conv 32 (5×5, stride 2) → MaxPool → BatchNorm → Conv 64 (3×3, stride 2) → MaxPool → BatchNorm → Conv 128 (3×3) → GlobalAveragePooling → Dense 256 → Dropout 0.5 → Dense 5 (softmax) |
| Input | 256 × 256 × 3 (grayscale scalogram repeated on 3 channels), scaled to [0, 1] |
| Parameters | 129,477 |
| Training data | 39,982 beat scalograms (all minority-class beats + 20,000 normal beats) from 48 MIT-BIH Arrhythmia Database records and 27 MIT-BIH ST Change Database records (301–327) |
| Framework | TensorFlow / Keras 3.13.2 (checkpoint saved 2026-04-16) |

### Historical result (not independently validated)
The original notebook reports **94.64 % accuracy** (macro F1 0.87) on 7,997 beats. Those beats were drawn
from the same pool used for training, with a beat-level (not patient-level) random split, and most of them
were probably seen during training. The figure is therefore **not** an estimate of performance on new
patients. No patient-wise evaluation of this model exists.

### Limitations & disclaimer
- Research and portfolio demonstration only. **Not a medical device; not for diagnosis or treatment.**
- Input must already be a single, R-peak-centred, 360 Hz beat (or its V4 scalogram). No QRS detection,
  filtering or quality checks are performed.
- The example scalograms and beats come from records used in training, so predictions on them are not a test of generalisation.
- Softmax probabilities are not calibrated confidence.

Contains information from the MIT-BIH Arrhythmia Database and the MIT-BIH ST Change Database (PhysioNet), which are
made available under the [ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/). The example inputs come
from the MIT-BIH Arrhythmia Database. See NOTICE.md in the repository for citations.
"""

CSS = """
:root { --accent: #38bdf8; --accent-2: #a78bfa; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
.app-header { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap;
  padding: 26px 28px; border-radius: 18px; border: 1px solid var(--border-color-primary);
  background: linear-gradient(120deg, rgba(56,189,248,.14), rgba(167,139,250,.14)); }
.app-kicker { text-transform: uppercase; letter-spacing: .12em; font-size: 12px; font-weight: 600; color: var(--accent); }
.app-title { margin: 6px 0 6px; font-size: 30px; line-height: 1.15; font-weight: 700; }
.app-sub { margin: 0; max-width: 640px; color: var(--body-text-color-subdued); font-size: 15px; }
.app-badges { display:flex; flex-wrap:wrap; gap:8px; }
.badge { font-size: 12px; font-weight: 600; padding: 5px 11px; border-radius: 999px;
  border: 1px solid rgba(56,189,248,.45); color: var(--accent); background: rgba(56,189,248,.08); }
.badge-warn { border-color: rgba(251,191,36,.5); color: #fbbf24; background: rgba(251,191,36,.08); }
.panel { border-radius: 16px !important; }
.result-card { border-radius: 14px; padding: 18px 20px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); min-height: 150px; }
.result-card.ok { border-left: 4px solid #34d399; }
.result-card.warn { border-left: 4px solid #fbbf24; }
.result-card.error { border-left: 4px solid #f87171; }
.rc-eyebrow { text-transform: uppercase; letter-spacing: .1em; font-size: 11px; font-weight: 600; color: var(--body-text-color-subdued); }
.rc-title { display:flex; align-items:center; gap:10px; font-size: 24px; font-weight: 700; margin: 6px 0 4px; }
.rc-dot { width: 12px; height: 12px; border-radius: 50%; display:inline-block; }
.rc-metric { display:flex; align-items:baseline; gap:10px; margin: 4px 0 8px; }
.rc-value { font-size: 40px; font-weight: 800; color: var(--accent); font-variant-numeric: tabular-nums; }
.rc-unit { font-size: 13px; color: var(--body-text-color-subdued); }
.rc-text { margin: 0; font-size: 14px; color: var(--body-text-color-subdued); }
.rc-warnings { margin: 12px 0 0; padding-left: 18px; font-size: 13px; color: #fbbf24; }
.footer-note { text-align:center; font-size: 12px; color: var(--body-text-color-subdued); padding: 8px 0 16px; }
@media (max-width: 640px) { .app-title { font-size: 24px; } .app-header { padding: 20px; } .rc-value { font-size: 32px; } }
"""

THEME = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="violet",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "monospace"],
)

FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') { document.body.classList.add('dark'); }
}
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ECG Arrhythmia Classification · CWT + CNN") as demo:
        gr.HTML(HEADER)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                with gr.Tabs():
                    with gr.Tab("Scalogram image", id="image"):
                        image_in = gr.Image(
                            type="pil", image_mode=None, label="V4 scalogram (grayscale, 256×256)",
                            sources=["upload", "clipboard"], height=300,
                        )
                        gr.Markdown(IMAGE_HELP)
                        with gr.Row():
                            image_clear = gr.Button("Clear", variant="secondary")
                            image_btn = gr.Button("Analyse scalogram", variant="primary")
                        gr.Examples(
                            examples=IMAGE_EXAMPLES, inputs=[image_in], example_labels=IMAGE_EXAMPLE_LABELS,
                            label="Example scalograms (training-pool samples from the V4 balanced dataset)",
                            examples_per_page=10,
                        )
                    with gr.Tab("ECG beat (CSV)", id="beat"):
                        beat_in = gr.File(label="Single-beat CSV (300 samples @ 360 Hz)", file_types=[".csv", ".txt"], type="filepath")
                        fs_in = gr.Number(
                            label="Sampling frequency of this beat (Hz)", value=None, precision=None,
                            info="Required. Only 360 Hz is accepted; signals are never resampled.",
                        )
                        gr.Markdown(BEAT_HELP)
                        with gr.Row():
                            beat_clear = gr.Button("Clear", variant="secondary")
                            beat_btn = gr.Button("Analyse beat", variant="primary")
                        gr.Examples(
                            examples=BEAT_EXAMPLES, inputs=[beat_in, fs_in], example_labels=BEAT_EXAMPLE_LABELS,
                            label="Example beats (MIT-BIH records that were part of the training pool)",
                        )

            with gr.Column(scale=6, min_width=320):
                result_html = gr.HTML(EMPTY_CARD)
                prob_out = gr.Label(label="Class probabilities (softmax)", num_top_classes=5, show_heading=False)
                with gr.Row():
                    scalogram_out = gr.Image(
                        label="Model input scalogram (exactly what the CNN received)", type="numpy",
                        image_mode="L", interactive=False, height=256,
                    )
                beat_plot = gr.LinePlot(
                    value=_beat_frame(None), x="sample", y="amplitude", label="Uploaded ECG beat",
                    x_title="Sample (360 Hz)", y_title="Amplitude", height=240, visible=False,
                )

        with gr.Accordion("About the project, model, results and limitations", open=False):
            gr.Markdown(ABOUT_MD)

        gr.HTML(
            '<div class="footer-note">Educational research demo. Not for clinical use. '
            "Predictions are produced live by the original V4 checkpoint.</div>"
        )

        outputs = [result_html, prob_out, scalogram_out, beat_plot]
        image_btn.click(analyse_image, inputs=[image_in], outputs=outputs, api_name="predict_image")
        beat_btn.click(analyse_beat, inputs=[beat_in, fs_in], outputs=outputs, api_name="predict_beat")
        image_clear.click(lambda: None, outputs=[image_in], api_name=False).then(clear_outputs, outputs=outputs, api_name=False)
        beat_clear.click(lambda: (None, None), outputs=[beat_in, fs_in], api_name=False).then(
            clear_outputs, outputs=outputs, api_name=False
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    try:
        inf.load_model()  # load once at startup so the first request is fast
        print(f"Model loaded: {inf.MODEL_PATH.name}")
    except inf.ModelUnavailableError as exc:  # the UI still starts and shows the error per request
        print(f"WARNING: {exc}")
    demo.launch(theme=THEME, css=CSS, js=FORCE_DARK_JS)
