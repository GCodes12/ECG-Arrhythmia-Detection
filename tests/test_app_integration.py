"""End-to-end test: launch the Gradio app and call its API endpoints over HTTP."""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "examples" / "manifest.json").read_text())


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


@pytest.fixture(scope="module")
def client():
    from gradio_client import Client

    from app.app import CSS, THEME, demo

    port = _free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True, theme=THEME, css=CSS)
    try:
        yield Client(f"http://127.0.0.1:{port}/", verbose=False)
    finally:
        demo.close()


def test_api_exposes_only_prediction_endpoints(client):
    endpoints = set(client.view_api(return_format="dict", print_info=False)["named_endpoints"])
    assert endpoints == {"/predict_image", "/predict_beat"}


def test_image_endpoint_returns_real_prediction(client):
    from gradio_client import handle_file

    entry = MANIFEST["scalograms"][0]
    html, label, scalogram, _ = client.predict(
        handle_file(str(ROOT / "examples" / "scalograms" / entry["file"])), api_name="/predict_image"
    )
    assert "Model output" in _text(html)
    assert label["label"] in {c["label"] for c in label["confidences"]}
    assert abs(sum(c["confidence"] for c in label["confidences"]) - 1) < 1e-3
    assert scalogram is not None


def test_beat_endpoint_returns_real_prediction(client):
    from gradio_client import handle_file

    entry = MANIFEST["beats"][0]
    html, label, scalogram, plot = client.predict(
        handle_file(str(ROOT / "examples" / "beats" / entry["file"])), 360, api_name="/predict_beat"
    )
    assert "ECG beat CSV" in _text(html)
    assert len(label["confidences"]) == 5
    assert scalogram is not None


def test_image_endpoint_rejects_colour_image(client):
    from gradio_client import handle_file

    html, label, scalogram, _ = client.predict(
        handle_file(str(ROOT / "docs" / "assets" / "colour_scalogram_example.png")), api_name="/predict_image"
    )
    assert "Input not accepted" in _text(html) and "colour" in _text(html)
    assert not label and not scalogram  # cleared outputs (the client returns {} / None)


def test_beat_endpoint_rejects_wrong_sampling_rate(client):
    from gradio_client import handle_file

    entry = MANIFEST["beats"][0]
    html, label, _, _ = client.predict(
        handle_file(str(ROOT / "examples" / "beats" / entry["file"])), 250, api_name="/predict_beat"
    )
    assert "360 Hz" in _text(html) and not label
