"""Capture real screenshots of the running Gradio app (development helper).

Start the app first (``python app/app.py``), then:

    python scripts/take_screenshots.py --url http://127.0.0.1:7860

Uses Playwright with the locally installed Google Chrome (``channel="chrome"``).
Every screenshot shows live output from the model; nothing is mocked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots"


def wait_result(page, text: str, timeout: int = 60_000) -> None:
    page.locator(".result-card", has_text=text).first.wait_for(timeout=timeout)
    page.wait_for_timeout(800)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)

        # Desktop: scalogram example -> prediction
        page = browser.new_page(viewport={"width": 1360, "height": 1000}, device_scale_factor=2)
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=OUT / "01_home.png", full_page=True)

        page.get_by_text("Ventricular (V) · dataset idx", exact=False).first.click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="Analyse scalogram").click()
        wait_result(page, "Model output")
        page.screenshot(path=OUT / "02_scalogram_prediction.png", full_page=True)

        # Desktop: CSV beat example -> CWT -> prediction (with waveform)
        page.get_by_role("tab", name="ECG beat (CSV)").click()
        page.wait_for_timeout(500)
        page.get_by_text("MIT-BIH 208", exact=False).first.click()
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Analyse beat").click()
        wait_result(page, "ECG beat CSV")
        page.screenshot(path=OUT / "03_csv_beat_prediction.png", full_page=True)

        # Validation error: colour (jet) image is rejected
        page.get_by_role("tab", name="Scalogram image").click()
        page.wait_for_timeout(500)
        page.locator("input[type=file]").first.set_input_files(str(ROOT / "docs" / "assets" / "colour_scalogram_example.png"))
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Analyse scalogram").click()
        wait_result(page, "Input not accepted")
        page.screenshot(path=OUT / "04_validation_error.png", full_page=False)
        page.close()

        # Mobile layout
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mobile.goto(args.url, wait_until="networkidle")
        mobile.wait_for_timeout(1500)
        mobile.get_by_text("Normal (N) · dataset idx", exact=False).first.click()
        mobile.wait_for_timeout(800)
        mobile.get_by_role("button", name="Analyse scalogram").click()
        wait_result(mobile, "Model output")
        mobile.screenshot(path=OUT / "05_mobile.png", full_page=True)
        browser.close()
    print("Saved:", sorted(x.name for x in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
