# Automatioc-product-creater-for-POS-ERP

**OCR + Detection + Masking + Hero Cutout** in one Python script.  
This pipeline detects product regions with YOLO‑World, reads text with EasyOCR, generates helpful overlays, masks non‑text areas for focus, and produces an e‑commerce–ready hero image on a pure white background.

## What this repo contains

- `ocr_pipeline.py` — single, self-contained script (your exact code) that:
  - Runs **YOLO‑World** open‑vocabulary detection on the image.
  - Runs **EasyOCR** to read text lines.
  - Draws **red** detection and OCR boxes.
  - Builds a **masked image** (white outside text clusters; text areas remain visible).
  - Exports **two hero variants** (if a product‑like detection is found):
    - `*_hero.jpeg` — square crop around the best product box on white.
    - `*_hero_cutout.jpeg` — GrabCut background removal + square crop on pure white.
  - Writes a **JSON** with detections, OCR results, timings, and output paths.

- `requirements.txt` — Python dependencies.
- `input/` — put exactly **one** image here before running.
- `output/` — all generated artifacts will be written here (created automatically).
- `output/crops/` — reserved for future per‑box crops.
- `services/product_api/` — FastAPI microservice scaffold for persisting and serving structured product data (first step toward the omnichannel ERP/CRM/POS platform).
  - Includes Postgres + `pgvector` migrations for storing embeddings and performing similarity search.

### Optional product API integration

Set `PRODUCT_API_URL` (for example `http://localhost:9000/api/products`) to have the pipeline upsert products into the FastAPI service automatically after each group is processed.

## Outputs per run

For an input like `input/IMG_5077.jpg`, the script writes to `output/`:

- `IMG_5077_detect.jpeg` — YOLO detections overlay (red boxes + labels).
- `IMG_5077_ocr.jpeg` — OCR line boxes overlay (red boxes + confidences).
- `IMG_5077_masked.jpeg` — everything white **except** text clusters.
- `IMG_5077_hero.jpeg` — square hero crop (white background).
- `IMG_5077_hero_cutout.jpeg` — background‑removed hero on white (requires OpenCV).
- `IMG_5077.json` — structured results (detections, OCR, barcodes, summary, paths).

> If a product‑like box cannot be selected, hero images are skipped (logged in console and JSON).

## Repo hygiene & samples

- `output/` and `input/` are ignored from git (generated artifacts and local images). Regenerate outputs by running the pipeline.
- Keep secrets out of code: set `OPENAI_API_KEY` in your env (the default is now empty).
- A tiny reference JSON lives at `samples/output_sample.json` to show the combined catalog shape without shipping real data.

## Configuration (env vars)

You can tune behavior without touching code by setting environment variables:

- `MAX_SIDE` (default `2000`) — resize the longer image side before inference.
- `YOLO_MODEL` (default `yolov8m-world.pt`) — YOLO‑World model weights.
- `YOLO_CONF` (default `0.25`) — detection confidence threshold.
- `YOLO_IOU` (default `0.45`) — NMS IoU threshold.
- `YOLO_CLASSES` — comma‑separated open‑vocabulary prompts (already set for products/labels).
- `OCR_LANGS` (default `en`) — languages for EasyOCR (comma separated).
- `HERO_SIZE` (default `1024`) — output side (px) for hero images.
- `CUT_PAD_PCT` (default `0.06`) — padding around product box for GrabCut init.
- `CUT_ITERS` (default `5`) — GrabCut iterations.

## Requirements

- **Python 3.11+** (uses `datetime.UTC`).
- System package for **ZBar** (needed by `pyzbar`) — e.g. `zbar` (Linux), `brew install zbar` (macOS), or the Windows installer.
- GPU is optional; script runs on CPU. (YOLO & OCR will be slower on CPU.)

Install Python deps:

```bash
pip install -r requirements.txt
```

## High‑level flow

1. Read single image from `input/` and optionally downscale to `MAX_SIDE`.
2. YOLO‑World open‑vocabulary detection → product/label/packaging boxes.
3. EasyOCR line‑level reading → text boxes + confidences.
4. Draw overlays for detections and OCR lines.
5. Build text‑keep mask (dilated) → white out non‑text regions.
6. Pick best product‑like box → export:
   - Square hero crop on white.
   - Tight hero cutout on white (GrabCut + largest component cleanup).
7. Save JSON with metadata, timings, outputs, and notes.

## Notes

- `output/` and `output/crops/` are auto‑created by the script.
- If OpenCV is unavailable, the cutout hero is skipped and a note is recorded.
- If no OCR lines are detected, the masked image is omitted.
- If no suitable product box is detected, hero images are omitted.
