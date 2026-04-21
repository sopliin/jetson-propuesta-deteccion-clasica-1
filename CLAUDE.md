# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Classical computer vision pipeline for real-time pedestrian and vehicle detection on a **Jetson Orin Nano (JetPack 6.2.2)**. No deep learning — the detector is MOG2 background subtraction + morphological filtering + contour extraction + geometric classification + centroid tracking. Output is a real-time OpenCV window.

The `orin-nano-ui-rtsp/` subdirectory is a previous YOLO/FastAPI version kept for reference. Active development is at the repo root.

## Setup & Run

```bash
# One-time setup (Jetson: inherits system OpenCV via --system-site-packages)
chmod +x scripts/setup.sh && ./scripts/setup.sh
source .venv/bin/activate

# All commands must be run from the project root
# Video file
python src/main.py --source video --input assets/videos/test1.mp4

# MIPI CSI camera
python src/main.py --source csi

# Headless / SSH — no OpenCV window, terminal stats only
python src/main.py --source video --input assets/videos/test1.mp4 --no-display

# With precision/recall evaluation
python src/main.py --source video --input assets/videos/test1.mp4 \
                   --eval results/annotations/sample.json

# Custom config (defaults to config/default.yaml)
python src/main.py --source video --input assets/videos/test1.mp4 \
                   --config config/default.yaml
```

**Jetson only:** Do NOT `pip install opencv-python` — it replaces the JetPack CUDA build.

**Non-Jetson dev machine:** `pip install opencv-python numpy pyyaml`

## Architecture

**Single classical pipeline — one frame at a time:**

```
capture (source.py)
  → MOG2 background subtraction (background.py)
  → morphological open+close (contours.py)
  → contour extraction + area filter (contours.py)
  → geometric classifier: pedestrian/vehicle (classifier.py)
  → centroid tracker — stable IDs across frames (centroid.py)
  → HUD overlay + cv2.imshow (overlay.py, main.py)
```

All pipeline stages are stateless pure functions except:
- `BackgroundSubtractorMOG2` — owns the background model (stateful, one instance)
- `CentroidTracker` — owns the track dictionary (stateful, one instance)

Both live in `run()` in `main.py` and are created once per session.

**Data contract between stages:** detections flow as `list[tuple[int, int, int, int, str]]` — each tuple is `(x, y, w, h, label)` where label is `"pedestrian"`, `"vehicle"`, or `"unknown"`. Before classification, contours are plain `(x, y, w, h)` tuples.

## Key Files

| File | Responsibility |
|---|---|
| `src/main.py` | CLI, main loop, terminal status, session teardown |
| `src/capture/source.py` | `open_source()` — video file or GStreamer CSI pipeline |
| `src/detection/background.py` | MOG2 creation and per-frame application |
| `src/detection/contours.py` | `clean_mask()`, `extract_contours()`, `filter_contours()` |
| `src/detection/classifier.py` | `classify_detections()` — geometric rules |
| `src/tracking/centroid.py` | `CentroidTracker` — greedy nearest-neighbour matching |
| `src/metrics/fps.py` | `FPSCounter` — rolling average FPS over a frame window |
| `src/metrics/stats.py` | `DetectionStats` — session summary (avg time, det rate) |
| `src/metrics/evaluator.py` | IoU matching, precision/recall/F1 from JSON annotations |
| `src/visualization/overlay.py` | `draw_detections()`, `draw_tracks()`, `draw_hud()` — all OpenCV drawing |
| `src/utils/system_monitor.py` | `format_status()` — CPU/GPU/RAM metrics for terminal output |
| `src/utils/gpu_check.py` | `check_gpu()` — cv2.cuda device count |
| `config/default.yaml` | All tunable thresholds (areas, aspect ratios, kernel sizes) |

## Configuration

All detection thresholds live in `config/default.yaml`. There is no hard-coded threshold in source code — every numeric limit is loaded via `cfg.get("key", default)`. Edit the YAML to tune without touching Python.

**Classifier threshold overlap is intentional:** `ped_max_area` (12,000) and `veh_min_area` (4,000) intentionally overlap. When a blob falls in the 4,000–12,000 px² range, aspect ratio decides: `h/w ≥ ped_aspect_min` → pedestrian, otherwise → vehicle. For ambiguous cases outside this, area is the final tie-breaker (see `classifier.py:classify_box`).

## Evaluation Annotations

Ground-truth files go in `results/annotations/` as JSON with frame keys `frame_XXXX` (4-digit zero-padded frame index). Boxes are `[x, y, w, h]` in pixels. The plural class keys are `"pedestrians"` and `"vehicles"`. See `results/annotations/sample.json` for the format. Run with `--eval <path>` to get precision/recall/F1 per class.

## Known Limitations

- **Stationary vehicles** are absorbed into the background model after `bg_history` frames and vanish from detections.
- **Camera shake / rapid lighting changes** generate many false positives.
- **Motorcycles and bicycles** have borderline aspect ratio and area — classification is ambiguous.
- Detections are **not scale-normalised**: a distant vehicle and a nearby pedestrian can share the same bounding-box area, causing misclassification.

## Dependencies

- `numpy` — centroid distance matrix
- `pyyaml` — config loading
- `psutil` — CPU/RAM usage for terminal metrics (`system_monitor.py`)
- `opencv-python` — everything else (on Jetson: use system install)
