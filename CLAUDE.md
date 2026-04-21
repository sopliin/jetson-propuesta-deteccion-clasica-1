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

# Video file
python src/main.py --source video --input assets/videos/test1.mp4

# MIPI CSI camera
python src/main.py --source csi

# With precision/recall evaluation
python src/main.py --source video --input assets/videos/test1.mp4 \
                   --eval results/annotations/sample.json
```

**Jetson only:** Do NOT `pip install opencv-python` — it replaces the JetPack CUDA build.

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

## Key Files

| File | Responsibility |
|---|---|
| `src/main.py` | CLI, main loop, terminal status, session teardown |
| `src/capture/source.py` | `open_source()` — video file or GStreamer CSI pipeline |
| `src/detection/background.py` | MOG2 creation and per-frame application |
| `src/detection/contours.py` | `clean_mask()`, `extract_contours()`, `filter_contours()` |
| `src/detection/classifier.py` | `classify_detections()` — geometric rules |
| `src/tracking/centroid.py` | `CentroidTracker` — greedy nearest-neighbour matching |
| `src/metrics/evaluator.py` | IoU matching, precision/recall/F1 from JSON annotations |
| `src/utils/gpu_check.py` | `check_gpu()` — cv2.cuda device count |
| `config/default.yaml` | All tunable thresholds (areas, aspect ratios, kernel sizes) |

## Configuration

All detection thresholds live in `config/default.yaml`. There is no hard-coded threshold in source code — every numeric limit is loaded via `cfg.get("key", default)`. Edit the YAML to tune without touching Python.

## Evaluation Annotations

Ground-truth files go in `results/annotations/` as JSON with frame keys `frame_XXXX`. See `results/annotations/sample.json` for the format. Run with `--eval <path>` to get precision/recall/F1 per class.

## Dependencies

- `numpy` — centroid distance matrix
- `pyyaml` — config loading
- `opencv-python` — everything else (on Jetson: use system install)
