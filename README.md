# Classical CV Traffic Monitor — Jetson Orin Nano

Real-time pedestrian and vehicle detection using a **single classical computer vision pipeline**: background subtraction → morphological filtering → contour extraction → geometric classification → centroid tracking. No deep learning, no external models.

Designed for academic demonstration on the **Jetson Orin Nano (JetPack 6.2.2)**.

---

## Algorithm in One Sentence

MOG2 background subtraction segments moving objects; morphological operations clean the mask; contour analysis extracts blobs; geometric rules separate pedestrians (tall boxes) from vehicles (wide/large boxes); centroid matching tracks objects across frames.

See [docs/algorithm.md](docs/algorithm.md) for the full step-by-step explanation.

---

## Project Structure

```
E2/
├── src/
│   ├── main.py                      # Entry point and main loop
│   ├── capture/source.py            # Video file / MIPI CSI input
│   ├── detection/
│   │   ├── background.py            # MOG2 background subtractor
│   │   ├── contours.py              # Morphological filter + contour extraction
│   │   └── classifier.py           # Geometric pedestrian/vehicle classifier
│   ├── tracking/centroid.py         # Centroid tracker (stable IDs)
│   ├── metrics/
│   │   ├── fps.py                   # Rolling FPS counter
│   │   ├── stats.py                 # Session statistics
│   │   └── evaluator.py            # Precision / recall evaluation
│   ├── visualization/overlay.py    # HUD and bounding-box drawing
│   └── utils/gpu_check.py          # OpenCV CUDA detection
├── config/default.yaml             # All tunable parameters
├── assets/videos/                  # Place test videos here
├── results/annotations/            # Ground-truth JSON files
├── docs/algorithm.md               # Detailed algorithm explanation
├── scripts/setup.sh                # One-time environment setup
└── requirements.txt
```

---

## Installation (Jetson Orin Nano — JetPack 6.2.2)

JetPack ships with an OpenCV build that includes CUDA support. **Do not install `opencv-python` via pip** — it replaces the GPU build with a CPU-only wheel.

```bash
# 1. Verify JetPack OpenCV has CUDA
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda

# 2. Run the setup script (creates .venv with --system-site-packages)
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Activate the environment
source .venv/bin/activate
```

### Development machine (non-Jetson)

```bash
pip install opencv-python numpy pyyaml
```

---

## Usage

```bash
source .venv/bin/activate

# Video file
python src/main.py --source video --input assets/videos/test1.mp4

# MIPI CSI camera (Jetson Orin Nano)
python src/main.py --source csi

# Custom configuration file
python src/main.py --source video --input assets/videos/test1.mp4 \
                   --config config/default.yaml

# With precision/recall evaluation
python src/main.py --source video --input assets/videos/test1.mp4 \
                   --eval results/annotations/sample.json

# Headless (no display window — terminal output only)
python src/main.py --source video --input assets/videos/test1.mp4 --no-display
```

---

## Runtime Outputs

### OpenCV Window

| Element | Description |
|---|---|
| Green box + **P** | Pedestrian detection |
| Orange box + **V** | Vehicle detection |
| Coloured dot + ID | Centroid tracker ID |
| Top-left HUD | FPS / pedestrian count / vehicle count / frame number |

### Terminal (printed every 2 seconds)

```
====================================================
  Compute device : GPU: Orin (integrated) (CUDA device 0)
  Inference mode : GPU-accelerated (CUDA)
====================================================
[INFO] Source   : video file — assets/videos/test1.mp4
[INFO] Resolution : 1280x720  |  Source FPS : 30.0
[INFO] Detection loop running. Press 'q' in the window to stop.

[000060]  FPS= 28.4  PED=  2  VEH=  1  time= 35.2 ms  res=1280x720  src=video  dev=GPU  det= 72.5%
[000120]  FPS= 29.1  PED=  1  VEH=  2  time= 34.4 ms  res=1280x720  src=video  dev=GPU  det= 68.3%
```

### End-of-session summary

```
============================================================
  SESSION SUMMARY
============================================================
  Total frames processed  : 900
  Average FPS             : 28.9
  Avg frame time          : 34.6 ms
  Worst frame time        : 62.1 ms
  Frames with detections  : 623  (69.2 %)
  Total pedestrian obs.   : 1240
  Total vehicle obs.      : 874
============================================================
```

---

## Precision / Recall Evaluation

Create a JSON annotation file in `results/annotations/`:

```json
{
  "frame_0050": {
    "pedestrians": [[118, 195, 44, 112], [298, 178, 41, 98]],
    "vehicles":    [[445, 148, 185, 105]]
  },
  "frame_0100": {
    "pedestrians": [[128, 200, 44, 112]],
    "vehicles":    [[455, 152, 190, 102]]
  }
}
```

Frame keys follow `frame_XXXX` (zero-padded frame index). Boxes are `[x, y, w, h]` in pixels.

Run with `--eval` to see:

```
============================================================
  PRECISION / RECALL EVALUATION
============================================================
  PEDESTRIAN    Precision=0.812  Recall=0.765  F1=0.788  [TP=13  FP=3  FN=4]
  VEHICLE       Precision=0.875  Recall=0.824  F1=0.849  [TP=14  FP=2  FN=3]
============================================================
```

IoU threshold is configurable (`iou_threshold` in `config/default.yaml`, default 0.4).

---

## Configuration

All tunable parameters are in [config/default.yaml](config/default.yaml). Key entries:

| Parameter | Default | Effect |
|---|---|---|
| `bg_history` | 300 | MOG2 background model length (frames) |
| `bg_var_threshold` | 50 | MOG2 sensitivity (lower = more foreground) |
| `morph_open_ksize` | 5 | Noise removal kernel size (px) |
| `morph_close_ksize` | 15 | Hole-filling kernel size (px) |
| `contour_min_area` | 800 | Smallest accepted blob (px²) |
| `contour_max_area` | 80000 | Largest accepted blob (px²) |
| `ped_aspect_min` | 1.2 | Minimum h/w for pedestrian |
| `veh_min_area` | 4000 | Minimum area for vehicle (px²) |
| `tracker_max_disappeared` | 30 | Frames before a track is deleted |
| `iou_threshold` | 0.4 | IoU for evaluation |

---

## GPU / CPU Validation

At startup the system checks `cv2.cuda.getCudaEnabledDeviceCount()`:

- **Jetson Orin Nano with JetPack 6:** CUDA device 0 present → GPU mode reported.
- **Machine without CUDA OpenCV:** CPU mode reported; pipeline runs identically.

Background subtraction (MOG2) currently runs on CPU because it is already
fast enough for real-time at 720p. The GPU check is printed for transparency
and for future CUDA-accelerated extensions.

---

## Web Server (Future Extension)

The previous version of this project (`orin-nano-ui-rtsp/`) used a FastAPI
web server with MJPEG streaming and a JSON REST API. That approach is suitable
for remote monitoring but adds latency and infrastructure complexity.

**For this academic demo, the real-time OpenCV window is the primary interface.**
A web API (exposing `/api/counts` and `/video_feed`) could be added as an
optional extension — see `orin-nano-ui-rtsp/` for a working reference
implementation — but it is not part of the main demo flow.

---

## Known Limitations

- MOG2 struggles with **stationary vehicles**: after `bg_history` frames the
  vehicle is absorbed into the background model and disappears.
- **Camera shake** or **rapid lighting changes** generate many false positives.
- **Motorcycle / bicycle geometry** is ambiguous (borderline aspect ratio and area).
- Detections are **not scale-normalised**: a distant vehicle and a nearby
  pedestrian can have the same bounding-box area.

These are expected and should be discussed in the academic presentation.
# jetson-propuesta-deteccion-clasica-1
