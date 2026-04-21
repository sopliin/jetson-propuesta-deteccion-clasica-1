"""
main.py — Classical CV Traffic Monitor
=======================================
Real-time pedestrian and vehicle detection for Jetson Orin Nano.

Pipeline (one integrated classical proposal):
  capture frame
    → background subtraction (MOG2)
    → morphological filtering (open + close)
    → contour extraction + area filter
    → geometric classifier (pedestrian / vehicle)
    → centroid tracker (stable IDs across frames)
    → overlay (HUD + boxes + centroids)
    → display in OpenCV window

Usage:
  python src/main.py --source video --input assets/videos/test1.mp4
  python src/main.py --source csi
  python src/main.py --source video --input assets/videos/test1.mp4 \\
                     --eval results/annotations/sample.json

Press 'q' in the display window to stop.
"""

import argparse
import os
import sys
import time

import cv2
import yaml

# ---- Make src/ importable when running as "python src/main.py" from project root
sys.path.insert(0, os.path.dirname(__file__))

from capture.source        import open_source, get_frame_properties
from detection.background  import create_subtractor, apply_subtractor
from detection.contours    import clean_mask, extract_contours, filter_contours
from detection.classifier  import classify_detections
from tracking.centroid     import CentroidTracker
from metrics.fps           import FPSCounter
from metrics.stats         import DetectionStats
from metrics.evaluator     import load_annotations, evaluate, print_evaluation
from utils.gpu_check       import check_gpu, print_device_status
from utils.system_monitor  import format_status
from visualization.overlay import draw_detections, draw_tracks, draw_hud


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CFG = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")


def load_config(path: str) -> dict:
    """Load YAML config file; fall back silently to an empty dict."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"[WARN] Config file not found: {path}. Using built-in defaults.")
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classical CV Traffic Monitor — Jetson Orin Nano",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--source", required=True, choices=["video", "csi"],
        help="Input source:\n  video — read from a file\n  csi   — MIPI CSI camera",
    )
    p.add_argument(
        "--input", default=None,
        help="Path to video file (required when --source video)",
    )
    p.add_argument(
        "--config", default=_DEFAULT_CFG,
        help="Path to YAML configuration file",
    )
    p.add_argument(
        "--eval", default=None, metavar="ANNOTATIONS_JSON",
        help="JSON ground-truth file for precision/recall evaluation",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Run without an OpenCV window (terminal output only)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Terminal status line
# ---------------------------------------------------------------------------

_PRINT_INTERVAL_S = 2.0   # seconds between terminal status updates

# FPS below this value triggers an alert line in the terminal
_FPS_ALERT_THRESHOLD = 15.0


def _print_status(
    frame_id: int,
    fps: float,
    n_ped: int,
    n_veh: int,
    frame_ms: float,
    resolution: str,
    source: str,
    dev_short: str,
    det_rate: float,
) -> None:
    # --- Line 1: detection metrics ---
    fps_flag = "  [!FPS]" if fps < _FPS_ALERT_THRESHOLD and fps > 0 else ""
    print(
        f"[{frame_id:06d}]  "
        f"FPS={fps:5.1f}{fps_flag}  "
        f"PED={n_ped:3d}  VEH={n_veh:3d}  "
        f"time={frame_ms:5.1f}ms  "
        f"det={det_rate:5.1f}%  "
        f"res={resolution}  src={source}  dev={dev_short}"
    )

    # --- Line 2: system resources ---
    sys_line, alerts = format_status()
    print(f"         {sys_line}")

    # --- Alert lines (only when triggered) ---
    for alert in alerts:
        print(f"         {alert}")


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace, cfg: dict) -> None:

    # ---- 1. GPU check ---------------------------------------------------
    gpu_available, gpu_desc = check_gpu()
    print_device_status(gpu_available, gpu_desc)
    dev_short = "GPU" if gpu_available else "CPU"

    # ---- 2. Open input source -------------------------------------------
    cap = open_source(args.source, args.input, cfg)
    width, height, src_fps = get_frame_properties(cap)
    resolution = f"{width}x{height}"
    print(f"[INFO] Resolution : {resolution}  |  Source FPS : {src_fps:.1f}")

    # ---- 3. Build pipeline components -----------------------------------
    subtractor = create_subtractor(
        history       = cfg.get("bg_history",       300),
        var_threshold = cfg.get("bg_var_threshold",  50),
    )
    tracker = CentroidTracker(
        max_disappeared = cfg.get("tracker_max_disappeared", 30),
        max_distance    = cfg.get("tracker_max_distance",    80),
    )
    fps_counter = FPSCounter(window=cfg.get("fps_window", 30))
    stats       = DetectionStats()

    # ---- 4. Evaluation book-keeping -------------------------------------
    annotations: dict | None      = load_annotations(args.eval) if args.eval else None
    frame_detections: dict         = {}   # frame_key → detections list

    # ---- 5. Display window ----------------------------------------------
    window = "Traffic Monitor — Classical CV"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(width, 1280), min(height, 720))

    # ---- 6. Main loop ---------------------------------------------------
    frame_id   = 0
    last_print = time.perf_counter()
    n_ped = n_veh = 0

    print("[INFO] Detection loop running. Press 'q' in the window to stop.\n")

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended or read error — stopping.")
            break

        # --- Background subtraction ---
        fg_mask = apply_subtractor(subtractor, frame)

        # --- Morphological filtering ---
        clean = clean_mask(
            fg_mask,
            open_ksize  = cfg.get("morph_open_ksize",   5),
            close_ksize = cfg.get("morph_close_ksize",  15),
        )

        # --- Contour extraction ---
        contours   = extract_contours(clean)
        boxes      = filter_contours(
            contours,
            min_area = cfg.get("contour_min_area",    800),
            max_area = cfg.get("contour_max_area",  80_000),
        )

        # --- Geometric classification ---
        detections = classify_detections(boxes, cfg)

        # --- Centroid tracking ---
        tracks       = tracker.update(detections)
        n_ped, n_veh = tracker.count()

        # --- Timing & stats ---
        frame_ms = (time.perf_counter() - t0) * 1_000.0
        fps_counter.tick()
        fps = fps_counter.fps()
        stats.update(n_ped, n_veh, frame_ms)

        # --- Store for evaluation if this frame is annotated ---
        if annotations is not None:
            key = f"frame_{frame_id:04d}"
            if key in annotations:
                frame_detections[key] = detections

        # --- Terminal status every N seconds ---
        now = time.perf_counter()
        if now - last_print >= _PRINT_INTERVAL_S:
            _print_status(frame_id, fps, n_ped, n_veh, frame_ms,
                          resolution, args.source, dev_short,
                          stats.detection_rate())
            last_print = now

        # --- Draw and display ---
        if not args.no_display:
            vis = frame.copy()
            draw_detections(vis, detections)
            draw_tracks(vis, tracks)
            draw_hud(vis, fps, n_ped, n_veh, frame_id)
            cv2.imshow(window, vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("[INFO] User requested quit.")
                break

        frame_id += 1

    # ---- 7. Cleanup -----------------------------------------------------
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ---- 8. End-of-session reports --------------------------------------
    stats.print_summary()

    if annotations is not None:
        if frame_detections:
            results = evaluate(
                annotations,
                frame_detections,
                iou_threshold=cfg.get("iou_threshold", 0.4),
            )
            print_evaluation(results)
        else:
            print("[WARN] --eval provided but no annotated frame keys matched the stream.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    run(args, cfg)
