"""
Input source abstraction.

Supports two modes:
  'video' — reads from a local video file (MP4, AVI, etc.)
  'csi'   — reads from the MIPI CSI camera on Jetson Orin Nano
             via a GStreamer pipeline (requires JetPack nvarguscamerasrc).

The rest of the pipeline does not need to know which source is active.
"""
import sys
import cv2


# ---------------------------------------------------------------------------
# GStreamer pipeline builder for MIPI CSI camera
# ---------------------------------------------------------------------------

def _csi_gstreamer_pipeline(width: int, height: int, fps: int, flip: int) -> str:
    """
    Build the GStreamer string for nvarguscamerasrc (Jetson JetPack 6.x).

    flip_method:
      0 = none, 1 = counterclockwise, 2 = 180°, 3 = clockwise
    """
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=2"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def open_source(source_type: str, input_path: str | None, cfg: dict) -> cv2.VideoCapture:
    """
    Open and return a VideoCapture for the requested source.

    Args:
        source_type : 'video' | 'csi'
        input_path  : path to video file; required when source_type == 'video'
        cfg         : configuration dict (used for CSI camera parameters)

    Returns:
        cv2.VideoCapture — already opened

    Raises:
        SystemExit on any failure so the error is printed before exit.
    """
    if source_type == "video":
        if not input_path:
            print("[ERROR] --input <path> is required when --source is 'video'.")
            sys.exit(1)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {input_path}")
            sys.exit(1)
        print(f"[INFO] Source   : video file — {input_path}")

    elif source_type == "csi":
        pipeline = _csi_gstreamer_pipeline(
            width  = cfg.get("csi_width",  1280),
            height = cfg.get("csi_height",  720),
            fps    = cfg.get("csi_fps",      30),
            flip   = cfg.get("csi_flip",      0),
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("[ERROR] Cannot open CSI camera. Verify GStreamer/JetPack installation.")
            print(f"        Pipeline tried: {pipeline}")
            sys.exit(1)
        print("[INFO] Source   : MIPI CSI camera (GStreamer)")

    else:
        print(f"[ERROR] Unknown source '{source_type}'. Use 'video' or 'csi'.")
        sys.exit(1)

    return cap


def get_frame_properties(cap: cv2.VideoCapture) -> tuple[int, int, float]:
    """
    Return (width, height, fps) from a VideoCapture.

    Note: cameras often report fps=0; we default to 30 in that case.
    """
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return width, height, fps
