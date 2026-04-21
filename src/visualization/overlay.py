"""
Visualization helpers — all drawing operations are here so that
the main loop stays free of OpenCV drawing calls.

Color convention (BGR):
  Pedestrian  →  green  (0, 200, 0)
  Vehicle     →  orange (0, 140, 255)
  Unknown     →  grey   (160, 160, 160)
"""
import cv2

# BGR color palette
_COLOR = {
    "pedestrian": (0, 200, 0),
    "vehicle":    (0, 140, 255),
    "unknown":    (160, 160, 160),
}


def draw_detections(frame, detections: list[tuple]) -> None:
    """
    Draw a bounding box and class initial for each detection.

    Args:
        frame      : BGR frame — modified in place
        detections : list of (x, y, w, h, label)
    """
    for (x, y, w, h, label) in detections:
        color = _COLOR.get(label, _COLOR["unknown"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Single-letter tag above the box: 'P' or 'V'
        tag = label[0].upper()
        cv2.putText(frame, tag, (x + 3, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_tracks(frame, tracks: dict) -> None:
    """
    Draw centroid dots and track IDs for all active tracks.

    Args:
        frame  : BGR frame — modified in place
        tracks : dict { id: (cx, cy, label) }
    """
    for obj_id, (cx, cy, label) in tracks.items():
        color = _COLOR.get(label, _COLOR["unknown"])
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, str(obj_id), (cx + 6, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, n_ped: int, n_veh: int, frame_id: int) -> None:
    """
    Render a semi-transparent HUD in the top-left corner of the frame.

    Shows the four most important real-time metrics:
      FPS, pedestrian count, vehicle count, and frame number.

    The semi-transparency is achieved by blending an opaque rectangle
    onto a copy of the frame, then replacing the ROI.
    """
    lines = [
        f"FPS  {fps:5.1f}",
        f"PED  {n_ped:4d}",
        f"VEH  {n_veh:4d}",
        f"FRM  {frame_id:6d}",
    ]

    pad      = 6
    line_h   = 20
    box_w    = 128
    box_h    = len(lines) * line_h + pad * 2
    x0, y0   = 8, 8

    # Draw semi-transparent background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        y_text = y0 + pad + 13 + i * line_h
        cv2.putText(frame, line, (x0 + pad, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
