"""
Geometric classifier — separates pedestrians from vehicles.

Rationale:
  In typical traffic surveillance footage:
    • Pedestrians appear as tall, narrow blobs (aspect ratio h/w > 1.2).
      Their bounding boxes are relatively small (< ~12 000 px²).
    • Vehicles appear wider or roughly square and occupy more area (> ~4 000 px²).

  The thresholds overlap on purpose: when area and aspect ratio conflict,
  area is the tie-breaker (larger → vehicle).

All thresholds are loaded from config/default.yaml so they can be tuned
without touching source code.

Limitations:
  • A crouching pedestrian or a motorcycle has ambiguous geometry.
  • A very distant vehicle may be labelled 'pedestrian' due to small area.
  These cases are expected and documented as known limitations.
"""


def classify_box(x: int, y: int, w: int, h: int, cfg: dict) -> str:
    """
    Return 'pedestrian', 'vehicle', or 'unknown' for a bounding box.

    Args:
        x, y, w, h : bounding rectangle (pixels)
        cfg        : configuration dict (see config/default.yaml)

    Returns:
        str label
    """
    area         = w * h
    aspect_ratio = h / w if w > 0 else 0.0   # > 1.0 means taller than wide

    ped_min   = cfg.get("ped_min_area",    600)
    ped_max   = cfg.get("ped_max_area",  12_000)
    veh_min   = cfg.get("veh_min_area",   4_000)
    asp_min   = cfg.get("ped_aspect_min",   1.2)

    # Unambiguous pedestrian: tall shape AND small enough to be a person
    if aspect_ratio >= asp_min and ped_min <= area <= ped_max:
        return "pedestrian"

    # Unambiguous vehicle: large AND not excessively tall
    if area >= veh_min and aspect_ratio < asp_min:
        return "vehicle"

    # Ambiguous: use area as tie-breaker
    if area < veh_min:
        return "pedestrian"

    return "vehicle"


def classify_detections(boxes: list[tuple], cfg: dict) -> list[tuple]:
    """
    Classify a list of bounding boxes.

    Args:
        boxes : list of (x, y, w, h)
        cfg   : configuration dict

    Returns:
        list of (x, y, w, h, label)
    """
    return [(x, y, w, h, classify_box(x, y, w, h, cfg)) for (x, y, w, h) in boxes]
