"""
Lightweight precision / recall evaluator.

Ground-truth format (JSON):
  {
    "frame_0050": {
      "pedestrians": [[x, y, w, h], ...],
      "vehicles":    [[x, y, w, h], ...]
    },
    "frame_0100": { ... }
  }

Frame keys use the format "frame_XXXX" (zero-padded 4-digit frame index).

For each annotated frame the evaluator greedily matches detected boxes to
ground-truth boxes using IoU ≥ iou_threshold as the acceptance criterion.

Metrics computed:
  Precision = TP / (TP + FP)   "of all detections, how many were correct?"
  Recall    = TP / (TP + FN)   "of all ground-truth objects, how many were found?"
  F1        = harmonic mean of precision and recall

These are per-class (pedestrian, vehicle) and easy to explain in a demo.

Usage:
  annotations = load_annotations("results/annotations/my_video.json")
  results     = evaluate(annotations, frame_detections, iou_threshold=0.4)
  print_evaluation(results)
"""
import json


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------

def load_annotations(path: str) -> dict:
    """Load ground-truth annotations from a JSON file."""
    with open(path, "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou(a: tuple, b: tuple) -> float:
    """
    Intersection-over-Union for two (x, y, w, h) boxes.

    IoU = area_of_overlap / area_of_union

    Returns a float in [0, 1].
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_boxes(detected: list[tuple], ground_truth: list[tuple],
                iou_threshold: float = 0.4) -> tuple[int, int, int]:
    """
    Greedy match of detected boxes against ground-truth boxes.

    Each ground-truth box can only be matched once.

    Returns:
        (tp, fp, fn)
    """
    matched_gt: set = set()
    tp = 0

    for det in detected:
        best_iou  = 0.0
        best_idx  = -1
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            score = iou(det, gt)
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(detected) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    annotations: dict,
    frame_detections: dict,
    iou_threshold: float = 0.4,
) -> dict:
    """
    Compute precision, recall, and F1 per class across all annotated frames.

    Args:
        annotations      : dict loaded from load_annotations()
        frame_detections : { "frame_XXXX": [(x,y,w,h,label), ...] }
        iou_threshold    : minimum IoU to count a detection as a true positive

    Returns:
        { "pedestrian": {"tp":…, "fp":…, "fn":…, "precision":…, "recall":…, "f1":…},
          "vehicle":    { … } }
    """
    accum = {
        "pedestrian": {"tp": 0, "fp": 0, "fn": 0},
        "vehicle":    {"tp": 0, "fp": 0, "fn": 0},
    }

    for frame_key, gt in annotations.items():
        dets = frame_detections.get(frame_key, [])

        for cls in ("pedestrian", "vehicle"):
            gt_boxes  = [tuple(b) for b in gt.get(cls + "s", [])]
            det_boxes = [(x, y, w, h) for (x, y, w, h, lbl) in dets if lbl == cls]
            tp, fp, fn = match_boxes(det_boxes, gt_boxes, iou_threshold)
            accum[cls]["tp"] += tp
            accum[cls]["fp"] += fp
            accum[cls]["fn"] += fn

    results = {}
    for cls, s in accum.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1         = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
        results[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3),
        }
    return results


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_evaluation(results: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  PRECISION / RECALL EVALUATION")
    print(sep)
    for cls, m in results.items():
        print(f"  {cls.upper():12s}  "
              f"Precision={m['precision']:.3f}  "
              f"Recall={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  "
              f"[TP={m['tp']}  FP={m['fp']}  FN={m['fn']}]")
    print(f"{sep}\n")
