"""
Centroid tracker.

How it works:
  1. Each bounding box is represented by its centroid (cx, cy).
  2. On the first frame, every centroid is registered as a new track with a
     unique integer ID.
  3. On subsequent frames, incoming centroids are matched to existing tracks
     using Euclidean distance (greedy nearest-neighbour).
  4. Tracks that fail to match for more than `max_disappeared` consecutive
     frames are deleted.

This is the classical approach described in:
  Bradski & Kaehler, "Learning OpenCV 3" (O'Reilly, 2017), Chapter 11.

Greedy vs Hungarian:
  The full Hungarian algorithm (O(n³)) gives the optimal assignment but is
  harder to explain. The greedy approach is O(n² log n) and produces the
  same result for typical low-density traffic scenes (< 20 objects).
"""
from collections import OrderedDict
import numpy as np


def _cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of 2-D points.

    Args:
        a : (M, 2) array
        b : (N, 2) array

    Returns:
        (M, N) distance matrix
    """
    # Broadcasting: expand dims to (M,1,2) and (1,N,2), then sum squared diffs
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


class CentroidTracker:
    """Tracks moving objects across frames by matching centroids."""

    def __init__(self, max_disappeared: int = 30, max_distance: int = 80):
        """
        Args:
            max_disappeared : frames a track survives without a detection match
            max_distance    : maximum pixel distance for centroid association
        """
        self.next_id         = 0
        # { id: (cx, cy, label) }
        self.objects         = OrderedDict()
        # { id: frames_without_match }
        self.disappeared     = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    # ------------------------------------------------------------------
    def _register(self, cx: int, cy: int, label: str) -> None:
        self.objects[self.next_id]     = (cx, cy, label)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id: int) -> None:
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    # ------------------------------------------------------------------
    def update(self, detections: list[tuple]) -> dict:
        """
        Update tracks from the current frame's detections.

        Args:
            detections : list of (x, y, w, h, label)

        Returns:
            dict { id: (cx, cy, label) } for all active tracks
        """
        # Compute centroids for this frame's detections
        if detections:
            input_centroids = np.array(
                [(x + w // 2, y + h // 2) for (x, y, w, h, _) in detections],
                dtype=int,
            )
            input_labels = [lbl for (_, _, _, _, lbl) in detections]
        else:
            input_centroids = np.empty((0, 2), dtype=int)
            input_labels    = []

        # No tracks yet → register every detection
        if len(self.objects) == 0:
            for i, (cx, cy) in enumerate(input_centroids):
                self._register(cx, cy, input_labels[i])
            return self.objects

        # No detections → age all existing tracks
        if len(input_centroids) == 0:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects

        # Build cost matrix
        obj_ids    = list(self.objects.keys())
        obj_cents  = np.array(
            [(cx, cy) for (cx, cy, _) in self.objects.values()], dtype=int
        )
        D = _cdist(obj_cents, input_centroids)   # (M existing, N new)

        # Greedy matching: sort existing tracks by their closest new detection
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            obj_id                   = obj_ids[row]
            cx, cy                   = input_centroids[col]
            self.objects[obj_id]     = (cx, cy, input_labels[col])
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched new detections → new tracks
        for col in range(len(input_centroids)):
            if col not in used_cols:
                cx, cy = input_centroids[col]
                self._register(cx, cy, input_labels[col])

        # Unmatched existing tracks → age them
        for row in range(len(obj_ids)):
            if row not in used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        return self.objects

    # ------------------------------------------------------------------
    def count(self) -> tuple[int, int]:
        """Return (n_pedestrians, n_vehicles) of currently active tracks."""
        n_ped = sum(1 for (_, _, lbl) in self.objects.values() if lbl == "pedestrian")
        n_veh = sum(1 for (_, _, lbl) in self.objects.values() if lbl == "vehicle")
        return n_ped, n_veh
