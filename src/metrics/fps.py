"""
Rolling FPS counter.

Stores a fixed-size window of frame timestamps and computes
the average frame rate from the oldest to newest entry.
Using a rolling window (rather than a cumulative average)
means the FPS reading responds quickly to sudden slowdowns.
"""
import time
from collections import deque


class FPSCounter:
    """Compute rolling average FPS over the last `window` frames."""

    def __init__(self, window: int = 30):
        """
        Args:
            window : number of recent timestamps to keep.
                     30 frames ≈ 1 s at 30 FPS — a good balance between
                     responsiveness and stability.
        """
        self._times: deque = deque(maxlen=window)

    def tick(self) -> None:
        """Record a frame timestamp. Call once per processed frame."""
        self._times.append(time.perf_counter())

    def fps(self) -> float:
        """
        Return average FPS over the current window.
        Returns 0.0 until at least 2 ticks have been recorded.
        """
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
