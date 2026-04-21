"""
Session-level detection statistics.

Accumulates per-frame measurements and produces a summary at the end of
a run. The summary is suitable for inclusion in an academic report.
"""
import time


class DetectionStats:
    """Accumulate frame-by-frame metrics during a session."""

    def __init__(self):
        self.total_frames           = 0
        self.frames_with_detections = 0
        self.total_pedestrian_obs   = 0   # sum of per-frame pedestrian counts
        self.total_vehicle_obs      = 0
        self._frame_times_ms: list  = []
        self._t_start               = time.perf_counter()

    # ------------------------------------------------------------------
    def update(self, n_ped: int, n_veh: int, frame_time_ms: float) -> None:
        """
        Record one processed frame.

        Args:
            n_ped         : active pedestrian tracks this frame
            n_veh         : active vehicle tracks this frame
            frame_time_ms : wall-clock time to process this frame (ms)
        """
        self.total_frames += 1
        if n_ped + n_veh > 0:
            self.frames_with_detections += 1
        self.total_pedestrian_obs += n_ped
        self.total_vehicle_obs    += n_veh
        self._frame_times_ms.append(frame_time_ms)

    # ------------------------------------------------------------------
    def detection_rate(self) -> float:
        """Percentage of frames that contained at least one detection."""
        if self.total_frames == 0:
            return 0.0
        return 100.0 * self.frames_with_detections / self.total_frames

    def avg_frame_time_ms(self) -> float:
        if not self._frame_times_ms:
            return 0.0
        return sum(self._frame_times_ms) / len(self._frame_times_ms)

    def avg_fps(self) -> float:
        elapsed = time.perf_counter() - self._t_start
        return self.total_frames / elapsed if elapsed > 0 else 0.0

    def max_frame_time_ms(self) -> float:
        return max(self._frame_times_ms) if self._frame_times_ms else 0.0

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print a formatted end-of-session report to stdout."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("  SESSION SUMMARY")
        print(sep)
        print(f"  Total frames processed  : {self.total_frames}")
        print(f"  Average FPS             : {self.avg_fps():.1f}")
        print(f"  Avg frame time          : {self.avg_frame_time_ms():.1f} ms")
        print(f"  Worst frame time        : {self.max_frame_time_ms():.1f} ms")
        print(f"  Frames with detections  : {self.frames_with_detections}"
              f"  ({self.detection_rate():.1f} %)")
        print(f"  Total pedestrian obs.   : {self.total_pedestrian_obs}")
        print(f"  Total vehicle obs.      : {self.total_vehicle_obs}")
        print(f"{sep}\n")
