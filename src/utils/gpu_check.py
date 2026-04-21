"""
GPU/CUDA availability check for Jetson Orin Nano.

On Jetson, OpenCV is compiled against the Tegra GPU (CUDA device 0).
This module checks whether that CUDA build is present and prints
a clear status that can be referenced in the academic report.
"""
import cv2


def check_gpu():
    """
    Detect whether OpenCV was built with CUDA support and a device is reachable.

    Returns:
        (gpu_available: bool, device_desc: str)
    """
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            info = cv2.cuda.DeviceInfo(0)
            name = info.name()
            return True, f"GPU: {name} (CUDA device 0)"
        return False, "CPU: no CUDA-enabled GPU detected"
    except AttributeError:
        # cv2.cuda module missing — OpenCV built without CUDA
        return False, "CPU: OpenCV built without CUDA support"
    except Exception as exc:
        return False, f"CPU: GPU check error ({exc})"


def print_device_status(gpu_available: bool, device_desc: str) -> None:
    """Print a clear, bordered status line at startup."""
    sep = "=" * 52
    print(sep)
    print(f"  Compute device : {device_desc}")
    print(f"  Inference mode : {'GPU-accelerated (CUDA)' if gpu_available else 'CPU only'}")
    print(sep)
