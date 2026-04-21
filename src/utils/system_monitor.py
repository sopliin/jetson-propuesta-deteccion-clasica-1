"""
System resource monitor — CPU, RAM, and GPU usage for terminal display.

CPU and RAM use psutil (cross-platform, works on Jetson and dev machine).
GPU usage tries the Jetson sysfs path first, then falls back to 0.
"""

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# Sysfs paths where Jetson exposes GPU load (varies by SoC/JetPack version).
# Value is typically 0–1000; we divide by 10 to get a percentage.
_JETSON_GPU_PATHS = [
    "/sys/devices/gpu.0/load",
    "/sys/class/devfreq/17000000.ga10b/device/load",
    "/sys/class/devfreq/57000000.ga10b/device/load",
]


def cpu_total() -> float:
    """Total CPU usage as a percentage (0–100)."""
    if not _HAS_PSUTIL:
        return 0.0
    return psutil.cpu_percent()


def cpu_per_core() -> list[float]:
    """Per-core CPU usage as a list of percentages."""
    if not _HAS_PSUTIL:
        return []
    return psutil.cpu_percent(percpu=True)


def ram_usage() -> tuple[float, float, float]:
    """Return (used_GB, total_GB, percent)."""
    if not _HAS_PSUTIL:
        return 0.0, 0.0, 0.0
    vm = psutil.virtual_memory()
    return vm.used / 1024**3, vm.total / 1024**3, vm.percent


def gpu_percent() -> float:
    """
    GPU load as a percentage.
    Reads the Jetson sysfs load file; returns 0.0 on non-Jetson machines.
    """
    for path in _JETSON_GPU_PATHS:
        try:
            with open(path) as fh:
                return int(fh.read().strip()) / 10.0
        except (FileNotFoundError, ValueError, PermissionError):
            continue
    return 0.0


def format_status(fps_alert_threshold: float = 15.0,
                  cpu_alert_threshold: float = 90.0,
                  ram_alert_threshold: float = 85.0) -> tuple[str, list[str]]:
    """
    Build a compact multi-field status string and a list of alert strings.

    Returns:
        (status_line, alerts)
    """
    cpu   = cpu_total()
    cores = cpu_per_core()
    gpu   = gpu_percent()
    ram_used, ram_total, ram_pct = ram_usage()

    core_str = "  ".join(f"C{i}:{v:.0f}%" for i, v in enumerate(cores)) if cores else ""
    ram_str  = f"{ram_used:.1f}/{ram_total:.1f}GB({ram_pct:.0f}%)"

    line = (
        f"CPU:{cpu:5.1f}%"
        + (f"  [{core_str}]" if core_str else "")
        + f"  GPU:{gpu:5.1f}%"
        + f"  RAM:{ram_str}"
    )

    alerts: list[str] = []
    if cpu >= cpu_alert_threshold:
        alerts.append(f"[ALERT] CPU at {cpu:.0f}%")
    if ram_pct >= ram_alert_threshold:
        alerts.append(f"[ALERT] RAM at {ram_pct:.0f}%")

    return line, alerts
