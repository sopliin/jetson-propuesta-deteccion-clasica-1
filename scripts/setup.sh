#!/usr/bin/env bash
# =============================================================
# Setup script — Classical CV Traffic Monitor
# Target: Jetson Orin Nano, JetPack 6.2.2
# =============================================================
# JetPack already ships with a CUDA-enabled OpenCV build.
# We use --system-site-packages so the venv inherits it.
# =============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "  Classical CV Traffic Monitor — Setup"
echo "  Project root: $PROJECT_ROOT"
echo "=================================================="

# --- Python check
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO] Python version: $PY_VER"

# --- Create venv (inherit system packages for JetPack OpenCV)
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv) with --system-site-packages..."
    python3 -m venv .venv --system-site-packages
else
    echo "[INFO] Virtual environment already exists."
fi

source .venv/bin/activate

# --- Upgrade pip silently
pip install --upgrade pip -q

# --- Install dependencies (numpy + pyyaml; OpenCV from system)
echo "[INFO] Installing numpy and pyyaml..."
pip install numpy pyyaml -q

# --- Verify OpenCV
echo ""
echo "[INFO] Verifying OpenCV installation..."
python3 - <<'EOF'
import cv2
print(f"  OpenCV version : {cv2.__version__}")
try:
    n = cv2.cuda.getCudaEnabledDeviceCount()
    if n > 0:
        info = cv2.cuda.DeviceInfo(0)
        print(f"  CUDA device    : {info.name()} (device 0) — GPU mode available")
    else:
        print("  CUDA           : no CUDA-enabled GPU found — CPU mode only")
except AttributeError:
    print("  CUDA           : OpenCV built without CUDA support — CPU mode only")
EOF

# --- Create assets directory if missing
mkdir -p assets/videos results/annotations

echo ""
echo "=================================================="
echo "  Setup complete."
echo ""
echo "  Activate env  :  source .venv/bin/activate"
echo "  Run (video)   :  python src/main.py --source video --input assets/videos/test1.mp4"
echo "  Run (CSI cam) :  python src/main.py --source csi"
echo "=================================================="
