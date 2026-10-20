#!/usr/bin/env bash
# Helmet Vision launcher script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment 'venv' not found in $SCRIPT_DIR."
    echo "Creating venv..."
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
fi

# Ensure GUI platform compatibility on Wayland/X11 desktops
if [ -n "$WAYLAND_DISPLAY" ] && [ -z "$DISPLAY" ]; then
    export DISPLAY=":0"
fi
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

source venv/bin/activate

if [ $# -eq 0 ]; then
    exec python app.py --ip localhost:4000
else
    exec python app.py "$@"
fi
