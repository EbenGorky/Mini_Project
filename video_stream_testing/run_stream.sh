#!/usr/bin/env bash
# Start the test MJPEG stream server on port 4000
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VIDEO_FILE="${1:-site_test.mp4}"

echo "[INFO] Starting video stream server on port 4000..."
echo "[INFO] Streaming video: $VIDEO_FILE"
exec ../vison_/venv/bin/python server.py --video "$VIDEO_FILE" --port 4000

