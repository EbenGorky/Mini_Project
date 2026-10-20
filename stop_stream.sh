#!/usr/bin/env bash
# Stop the video stream server on port 4000
fuser -k 4000/tcp 2>/dev/null || true
pkill -f "python.*server\.py" 2>/dev/null || true
echo "[INFO] Video stream server stopped."

