#!/usr/bin/env bash
# Stop both the stream server and Helmet Vision
fuser -k 4000/tcp 2>/dev/null || true
pkill -f "python.*(server\.py|app\.py)" 2>/dev/null || true
echo "[INFO] All Helmet Vision and Stream Server processes stopped."

