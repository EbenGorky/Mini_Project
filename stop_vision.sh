#!/usr/bin/env bash
# Stop Helmet Vision AI analyzer
pkill -f "python.*app\.py" 2>/dev/null || true
echo "[INFO] Helmet Vision stopped."

