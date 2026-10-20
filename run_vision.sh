#!/usr/bin/env bash
# Start Helmet Vision AI safety monitor
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/vison_"
exec ./run_vision.sh "$@"
