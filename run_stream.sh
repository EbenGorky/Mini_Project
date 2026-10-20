#!/usr/bin/env bash
# Start the video stream server on port 4000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/video_stream_testing"
exec ./run_stream.sh "$@"
