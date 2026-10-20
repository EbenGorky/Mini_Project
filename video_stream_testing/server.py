#!/usr/bin/env python3
"""
Video Stream Testing Server
Streams an MP4 video file as an MJPEG HTTP stream at http://localhost:4000/stream
and interactive HTML viewer at http://localhost:4000/ with time reset button.

Usage:
    python server.py --video your_video.mp4 --port 4000
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2


class VideoStreamBroadcaster:
    """Reads video file in a loop and produces JPEG-encoded frames."""

    def __init__(self, video_path: str, target_fps: Optional[float] = None) -> None:
        self.video_path = video_path
        self.target_fps = target_fps
        self._current_jpeg: Optional[bytes] = None
        self._seek_requested: Optional[float] = None
        self.current_time_sec = 0.0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.fps = 30.0

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="VideoBroadcastLoop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def seek(self, seconds: float) -> None:
        """Request seeking video to specific timestamp in seconds."""
        with self._lock:
            self._seek_requested = max(0.0, float(seconds))

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._current_jpeg

    def _loop(self) -> None:
        while self._running.is_set():
            if not os.path.isfile(self.video_path):
                print(f"[STREAMER WARN] Video file '{self.video_path}' not found! Retrying in 2s...")
                time.sleep(2.0)
                continue

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"[STREAMER ERROR] Could not open video file '{self.video_path}'. Retrying in 2s...")
                time.sleep(2.0)
                continue

            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = self.target_fps or (cap_fps if cap_fps and cap_fps > 5 else 30.0)
            frame_delay = 1.0 / self.fps
            print(f"[STREAMER INFO] Streaming '{self.video_path}' at {self.fps:.1f} FPS (looping enabled)")

            while self._running.is_set():
                t_start = time.time()

                # Check if a seek was requested
                with self._lock:
                    if self._seek_requested is not None:
                        target_sec = self._seek_requested
                        self._seek_requested = None
                        target_frame = int(target_sec * self.fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        print(f"[STREAMER INFO] Seeked stream to {target_sec:.1f}s (frame {target_frame})")

                ret, frame = cap.read()

                if not ret or frame is None:
                    # Video finished: rewind to beginning for continuous looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue

                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                self.current_time_sec = pos_msec / 1000.0 if pos_msec > 0 else 0.0

                # Encode to JPEG
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    with self._lock:
                        self._current_jpeg = buffer.tobytes()
                    self.frame_count += 1

                elapsed = time.time() - t_start
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cap.release()


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests and serves multipart/x-mixed-replace MJPEG stream."""

    broadcaster: VideoStreamBroadcaster

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

    def do_GET(self) -> None:
        clean_path = self.path.split("?")[0]

        # Reset / Seek endpoint
        if clean_path in ("/reset13", "/seek"):
            query = self.path.split("?")[1] if "?" in self.path else ""
            target_sec = 13.6
            if "time=" in query:
                try:
                    for param in query.split("&"):
                        if param.startswith("time="):
                            target_sec = float(param.split("=")[1])
                except Exception:
                    target_sec = 13.6

            self.broadcaster.seek(target_sec)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(f'{{"status": "ok", "time": {target_sec}}}'.encode("utf-8"))
            return

        # Browser viewer landing page
        if clean_path == "/":
            html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Helmet Vision - Live Test Stream</title>
  <style>
    body {
      background-color: #121212;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 90vh;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 22px;
      color: #ff8c00;
      letter-spacing: 0.5px;
    }
    .badge {
      display: inline-block;
      background: #2e7d32;
      color: #fff;
      font-size: 12px;
      font-weight: bold;
      padding: 3px 8px;
      border-radius: 12px;
      margin-left: 8px;
    }
    p {
      margin: 0 0 16px 0;
      color: #888;
      font-size: 14px;
    }
    .stream-card {
      background: #1e1e1e;
      border: 1px solid #333;
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    img {
      display: block;
      max-width: 90vw;
      max-height: 70vh;
      border-radius: 6px;
      background: #000;
    }
    .controls-bar {
      margin-top: 14px;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    button {
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      padding: 9px 18px;
      border-radius: 6px;
      border: none;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .btn-reset {
      background: #ff8c00;
      color: #111;
    }
    .btn-reset:hover {
      background: #ffa733;
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: #333;
      color: #ddd;
      border: 1px solid #444;
    }
    .btn-secondary:hover {
      background: #444;
      color: #fff;
    }
    #status-msg {
      font-size: 13px;
      color: #4da6ff;
      opacity: 0;
      transition: opacity 0.3s ease;
      font-weight: 500;
    }
    .footer {
      margin-top: 16px;
      font-size: 13px;
      color: #777;
    }
    code {
      background: #252525;
      padding: 3px 6px;
      border-radius: 4px;
      color: #ffb86c;
    }
    a { color: #4da6ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>Helmet Vision - Video Stream Server <span class="badge">LIVE</span></h1>
  <p>Simulating ESP32-CAM MJPEG stream on <b>port 4000</b> &bull; Raw stream endpoint at <a href="/stream">/stream</a></p>

  <div class="stream-card">
    <img src="/stream" alt="Connecting to stream..." />
    <div class="controls-bar">
      <button class="btn-reset" onclick="seekTo(13.6)">⏪ Reset Crane (13.6s)</button>
      <button class="btn-secondary" onclick="seekTo(0)">⏮️ Restart 0s</button>
      <button class="btn-secondary" onclick="seekTo(30)">⏩ Jump to 30s</button>
      <button class="btn-secondary" onclick="seekTo(60)">⏩ Jump to 60s</button>
      <span id="status-msg"></span>
    </div>
  </div>

  <div class="footer">
    Next: Run Helmet Vision analysis with <code>python app.py --ip localhost:4000</code>
  </div>

  <script>
    function seekTo(sec) {
      fetch('/seek?time=' + sec)
        .then(res => res.json())
        .then(data => {
          const msg = document.getElementById('status-msg');
          msg.textContent = '⏱️ Stream reset to ' + data.time + 's!';
          msg.style.opacity = '1';
          setTimeout(() => { msg.style.opacity = '0'; }, 2500);
        })
        .catch(err => console.error('Seek error:', err));
    }
    // Keyboard shortcut 'r' or '1' to reset to crane lift (13.6s)
    window.addEventListener('keydown', (e) => {
      if (e.key === 'r' || e.key === '1') {
        seekTo(13.6);
      }
    });
  </script>
</body>
</html>"""
            encoded = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return

        # Raw MJPEG stream endpoint
        elif clean_path in ("/stream", "/video", "/mjpeg"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    jpeg = self.broadcaster.get_latest_jpeg()
                    if jpeg:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(1.0 / self.broadcaster.fps)
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found. Use /stream or /\n")

    def log_message(self, format: str, *args: object) -> None:
        return


def find_default_video() -> str:
    """Find any available video in current folder or fallback sample."""
    mp4_files = glob.glob("*.mp4")
    if mp4_files:
        if "site_test.mp4" in mp4_files:
            return "site_test.mp4"
        return mp4_files[0]
    fallback = os.path.join(os.path.dirname(__file__), "..", "vison_", "test_media", "sample_site.mp4")
    if os.path.isfile(fallback):
        return os.path.abspath(fallback)
    return "video.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MJPEG Stream Server (Port 4000)")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to .mp4 video file to stream (defaults to site_test.mp4)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4000,
        help="Port to stream on (default: 4000)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target streaming FPS (defaults to video native FPS or 30)",
    )
    args = parser.parse_args()

    video_path = args.video or find_default_video()

    broadcaster = VideoStreamBroadcaster(video_path=video_path, target_fps=args.fps)
    broadcaster.start()

    MJPEGStreamHandler.broadcaster = broadcaster
    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGStreamHandler)

    print("=" * 60)
    print(f"  MJPEG VIDEO STREAM TEST SERVER RUNNING")
    print("=" * 60)
    print(f" Source Video : {video_path}")
    print(f" Stream URL 1 : http://localhost:{args.port}/stream")
    print(f" Stream URL 2 : http://localhost:{args.port}/")
    print(f" To stop      : Press Ctrl+C")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STREAMER INFO] Shutting down stream server...")
    finally:
        broadcaster.stop()
        server.server_close()
        print("[STREAMER INFO] Stream server stopped.")


if __name__ == "__main__":
    main()
