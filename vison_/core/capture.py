"""
Video capture and stream management module for Helmet Vision.
Handles threaded MJPEG acquisition from ESP32-CAM, auto-reconnection,
latency tracking, and video playback mode.
"""

from __future__ import annotations

import ipaddress
import os
import re
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class StreamState(str, Enum):
    """Current state of the video stream."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"


def validate_ip_address(ip_str: str) -> Tuple[bool, str]:
    """
    Validate whether the input string is a valid IPv4 address, localhost, or host:port.
    Returns:
        (is_valid, cleaned_host_or_error_msg)
    """
    raw = ip_str.strip()
    if not raw:
        return False, "IP address cannot be empty"

    cleaned = re.sub(r"^https?://", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.split("/")[0]

    if ":" in cleaned:
        host, port_str = cleaned.split(":", 1)
        try:
            p = int(port_str)
            if not (1 <= p <= 65535):
                return False, f"Port {p} out of range (1-65535)"
        except ValueError:
            return False, f"Invalid port in '{raw}'"
    else:
        host = cleaned.strip()

    if host.lower() == "localhost":
        return True, raw

    try:
        ipaddress.IPv4Address(host)
        return True, raw
    except ValueError:
        return False, f"Invalid host/IP format '{raw}'"


def build_stream_url(ip: str, port: int = 81, path: str = "/stream") -> str:
    """
    Construct full stream URL from IP/host, port, and path.
    Handles if IP string already has scheme, port, or path.
    """
    raw = ip.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw

    cleaned = re.sub(r"^https?://", "", raw, flags=re.IGNORECASE)
    parts = cleaned.split("/", 1)
    host_port = parts[0]
    custom_path = f"/{parts[1]}" if len(parts) > 1 and parts[1] else path

    if ":" in host_port:
        host, custom_port_str = host_port.split(":", 1)
        try:
            custom_port = int(custom_port_str)
            clean_path = custom_path if custom_path.startswith("/") else f"/{custom_path}"
            return f"http://{host}:{custom_port}{clean_path}"
        except ValueError:
            pass

    clean_path = path if path.startswith("/") else f"/{path}"
    return f"http://{host_port}:{port}{clean_path}"


class StreamCapture:
    """
    Threaded video capture manager that maintains only the freshest frame,
    tracks streaming statistics, and automatically reconnects on signal loss.
    """

    TROUBLESHOOTING_HINTS = [
        "1. Check that the ESP32-CAM is powered and its status LED is lit.",
        "2. Ensure this laptop and the ESP32 are connected to the SAME Wi-Fi network.",
        "3. Confirm the camera IP address via router or Serial Monitor.",
        "4. Confirm port 81 (CameraWebServer default) and path /stream are open.",
        "5. Test opening the stream directly in a browser to check firmware output.",
    ]

    def __init__(
        self,
        source: str,
        is_video_file: bool = False,
        port: int = 81,
        path: str = "/stream",
        reconnect_interval_sec: float = 3.0,
        connection_timeout_sec: float = 5.0,
        loop_video: bool = True,
    ) -> None:
        self.port = port
        self.path = path
        self.reconnect_interval_sec = reconnect_interval_sec
        self.connection_timeout_sec = connection_timeout_sec
        self.loop_video = loop_video

        self._is_video_file = is_video_file
        self._source_raw = source
        self._active_url = ""
        self._current_ip: Optional[str] = None

        # Thread synchronization
        self._frame_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # Frame state
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_index: int = 0

        # Performance and status tracking
        self._state: StreamState = StreamState.DISCONNECTED
        self._last_error: str = ""
        self._reconnect_attempts: int = 0
        self._next_retry_timestamp: float = 0.0

        # FPS calculation
        self._fps: float = 0.0
        self._frame_count_for_fps: int = 0
        self._last_fps_calc_time: float = time.time()

        # Initialize source
        self._setup_source(source, is_video_file)

    def _setup_source(self, source: str, is_video_file: bool) -> None:
        """Resolve IP or path into active URL."""
        self._source_raw = source
        self._is_video_file = is_video_file

        if is_video_file or os.path.isfile(source):
            self._is_video_file = True
            self._active_url = source
            self._current_ip = None
        else:
            is_valid, res = validate_ip_address(source)
            if is_valid:
                self._current_ip = res
                self._active_url = build_stream_url(res, self.port, self.path)
            else:
                self._current_ip = source
                self._active_url = source

    @property
    def state(self) -> StreamState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: StreamState, error_msg: str = "") -> None:
        with self._state_lock:
            self._state = state
            if error_msg:
                self._last_error = error_msg

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic dictionary for UI and monitoring."""
        with self._state_lock:
            current_state = self._state
            last_err = self._last_error
            attempts = self._reconnect_attempts
            next_retry = max(0.0, self._next_retry_timestamp - time.time())

        with self._frame_lock:
            fps = self._fps
            latency_ms = (
                (time.time() - self._latest_timestamp) * 1000.0
                if self._latest_timestamp > 0
                else 0.0
            )
            has_frame = self._latest_frame is not None
            frame_idx = self._frame_index

        return {
            "state": current_state,
            "is_video_file": self._is_video_file,
            "ip": self._current_ip,
            "url": self._active_url,
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 1),
            "has_frame": has_frame,
            "frame_index": frame_idx,
            "reconnect_attempts": attempts,
            "next_retry_in_sec": round(next_retry, 1),
            "last_error": last_err,
            "troubleshooting_hints": self.TROUBLESHOOTING_HINTS,
        }

    def start(self) -> None:
        """Start the background stream worker thread."""
        if self._running.is_set():
            return

        self._running.set()
        self._capture_thread = threading.Thread(
            target=self._worker_loop,
            name="StreamCaptureWorker",
            daemon=True,
        )
        self._capture_thread.start()

    def stop(self) -> None:
        """Stop worker thread and release camera resources."""
        self._running.clear()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._set_state(StreamState.DISCONNECTED, "Stream stopped")

    def change_source(self, new_source: str, is_video_file: bool = False) -> Tuple[bool, str]:
        """
        Dynamically update IP or video file without restarting application.
        """
        if not is_video_file:
            is_valid, res = validate_ip_address(new_source)
            if not is_valid:
                return False, res

        self.stop()
        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = 0.0
            self._frame_count_for_fps = 0
            self._fps = 0.0

        self._setup_source(new_source, is_video_file)
        self._reconnect_attempts = 0
        self._next_retry_timestamp = 0.0
        self.start()
        return True, "Connecting to new source"

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Non-blocking retrieval of the latest available frame.
        Always returns only the freshest frame to prevent latency build-up.

        Returns:
            (has_new_frame, frame_bgr, capture_timestamp)
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None, 0.0
            # Return copy or reference for drawing
            return True, self._latest_frame.copy(), self._latest_timestamp

    def _worker_loop(self) -> None:
        """Main background loop handling connection, capture, and retries."""
        while self._running.is_set():
            # Validate IP if source is camera
            if not self._is_video_file and self._current_ip:
                is_valid, msg = validate_ip_address(self._current_ip)
                if not is_valid:
                    self._set_state(StreamState.ERROR, msg)
                    time.sleep(1.0)
                    continue

            self._set_state(
                StreamState.CONNECTING
                if self._reconnect_attempts == 0
                else StreamState.RECONNECTING,
                f"Connecting to {self._active_url}...",
            )

            cap = self._open_capture()
            if cap is None or not cap.isOpened():
                self._handle_connection_failure(cap)
                continue

            # Connected successfully
            self._set_state(StreamState.CONNECTED, "")
            self._reconnect_attempts = 0

            # Video playback rate limiter if local video file
            target_delay = 0.0
            if self._is_video_file:
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                target_delay = 1.0 / cap_fps if cap_fps and cap_fps > 0 else 1.0 / 30.0

            # Frame read loop
            consecutive_failures = 0
            while self._running.is_set():
                loop_start = time.time()
                ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if self._is_video_file and self.loop_video:
                        # Video file finished: rewind to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        consecutive_failures = 0
                        time.sleep(0.01)
                        continue

                    # If stream dropped or ended
                    if consecutive_failures >= 3:
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                now = time.time()

                # Update latest frame buffer (low latency)
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_timestamp = now
                    self._frame_index += 1
                    self._frame_count_for_fps += 1

                    # Recalculate FPS every 0.5s
                    elapsed = now - self._last_fps_calc_time
                    if elapsed >= 0.5:
                        self._fps = self._frame_count_for_fps / elapsed
                        self._frame_count_for_fps = 0
                        self._last_fps_calc_time = now

                # Sleep to maintain native video playback speed if local file
                if self._is_video_file and target_delay > 0:
                    spent = time.time() - loop_start
                    to_sleep = target_delay - spent
                    if to_sleep > 0:
                        time.sleep(to_sleep)

            # Cap read loop ended
            if cap is not None:
                cap.release()

            if self._running.is_set():
                # Stream dropped while running
                self._handle_connection_failure(None)

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open VideoCapture instance with low latency settings."""
        try:
            # Set OpenCV to prefer minimal buffering
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|max_delay;500000"
            )
            cap = cv2.VideoCapture(self._active_url)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception as exc:
            self._set_state(StreamState.ERROR, str(exc))
            return None

    def _handle_connection_failure(self, cap: Optional[cv2.VideoCapture]) -> None:
        """Handle stream loss or unreachable camera with auto-reconnect."""
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        self._reconnect_attempts += 1
        source_desc = self._current_ip if self._current_ip else self._active_url
        err_msg = (
            f"Cannot reach camera at {source_desc}"
            if not self._is_video_file
            else f"Failed to read video file '{self._source_raw}'"
        )
        self._set_state(StreamState.OFFLINE, err_msg)

        # Wait reconnect interval before next attempt
        self._next_retry_timestamp = time.time() + self.reconnect_interval_sec
        while self._running.is_set() and time.time() < self._next_retry_timestamp:
            time.sleep(0.1)

