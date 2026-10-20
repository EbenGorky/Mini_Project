#!/usr/bin/env python3
"""
Helmet Vision - Main Application Entrypoint
Milestone 1: ESP32-CAM Stream Acquisition, Auto-Reconnect & Interactive UI

Usage:
    python app.py --ip 192.168.1.50
    python app.py --video path/to/sample.mp4
    python app.py --headless --duration 5  # Test run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

# Ensure GUI platform compatibility on Wayland/X11 desktops
if "WAYLAND_DISPLAY" in os.environ and not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false;qt.font.*=false")

import cv2
import yaml

from core.capture import (
    StreamCapture,
    StreamState,
    build_stream_url,
    validate_ip_address,
)
from core.detector import YOLODetector
from core.hazards import HazardEngine, RiskLevel
from core.markings import MarkingDetector
from core.alerts import AlertManager
from core.overlay import OverlayUI


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    if not os.path.isfile(config_path):
        print(f"[WARN] Config file '{config_path}' not found. Using defaults.")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"[ERROR] Failed to parse config file: {exc}")
            return {}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Helmet Vision: Construction Safety Video Analysis System"
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="IP address of the ESP32-CAM (e.g. 192.168.1.50)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file for testing without hardware (e.g. demo.mp4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml configuration file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override camera stream port (default from config or 81)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without opening GUI window (for server/testing)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds and then exit (useful for automated verification)",
    )
    return parser.parse_args()


class HelmetVisionApp:
    """Main application coordinator for Helmet Vision."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_config(args.config)

        # Extract camera config
        cam_cfg = self.config.get("camera", {})
        display_cfg = self.config.get("display", {})
        video_cfg = self.config.get("video", {})

        self.port = args.port or cam_cfg.get("port", 81)
        self.path = cam_cfg.get("path", "/stream")
        self.reconnect_interval = cam_cfg.get("reconnect_interval_sec", 3.0)
        self.loop_video = video_cfg.get("loop", True)

        self.window_name = display_cfg.get(
            "window_title", "Helmet Vision - Construction Safety Monitor"
        )
        self.headless = args.headless

        # Determine initial source
        if args.video:
            self.source = args.video
            self.is_video_file = True
            self.initial_ip = "192.168.1.50"
        elif args.ip:
            self.source = args.ip
            self.is_video_file = False
            self.initial_ip = args.ip
        else:
            default_ip = cam_cfg.get("default_ip", "192.168.1.50")
            self.source = default_ip
            self.is_video_file = False
            self.initial_ip = default_ip

        # Validate IP upfront if using IP mode
        if not self.is_video_file:
            is_valid, msg = validate_ip_address(self.initial_ip)
            if not is_valid:
                print(f"[WARN] IP address '{self.initial_ip}' format warning: {msg}")

        # Initialize Capture Engine
        self.capture = StreamCapture(
            source=self.source,
            is_video_file=self.is_video_file,
            port=self.port,
            path=self.path,
            reconnect_interval_sec=self.reconnect_interval,
            loop_video=self.loop_video,
        )

        # Initialize Overlay and UI
        self.ui = OverlayUI(
            config=self.config,
            initial_ip=self.initial_ip,
            on_connect_callback=self.on_connect_requested,
            on_disconnect_callback=self.on_disconnect_requested,
        )

        # Initialize YOLO Detector Engine (Milestone 2)
        print("[INFO] Initializing YOLO safety detector...")
        self.detector = YOLODetector(config=self.config)

        # Initialize Hazard & Drop Zone Analysis Engine (Milestone 3)
        print("[INFO] Initializing hazard & drop-zone analysis engine...")
        self.hazard_engine = HazardEngine(config=self.config)

        # Initialize Marking & Sign Detection Engine (Milestone 4)
        print("[INFO] Initializing markings & hazard-tape detector...")
        self.marking_detector = MarkingDetector(config=self.config)

        # Initialize Alerts & Incident Logging Manager (Milestone 5)
        print("[INFO] Initializing alerts & event logging manager...")
        self.alert_manager = AlertManager(config=self.config)

        self.running = False

    def on_connect_requested(self, new_ip: str) -> None:
        """Callback when user clicks Connect or hits Enter."""
        is_valid, cleaned_ip = validate_ip_address(new_ip)
        if not is_valid:
            print(f"[UI] Cannot connect: {cleaned_ip}")
            return

        print(f"[UI] Connecting to IP: {cleaned_ip} (port {self.port})...")
        self.ui.set_ip_text(cleaned_ip)
        self.capture.change_source(cleaned_ip, is_video_file=False)

    def on_disconnect_requested(self) -> None:
        """Callback when user clicks Disconnect."""
        print("[UI] Disconnecting from stream...")
        self.capture.stop()

    def run(self) -> None:
        """Start streaming and run UI loop."""
        print("=" * 60)
        print("  HELMET VISION - CONSTRUCTION SAFETY MONITOR")
        print("=" * 60)
        if self.is_video_file:
            print(f" Mode: Video Test Mode ({self.source})")
        else:
            url = build_stream_url(self.initial_ip, self.port, self.path)
            print(f" Mode: ESP32-CAM Stream ({url})")
            print(f" Auto-reconnect: every {self.reconnect_interval}s if disconnected")
        print(" Controls:")
        print("   - Click the IP text box in the window to change IP")
        print("   - Click CONNECT or press ENTER to connect")
        print("   - Press 'D' or click DISCONNECT to stop stream")
        print("   - Press 'Q' or ESC to quit")
        print("=" * 60)

        # Start capture thread
        self.capture.start()
        self.running = True

        # GUI Window setup (if not headless)
        if not self.headless:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(self.window_name, self.ui.handle_mouse)
            except Exception as e:
                print(f"[WARN] Could not initialize GUI window: {e}")
                print("[INFO] Falling back to headless console mode.")
                self.headless = True

        start_time = time.time()
        last_log_time = time.time()

        try:
            while self.running:
                now = time.time()
                if self.args.duration and (now - start_time) >= self.args.duration:
                    print(f"[INFO] Duration limit reached ({self.args.duration}s). Exiting.")
                    break

                # 1. Fetch latest frame (thread-safe, low latency)
                has_frame, frame, capture_ts = self.capture.get_latest_frame()

                # 2. Fetch diagnostic status
                status = self.capture.get_status()

                # 3. Run AI object detection on live frame (Milestone 2)
                detections = []
                hazards = []
                overall_risk = RiskLevel.LOW
                suggestion_msg = None

                if has_frame and frame is not None:
                    # Clear temporal state across dark scene transitions
                    if status.get("is_dark_scene"):
                        self.detector.reset()
                        self.marking_detector.reset()
                        self.hazard_engine.reset()

                    # Run YOLO object detection (Milestone 2)
                    yolo_dets = self.detector.detect(frame)
                    status["inference_ms"] = round(self.detector.last_inference_ms, 1)

                    # Run Markings & Sign detection (Milestone 4)
                    marking_dets = self.marking_detector.detect(frame)
                    generic_markings = self.marking_detector.to_generic_detections(marking_dets)
                    detections = yolo_dets + generic_markings

                    # Compute Danger Zones, Drop Zones & Safety Perimeters (Milestone 3 & 4)
                    hazards, overall_risk, suggestion_msg = self.hazard_engine.process(frame.shape, detections)
                    status["risk_level"] = overall_risk.value
                    status["active_hazards"] = len(hazards)

                # 5. Render overlays (AI bounding boxes, Drop Zones, HUD, Alert Banner, Control bar)
                rendered = self.ui.draw(
                    frame if has_frame else None,
                    status,
                    detector=self.detector,
                    detections=detections,
                    hazards=hazards,
                    overall_risk=overall_risk,
                    suggestion_msg=suggestion_msg,
                )

                # 6. Audio Alarms & Incident Logging (Milestone 5)
                if has_frame and frame is not None:
                    self.alert_manager.process(overall_risk, hazards, suggestion_msg, rendered)

                # 4. Display or log
                if not self.headless:
                    cv2.imshow(self.window_name, rendered)

                    # Poll keys (15ms wait allows ~60Hz UI response)
                    key = cv2.waitKey(15) & 0xFF
                    action = self.ui.handle_key(key)

                    if action == "quit":
                        print("[INFO] Quit requested by user.")
                        break
                    elif action == "connect":
                        self.on_connect_requested(self.ui.ip_text)
                    elif action == "disconnect":
                        self.on_disconnect_requested()

                    # Check if window was closed via [X] button
                    try:
                        prop = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                        if prop < 1:
                            print("[INFO] Window closed by user.")
                            break
                    except Exception:
                        pass
                else:
                    # Headless mode: log status periodically
                    if now - last_log_time >= 1.0:
                        last_log_time = now
                        state = status["state"]
                        fps = status["fps"]
                        latency = status["latency_ms"]
                        attempts = status["reconnect_attempts"]
                        det_count = len(detections)
                        haz_count = len(hazards)
                        risk_str = overall_risk.value if overall_risk else "Low"
                        inf_ms = status.get("inference_ms", 0.0)
                        print(
                            f"[HEADLESS STATUS] State: {state.value:<12} | "
                            f"FPS: {fps:4.1f} | Latency: {latency:5.1f}ms | AI: {inf_ms:4.1f}ms | "
                            f"Dets: {det_count} | Hazards: {haz_count} | Risk: {risk_str:<6} | URL: {status['url']}"
                        )
                    time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupt received. Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release resources on exit."""
        self.running = False
        self.capture.stop()
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[INFO] Helmet Vision stopped cleanly.")


def main() -> None:
    args = parse_arguments()
    app = HelmetVisionApp(args)
    app.run()


if __name__ == "__main__":
    main()

