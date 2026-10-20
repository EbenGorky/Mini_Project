"""
Overlay and UI rendering module for Helmet Vision.
Draws live status badges, performance metrics (FPS, latency),
interactive IP input box, Connect/Disconnect buttons,
and the camera-offline troubleshooting card.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from .capture import StreamState


class OverlayUI:
    """
    Renders status HUD, interactive control bar, and diagnostic splash card
    onto OpenCV frames. Supports mouse clicks for GUI interaction.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        initial_ip: str = "192.168.1.50",
        on_connect_callback: Optional[Callable[[str], None]] = None,
        on_disconnect_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        display_cfg = config.get("display", {})
        self.target_width = display_cfg.get("target_width", 960)
        self.target_height = display_cfg.get("target_height", 540)
        self.show_stats = display_cfg.get("show_stats", True)
        self.show_control_bar = display_cfg.get("show_control_bar", True)

        # Callbacks
        self.on_connect_callback = on_connect_callback
        self.on_disconnect_callback = on_disconnect_callback

        # Interactive IP text field
        self.ip_text = initial_ip
        self.input_focused = False
        self._cursor_visible = True
        self._last_cursor_blink = time.time()

        # UI element bounding boxes (x1, y1, x2, y2)
        self.input_box_rect = (0, 0, 0, 0)
        self.connect_btn_rect = (0, 0, 0, 0)
        self.disconnect_btn_rect = (0, 0, 0, 0)

        # Colors (BGR)
        theme = display_cfg.get("theme", {})
        self.c_bg_dark = tuple(theme.get("bg_dark", [24, 24, 24]))
        self.c_panel_bg = tuple(theme.get("panel_bg", [40, 40, 40]))
        self.c_border = tuple(theme.get("border", [70, 70, 70]))
        self.c_text = tuple(theme.get("text", [240, 240, 240]))
        self.c_accent = tuple(theme.get("accent", [255, 140, 0]))
        self.c_green = tuple(theme.get("green", [50, 205, 50]))
        self.c_amber = tuple(theme.get("amber", [0, 191, 255]))
        self.c_red = tuple(theme.get("red", [50, 50, 240]))

        # Render FPS tracker
        self._render_fps = 0.0
        self._render_frame_count = 0
        self._last_render_fps_time = time.time()

    def set_ip_text(self, ip: str) -> None:
        self.ip_text = ip

    def update_render_fps(self) -> None:
        """Call once per frame in display loop to track UI FPS."""
        now = time.time()
        self._render_frame_count += 1
        elapsed = now - self._last_render_fps_time
        if elapsed >= 0.5:
            self._render_fps = self._render_frame_count / elapsed
            self._render_frame_count = 0
            self._last_render_fps_time = now

    def draw(
        self,
        frame: Optional[np.ndarray],
        status: Dict[str, Any],
        detector: Optional[Any] = None,
        detections: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Draw all overlays onto frame (or generate splash screen if frame is None).
        """
        self.update_render_fps()

        # Generate canvas
        if frame is None:
            canvas = np.full(
                (self.target_height, self.target_width, 3),
                self.c_bg_dark,
                dtype=np.uint8,
            )
        else:
            # Resize frame to target display resolution
            h, w = frame.shape[:2]
            if w != self.target_width or h != self.target_height:
                canvas = cv2.resize(frame, (self.target_width, self.target_height))
            else:
                canvas = frame.copy()

        # Update cursor blink
        if time.time() - self._last_cursor_blink > 0.5:
            self._cursor_visible = not self._cursor_visible
            self._last_cursor_blink = time.time()

        state = status.get("state", StreamState.DISCONNECTED)

        # Draw AI detections if active
        if frame is not None and detector is not None and detections:
            # Rescale detection boxes if target resolution differs from frame
            h_orig, w_orig = frame.shape[:2]
            scale_x = self.target_width / w_orig if w_orig > 0 else 1.0
            scale_y = self.target_height / h_orig if h_orig > 0 else 1.0

            scaled_dets = []
            if scale_x != 1.0 or scale_y != 1.0:
                for d in detections:
                    from .detector import Detection
                    x1, y1, x2, y2 = d.box
                    sx1 = int(x1 * scale_x)
                    sy1 = int(y1 * scale_y)
                    sx2 = int(x2 * scale_x)
                    sy2 = int(y2 * scale_y)
                    scaled_dets.append(
                        Detection(
                            box=(sx1, sy1, sx2, sy2),
                            confidence=d.confidence,
                            class_id=d.class_id,
                            class_name=d.class_name,
                            category=d.category,
                            color=d.color,
                        )
                    )
            else:
                scaled_dets = detections

            detector.draw_detections(canvas, scaled_dets)

        # If camera offline or connecting with no frame, draw troubleshooting card
        if frame is None or state in (StreamState.OFFLINE, StreamState.ERROR, StreamState.CONNECTING, StreamState.RECONNECTING):
            self._draw_splash_card(canvas, status)

        # Draw top HUD stats banner
        if self.show_stats:
            self._draw_header_hud(canvas, status)

        # Draw bottom interactive control bar
        if self.show_control_bar:
            self._draw_control_bar(canvas, status)

        return canvas

    def _draw_header_hud(self, canvas: np.ndarray, status: Dict[str, Any]) -> None:
        """Draw top banner with status badge, FPS, latency, and source."""
        h, w = canvas.shape[:2]
        hud_h = 44

        # Semi-transparent top bar
        sub_overlay = canvas[0:hud_h, 0:w].copy()
        cv2.rectangle(sub_overlay, (0, 0), (w, hud_h), (18, 18, 18), -1)
        cv2.addWeighted(sub_overlay, 0.85, canvas[0:hud_h, 0:w], 0.15, 0, canvas[0:hud_h, 0:w])
        cv2.line(canvas, (0, hud_h), (w, hud_h), self.c_border, 1)

        # Title
        cv2.putText(
            canvas,
            "HELMET VISION",
            (14, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Status badge pill
        state = status.get("state", StreamState.DISCONNECTED)
        badge_text, badge_color = self._get_badge_info(state)

        badge_x = 185
        badge_w = 135
        badge_y1 = 10
        badge_y2 = 34
        cv2.rectangle(canvas, (badge_x, badge_y1), (badge_x + badge_w, badge_y2), badge_color, -1)
        cv2.putText(
            canvas,
            badge_text,
            (badge_x + 10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )

        # Stream source
        is_video = status.get("is_video_file", False)
        source_label = (
            f"FILE: {status.get('url', '')}"
            if is_video
            else f"ESP32: {status.get('url', '')}"
        )
        if len(source_label) > 38:
            source_label = source_label[:35] + "..."
        cv2.putText(
            canvas,
            source_label,
            (335, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Performance stats on right
        stream_fps = status.get("fps", 0.0)
        render_fps = round(self._render_fps, 1)
        latency_ms = status.get("latency_ms", 0.0)
        inference_ms = status.get("inference_ms", 0.0)

        ai_str = f" | AI: {inference_ms:.1f}ms" if inference_ms > 0 else ""
        stats_str = f"Stream: {stream_fps} fps | UI: {render_fps} fps | Latency: {latency_ms} ms{ai_str}"
        (sw, _), _ = cv2.getTextSize(stats_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        stats_x = max(w - sw - 15, 600)
        cv2.putText(
            canvas,
            stats_str,
            (stats_x, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            self.c_accent,
            1,
            cv2.LINE_AA,
        )

    def _draw_control_bar(self, canvas: np.ndarray, status: Dict[str, Any]) -> None:
        """
        Draw bottom control bar with IP input box, Connect, Disconnect,
        and keyboard shortcuts.
        """
        h, w = canvas.shape[:2]
        bar_h = 50
        y_start = h - bar_h

        # Semi-transparent background
        sub_overlay = canvas[y_start:h, 0:w].copy()
        cv2.rectangle(sub_overlay, (0, 0), (w, bar_h), (22, 22, 22), -1)
        cv2.addWeighted(sub_overlay, 0.90, canvas[y_start:h, 0:w], 0.10, 0, canvas[y_start:h, 0:w])
        cv2.line(canvas, (0, y_start), (w, y_start), self.c_border, 1)

        # Input box label
        label_text = "Target IP:"
        cv2.putText(
            canvas,
            label_text,
            (15, y_start + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # Text input box
        box_x1 = 95
        box_y1 = y_start + 10
        box_w = 200
        box_h = 30
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h
        self.input_box_rect = (box_x1, box_y1, box_x2, box_y2)

        # Input box border color depends on focus
        border_col = self.c_accent if self.input_focused else self.c_border
        bg_col = (50, 50, 50) if self.input_focused else (35, 35, 35)

        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), bg_col, -1)
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), border_col, 2 if self.input_focused else 1)

        # Text inside input box
        display_text = self.ip_text
        if self.input_focused and self._cursor_visible:
            display_text += "|"

        cv2.putText(
            canvas,
            display_text,
            (box_x1 + 10, box_y1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Connect Button
        btn_w = 95
        btn_h = 30
        c_btn_x1 = box_x2 + 12
        c_btn_y1 = box_y1
        c_btn_x2 = c_btn_x1 + btn_w
        c_btn_y2 = c_btn_y1 + btn_h
        self.connect_btn_rect = (c_btn_x1, c_btn_y1, c_btn_x2, c_btn_y2)

        cv2.rectangle(canvas, (c_btn_x1, c_btn_y1), (c_btn_x2, c_btn_y2), self.c_green, -1)
        cv2.putText(
            canvas,
            "CONNECT",
            (c_btn_x1 + 13, c_btn_y1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )

        # Disconnect Button
        d_btn_x1 = c_btn_x2 + 10
        d_btn_y1 = box_y1
        d_btn_x2 = d_btn_x1 + btn_w + 15
        d_btn_y2 = d_btn_y1 + btn_h
        self.disconnect_btn_rect = (d_btn_x1, d_btn_y1, d_btn_x2, d_btn_y2)

        cv2.rectangle(canvas, (d_btn_x1, d_btn_y1), (d_btn_x2, d_btn_y2), (70, 70, 70), -1)
        cv2.putText(
            canvas,
            "DISCONNECT",
            (d_btn_x1 + 10, d_btn_y1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Shortcut hints on the right
        hints_str = "Enter: Connect | Click box to type IP | D: Disconnect | Q: Quit"
        (hw, _), _ = cv2.getTextSize(hints_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        hint_x = max(w - hw - 15, d_btn_x2 + 15)
        cv2.putText(
            canvas,
            hints_str,
            (hint_x, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    def _draw_splash_card(self, canvas: np.ndarray, status: Dict[str, Any]) -> None:
        """
        Draw an informative centered diagnostic card when stream is offline
        or attempting reconnection.
        """
        h, w = canvas.shape[:2]
        state = status.get("state", StreamState.DISCONNECTED)
        target_ip = status.get("ip") or status.get("url", "camera")
        reconnect_in = status.get("next_retry_in_sec", 0.0)
        attempts = status.get("reconnect_attempts", 0)

        # Card dimensions
        card_w = min(680, w - 60)
        card_h = 320
        cx1 = (w - card_w) // 2
        cy1 = (h - card_h) // 2 - 10
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h

        # Semi-transparent overlay backdrop
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (32, 32, 32), -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)

        # Card border
        border_col = self.c_red if state in (StreamState.OFFLINE, StreamState.ERROR) else self.c_amber
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), border_col, 2)

        # Top banner of card
        banner_h = 42
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy1 + banner_h), border_col, -1)

        title_text = f"CANNOT REACH CAMERA AT {target_ip}" if state == StreamState.OFFLINE else "CONNECTING TO STREAM..."
        cv2.putText(
            canvas,
            title_text,
            (cx1 + 20, cy1 + 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Status subtitle / countdown
        if state in (StreamState.OFFLINE, StreamState.RECONNECTING):
            sub_text = f"Status: Camera Offline (Attempt #{attempts}) | Retrying automatically in {reconnect_in:.1f}s..."
        else:
            sub_text = f"Status: Attempting connection to http://{target_ip}:81/stream..."

        cv2.putText(
            canvas,
            sub_text,
            (cx1 + 25, cy1 + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self.c_amber,
            1,
            cv2.LINE_AA,
        )

        # Troubleshooting divider line
        cv2.line(canvas, (cx1 + 25, cy1 + 90), (cx2 - 25, cy1 + 90), self.c_border, 1)

        # Troubleshooting heading
        cv2.putText(
            canvas,
            "Troubleshooting Checklist:",
            (cx1 + 25, cy1 + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        hints = status.get("troubleshooting_hints", [])
        hint_y = cy1 + 145
        for hint in hints:
            cv2.putText(
                canvas,
                hint,
                (cx1 + 35, hint_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            hint_y += 28

        # Note at bottom of card
        footer_note = "Tip: You can edit the IP below and click CONNECT at any time."
        cv2.putText(
            canvas,
            footer_note,
            (cx1 + 25, cy2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            self.c_accent,
            1,
            cv2.LINE_AA,
        )

    def _get_badge_info(self, state: StreamState) -> Tuple[str, Tuple[int, int, int]]:
        """Map stream state to UI pill text and BGR color."""
        if state == StreamState.CONNECTED:
            return "LIVE STREAM", self.c_green
        elif state == StreamState.CONNECTING:
            return "CONNECTING...", self.c_amber
        elif state == StreamState.RECONNECTING:
            return "RECONNECTING", self.c_amber
        elif state == StreamState.OFFLINE:
            return "CAMERA OFFLINE", self.c_red
        elif state == StreamState.ERROR:
            return "STREAM ERROR", self.c_red
        else:
            return "DISCONNECTED", (120, 120, 120)

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Process mouse clicks for input box focus and button triggers."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Check input box click
        ix1, iy1, ix2, iy2 = self.input_box_rect
        if ix1 <= x <= ix2 and iy1 <= y <= iy2:
            self.input_focused = True
            return
        else:
            self.input_focused = False

        # Check Connect button click
        cx1, cy1, cx2, cy2 = self.connect_btn_rect
        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            if self.on_connect_callback:
                self.on_connect_callback(self.ip_text.strip())
            return

        # Check Disconnect button click
        dx1, dy1, dx2, dy2 = self.disconnect_btn_rect
        if dx1 <= x <= dx2 and dy1 <= y <= dy2:
            if self.on_disconnect_callback:
                self.on_disconnect_callback()
            return

    def handle_key(self, key_code: int) -> Optional[str]:
        """
        Handle keyboard input for the IP text box and shortcuts.

        Returns:
            "connect" if Enter was pressed
            "disconnect" if 'd' or 'D' pressed outside text box
            "quit" if 'q', 'Q', or Esc pressed
            None otherwise
        """
        if key_code == -1:
            return None

        # Esc = 27
        if key_code == 27:
            return "quit"

        # If text box is focused: capture all typing
        if self.input_focused:
            # Enter key (13 or 10)
            if key_code in (13, 10):
                self.input_focused = False
                return "connect"

            # Backspace key (8)
            elif key_code == 8:
                if len(self.ip_text) > 0:
                    self.ip_text = self.ip_text[:-1]

            # Allowed IP characters (digits, dot, colon, slash)
            elif 32 <= key_code <= 126:
                char = chr(key_code)
                if char in "0123456789.:/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-":
                    if len(self.ip_text) < 40:
                        self.ip_text += char

            return None

        # If not focused, check shortcuts
        char = chr(key_code).lower() if 0 <= key_code <= 255 else ""
        if char == "q":
            return "quit"
        elif char == "d":
            return "disconnect"
        elif char == "c":
            # Focus the input box
            self.input_focused = True
            return None
        elif key_code in (13, 10):
            # Enter connects with current IP
            return "connect"

        return None

