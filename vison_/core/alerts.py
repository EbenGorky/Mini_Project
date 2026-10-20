"""
Alerts and Event Logging Module for Helmet Vision (Milestone 5).
Handles:
1. Audio alarms / beepers on High risk triggers (PulseAudio / PipeWire / system fallback).
2. Incident logging to structured CSV and JSON files.
3. Timestamped snapshot recording on high-risk breaches.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
import struct
import subprocess
import threading
import time
import wave
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .hazards import HazardZone, RiskLevel


class AlertManager:
    """
    Manages real-time audio warnings, event logging, and snapshot persistence.
    """

    def __init__(self, config: Dict[str, Any], base_dir: Optional[str] = None) -> None:
        self.config = config
        alerts_cfg = config.get("alerts", {})

        self.beep_enabled = alerts_cfg.get("beep_on_high_risk", True)
        self.logging_enabled = alerts_cfg.get("log_events", True)
        self.save_snapshots = alerts_cfg.get("save_snapshots", True)
        self.beep_interval = float(alerts_cfg.get("beep_interval_sec", 3.0))
        self.snapshot_cooldown = float(alerts_cfg.get("snapshot_cooldown_sec", 5.0))

        # Root logging directories
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(base_dir, alerts_cfg.get("log_dir", "logs"))
        self.snapshots_dir = os.path.join(self.log_dir, "snapshots")

        os.makedirs(self.log_dir, exist_ok=True)
        if self.save_snapshots:
            os.makedirs(self.snapshots_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, "events.csv")
        self.json_path = os.path.join(self.log_dir, "events.jsonl")

        self._init_csv()

        # Audio state
        self._last_beep_time = 0.0
        self._last_snapshot_time = 0.0
        self._last_logged_risk = RiskLevel.LOW
        self._last_logged_msg: Optional[str] = None
        self._audio_player = self._find_audio_player()
        self._beep_wav_path = self._generate_alert_wav()

    def _init_csv(self) -> None:
        """Create CSV file with headers if it does not exist."""
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "iso_time", "risk_level", "message", "active_zones_count", "snapshot_path"])

    def _find_audio_player(self) -> Optional[str]:
        """Detect available command line audio player on system."""
        for player in ("pw-play", "paplay", "aplay"):
            try:
                res = subprocess.run(["which", player], capture_output=True, text=True, check=False)
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.strip()
            except Exception:
                continue
        return None

    def _generate_alert_wav(self) -> str:
        """Generate a short 880Hz alert tone in temp directory."""
        wav_path = os.path.join(self.log_dir, "alert_tone.wav")
        if os.path.isfile(wav_path):
            return wav_path

        try:
            sample_rate = 22050
            duration = 0.18  # seconds
            freq = 880.0     # A5 tone
            num_samples = int(sample_rate * duration)

            with wave.open(wav_path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                frames = []
                for i in range(num_samples):
                    t = i / sample_rate
                    # Exponential decay envelope
                    env = math.exp(-3.5 * t)
                    val = int(32767.0 * 0.65 * env * math.sin(2.0 * math.pi * freq * t))
                    frames.append(struct.pack("<h", max(-32768, min(32767, val))))
                w.writeframes(b"".join(frames))
            return wav_path
        except Exception:
            return ""

    def process(
        self,
        overall_risk: RiskLevel,
        active_zones: List[HazardZone],
        suggestion_msg: Optional[str],
        annotated_frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        Evaluate safety state, trigger audio alarms, and log incidents.
        """
        now = time.time()
        is_high = overall_risk == RiskLevel.HIGH
        is_escalation = (overall_risk != self._last_logged_risk and is_high) or (suggestion_msg != self._last_logged_msg and is_high)

        # 1. Audio Alarm on High Risk
        if is_high and self.beep_enabled:
            if now - self._last_beep_time >= self.beep_interval:
                self._last_beep_time = now
                self._play_beep_async()

        # 2. Incident Logging and Snapshot on Escalation
        if is_escalation:
            snapshot_file = ""
            if self.save_snapshots and annotated_frame is not None and (now - self._last_snapshot_time >= self.snapshot_cooldown):
                self._last_snapshot_time = now
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"event_{timestamp_str}_{overall_risk.value}.jpg"
                snapshot_file = os.path.join(self.snapshots_dir, filename)
                try:
                    cv2.imwrite(snapshot_file, annotated_frame)
                except Exception as exc:
                    print(f"[ALERTS WARN] Could not save snapshot: {exc}")
                    snapshot_file = ""

            self._log_event(overall_risk, suggestion_msg or "", len(active_zones), snapshot_file)

        self._last_logged_risk = overall_risk
        self._last_logged_msg = suggestion_msg

    def _play_beep_async(self) -> None:
        """Play alert sound in a non-blocking daemon thread."""
        def _play():
            try:
                if self._audio_player and os.path.isfile(self._beep_wav_path):
                    subprocess.run(
                        [self._audio_player, self._beep_wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    # Terminal bell fallback
                    print("\a", end="", flush=True)
            except Exception:
                pass

        threading.Thread(target=_play, name="AudioAlertThread", daemon=True).start()

    def _log_event(
        self,
        risk: RiskLevel,
        msg: str,
        zone_count: int,
        snapshot_path: str,
    ) -> None:
        """Append incident record to CSV and JSONL."""
        if not self.logging_enabled:
            return

        now_dt = datetime.datetime.now()
        ts = now_dt.timestamp()
        iso = now_dt.isoformat()

        # CSV
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([f"{ts:.2f}", iso, risk.value, msg, zone_count, snapshot_path])
        except Exception as exc:
            print(f"[ALERTS WARN] Failed to write CSV event: {exc}")

        # JSON Lines
        try:
            event_obj = {
                "timestamp": ts,
                "iso_time": iso,
                "risk_level": risk.value,
                "message": msg,
                "active_zones": zone_count,
                "snapshot": snapshot_path,
            }
            with open(self.json_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_obj) + "\n")
        except Exception as exc:
            print(f"[ALERTS WARN] Failed to write JSON event: {exc}")
