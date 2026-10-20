"""
YOLO Object Detection Engine for Helmet Vision.
Detects workers, PPE, machinery/vehicles, signs, and barriers in real time
with GPU acceleration (NVIDIA CUDA) and automatic CPU fallback.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class HazardCategory(str, Enum):
    """Categorization of detected objects for safety rules."""
    WORKER = "WORKER"
    PPE_HELMET = "PPE_HELMET"
    PPE_VEST = "PPE_VEST"
    NO_PPE = "NO_PPE"
    MACHINERY = "MACHINERY"
    CRANE = "CRANE"
    SUSPENDED_LOAD = "SUSPENDED_LOAD"
    SIGN = "SIGN"
    BARRIER = "BARRIER"
    HAZARD_TAPE = "HAZARD_TAPE"
    OTHER = "OTHER"


@dataclass
class Detection:
    """Represents a single detected bounding box and classification."""
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int
    class_name: str
    category: HazardCategory
    color: Tuple[int, int, int]     # BGR color
    track_id: Optional[int] = None

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2

    @property
    def bottom_center(self) -> Tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, self.box[3]


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-Over-Union between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class TrackedTarget:
    """Internal tracker target with temporal persistence across frames."""
    track_id: int
    box: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    category: HazardCategory
    color: Tuple[int, int, int]
    frames_since_seen: int = 0
    hit_streak: int = 1

    def to_detection(self) -> Detection:
        return Detection(
            box=self.box,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
            category=self.category,
            color=self.color,
            track_id=self.track_id,
        )


class YOLODetector:
    """
    Manages YOLO model inference, GPU/CPU execution, class categorization,
    and visual detection box rendering.
    """

    # Category Color Mapping (BGR)
    CATEGORY_COLORS = {
        HazardCategory.WORKER: (255, 200, 0),       # Cyan-ish blue
        HazardCategory.PPE_HELMET: (50, 220, 50),   # Green
        HazardCategory.PPE_VEST: (50, 220, 50),     # Green
        HazardCategory.NO_PPE: (30, 30, 255),       # Bright Red
        HazardCategory.MACHINERY: (30, 70, 240),    # Orange-Red
        HazardCategory.CRANE: (0, 140, 255),        # Deep Orange
        HazardCategory.SUSPENDED_LOAD: (0, 0, 255), # Pure Red (Threat)
        HazardCategory.SIGN: (0, 215, 255),         # Yellow
        HazardCategory.BARRIER: (0, 140, 255),      # Orange
        HazardCategory.HAZARD_TAPE: (0, 215, 255),  # Yellow
        HazardCategory.OTHER: (180, 180, 180),      # Gray
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        det_cfg = config.get("detection", {})

        # Settings
        self.model_path = det_cfg.get("model_path", "models/yolov8n.pt")
        self.conf_threshold = float(det_cfg.get("confidence_threshold", 0.40))
        self.iou_threshold = float(det_cfg.get("iou_threshold", 0.45))
        self.imgsz = int(det_cfg.get("imgsz", 640))
        self.frame_skip = int(det_cfg.get("frame_skip", 0))

        # Ensure local Ultralytics config dir is writable
        os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath(".ultralytics"))

        # Setup compute device (GPU vs CPU fallback)
        self.device = self._resolve_device(det_cfg.get("device", "cuda:0"))

        # Load YOLO model
        self.model = self._load_model(self.model_path)

        # Performance metrics
        self.last_inference_ms: float = 0.0
        self._last_detections: List[Detection] = []
        self._frame_counter: int = 0

        # Temporal object tracker state (eliminates split-second flicker)
        self.tracked_targets: List[TrackedTarget] = []
        self.next_track_id: int = 1
        self.max_missed_frames = int(det_cfg.get("max_missed_frames", 8))
        self.min_iou_match = float(det_cfg.get("min_iou_match", 0.20))

        # Temporal Confirmation Filtering (kills random 1-frame glitches)
        temp_cfg = config.get("temporal_filtering", {})
        self.min_confirmation_hits = int(temp_cfg.get("min_confirmation_frames", 2))
        self.fast_track_conf = float(temp_cfg.get("high_conf_fast_track", 0.55))

    def _resolve_device(self, requested_device: str) -> str:
        """Check CUDA availability and return active torch device."""
        try:
            import torch
            if "cuda" in requested_device and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[DETECTOR INFO] GPU acceleration active: {gpu_name} ({requested_device})")
                return requested_device
            else:
                print(f"[DETECTOR INFO] CUDA not requested or unavailable. Using CPU fallback.")
                return "cpu"
        except Exception as exc:
            print(f"[DETECTOR WARN] Could not inspect PyTorch device ({exc}). Falling back to CPU.")
            return "cpu"

    def _load_model(self, model_path: str) -> Any:
        """Load Ultralytics YOLO model."""
        from ultralytics import YOLO

        # Fallback to yolov8n.pt if configured file doesn't exist
        actual_path = model_path
        if not os.path.isfile(actual_path):
            fallback_local = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt")
            if os.path.isfile(fallback_local):
                actual_path = fallback_local
            else:
                actual_path = "yolov8n.pt"

        print(f"[DETECTOR INFO] Loading YOLO model from '{actual_path}' on {self.device}...")
        model = YOLO(actual_path)
        print(f"[DETECTOR INFO] Model loaded successfully ({len(model.names)} classes).")
        return model

    def categorize_class(
        self,
        class_name: str,
        box: Optional[Tuple[int, int, int, int]] = None,
        frame_h: Optional[int] = None,
    ) -> HazardCategory:
        """Map raw class name (COCO or custom construction) to hazard category."""
        name = class_name.lower().replace("-", "_").replace(" ", "_")

        # Worker / Person
        if name in ("person", "worker", "pedestrian"):
            return HazardCategory.WORKER

        # PPE (Helmets, Vests)
        if "no_helmet" in name or "without_helmet" in name:
            return HazardCategory.NO_PPE
        if "helmet" in name or "hard_hat" in name or "hardhat" in name:
            return HazardCategory.PPE_HELMET

        if "no_vest" in name or "without_vest" in name:
            return HazardCategory.NO_PPE
        if "vest" in name or "safety_vest" in name:
            return HazardCategory.PPE_VEST

        # Cranes & Overhead Rigging
        if any(c in name for c in ("crane", "tower_crane", "mobile_crane", "derrick", "boom_crane")):
            return HazardCategory.CRANE

        # Suspended Load / Hook / Rigging
        if any(s in name for s in ("suspended", "load", "hook", "rigging", "sling", "cargo_lift")):
            return HazardCategory.SUSPENDED_LOAD

        # High elevation crane structures & suspended loads aloft in mid-air
        if box and frame_h:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # 1. Crane vertical mast / boom (e.g. left tower structure)
            if (any(c in name for c in ("crane", "bird")) or (x1 < 120 and y2 > frame_h * 0.50)):
                if y1 < frame_h * 0.35 and (x1 < 250 or "crane" in name):
                    return HazardCategory.CRANE

            # 2. Suspended load (aloft in the air, starts high y1 < 0.50*H, hanging down above floor)
            if any(s in name for s in ("kite", "airplane", "sports_ball", "sports ball", "boat", "load", "hook", "rigging", "container", "truck", "car", "box")):
                if y1 < frame_h * 0.50 and y2 < frame_h * 0.88:
                    return HazardCategory.SUSPENDED_LOAD

        # Machinery & Vehicles (on ground)
        if any(mach in name for mach in (
            "car", "truck", "bus", "train", "vehicle", "excavator",
            "forklift", "loader", "tractor", "bulldozer", "roller"
        )):
            return HazardCategory.MACHINERY

        # Signs
        if any(s in name for s in ("sign", "stop_sign", "traffic_light", "warning", "danger", "caution")):
            return HazardCategory.SIGN

        # Barriers / Cones
        if any(b in name for b in ("cone", "traffic_cone", "barrier", "bollard", "fence")):
            return HazardCategory.BARRIER

        return HazardCategory.OTHER

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on frame. Supports frame skipping for high throughput.

        Returns:
            List of Detection objects.
        """
        self._frame_counter += 1

        # Frame skipping optimization
        if self.frame_skip > 0 and (self._frame_counter % (self.frame_skip + 1) != 0):
            return self._last_detections

        t_start = time.time()

        try:
            # Run YOLO inference
            results = self.model.predict(
                source=frame,
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )

            detections: List[Detection] = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)

                    h_frame = frame.shape[0]
                    for box, conf, cls_id in zip(xyxy, confs, classes):
                        cls_name = self.model.names.get(cls_id, f"cls_{cls_id}")
                        x1, y1, x2, y2 = map(int, box)
                        category = self.categorize_class(cls_name, box=(x1, y1, x2, y2), frame_h=h_frame)
                        color = self.CATEGORY_COLORS.get(category, (180, 180, 180))

                        detections.append(
                            Detection(
                                box=(x1, y1, x2, y2),
                                confidence=float(conf),
                                class_id=cls_id,
                                class_name=cls_name,
                                category=category,
                                color=color,
                            )
                        )

            # Update temporal tracker to maintain continuous tracking across momentary frame drops
            self._update_tracks(detections)

            # Temporal Confirmation Filter: Only output confirmed targets (kills 1-frame blips)
            # Confident detections or overhead crane/load threats confirm immediately
            active_detections = []
            for t in self.tracked_targets:
                is_confirmed = (
                    t.hit_streak >= self.min_confirmation_hits
                    or t.confidence >= self.fast_track_conf
                    or t.category in (HazardCategory.CRANE, HazardCategory.SUSPENDED_LOAD)
                )
                if is_confirmed and t.frames_since_seen <= self.max_missed_frames:
                    active_detections.append(t.to_detection())

            self.last_inference_ms = (time.time() - t_start) * 1000.0
            self._last_detections = active_detections
            return active_detections

        except Exception as exc:
            print(f"[DETECTOR ERROR] Inference failed: {exc}")
            self.last_inference_ms = 0.0
            return self._last_detections

    def _update_tracks(self, new_detections: List[Detection]) -> None:
        """
        Match incoming detections with existing tracks using IOU.
        Smooths coordinates and holds objects alive across momentary misses.
        """
        unmatched_new = set(range(len(new_detections)))
        unmatched_existing = set(range(len(self.tracked_targets)))

        matches = []
        for e_idx, existing in enumerate(self.tracked_targets):
            best_iou = 0.0
            best_n_idx = -1
            for n_idx in unmatched_new:
                new_d = new_detections[n_idx]
                cat_match = (
                    existing.category == new_d.category
                    or (
                        existing.category in (HazardCategory.CRANE, HazardCategory.SUSPENDED_LOAD)
                        and new_d.category in (HazardCategory.CRANE, HazardCategory.SUSPENDED_LOAD)
                    )
                )
                if cat_match:
                    iou = compute_iou(existing.box, new_d.box)
                    if iou > best_iou:
                        best_iou = iou
                        best_n_idx = n_idx

            if best_iou >= self.min_iou_match and best_n_idx != -1:
                matches.append((e_idx, best_n_idx))
                unmatched_existing.discard(e_idx)
                unmatched_new.discard(best_n_idx)

        # Update matched targets (smooth position coordinates)
        for e_idx, n_idx in matches:
            t = self.tracked_targets[e_idx]
            d = new_detections[n_idx]
            smooth_box = (
                int(0.75 * d.box[0] + 0.25 * t.box[0]),
                int(0.75 * d.box[1] + 0.25 * t.box[1]),
                int(0.75 * d.box[2] + 0.25 * t.box[2]),
                int(0.75 * d.box[3] + 0.25 * t.box[3]),
            )
            t.box = smooth_box
            t.confidence = d.confidence
            t.class_name = d.class_name
            t.category = d.category
            t.color = d.color
            t.frames_since_seen = 0
            t.hit_streak += 1

        # Hold unmatched existing targets alive across momentary misses
        for e_idx in unmatched_existing:
            t = self.tracked_targets[e_idx]
            t.frames_since_seen += 1
            t.confidence = max(0.20, t.confidence * 0.95)

        # Spawn new targets for newly detected objects
        for n_idx in unmatched_new:
            d = new_detections[n_idx]
            new_target = TrackedTarget(
                track_id=self.next_track_id,
                box=d.box,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                category=d.category,
                color=d.color,
                frames_since_seen=0,
                hit_streak=1,
            )
            self.next_track_id += 1
            self.tracked_targets.append(new_target)

        # Prune candidates:
        # 1. Unconfirmed candidate (hit_streak < min_hits) that was missed: drop immediately (kills 1-frame glitches!)
        # 2. Confirmed candidate: allow persistence up to max_missed_frames to eliminate flicker
        survivors = []
        for t in self.tracked_targets:
            is_confirmed = (
                t.hit_streak >= self.min_confirmation_hits
                or t.confidence >= self.fast_track_conf
                or t.category in (HazardCategory.CRANE, HazardCategory.SUSPENDED_LOAD)
            )
            if is_confirmed:
                if t.frames_since_seen <= self.max_missed_frames:
                    survivors.append(t)
            else:
                if t.frames_since_seen == 0:
                    survivors.append(t)
        self.tracked_targets = survivors

    def draw_detections(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Draw sleek, high-visibility bounding boxes with corner brackets
        and confidence badges onto canvas.
        """
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = det.color
            label = f"{det.class_name.upper()} {int(det.confidence * 100)}%"

            # 1. Bounding box rectangle (thin border)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # 2. Sleek Corner Brackets
            corner_len = max(8, min(20, (x2 - x1) // 4, (y2 - y1) // 4))
            thick = 3
            # Top-left
            cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, thick)
            cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, thick)
            # Top-right
            cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, thick)
            cv2.line(canvas, (x2, y1), (x2, y2 - corner_len if y2 - corner_len > y1 else y1 + corner_len), color, thick)
            # Bottom-left
            cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, thick)
            cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, thick)
            # Bottom-right
            cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, thick)
            cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, thick)

            # 3. Label Badge Pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            badge_y1 = max(0, y1 - th - 6)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = x1 + tw + 10

            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Label text in dark contrast
            cv2.putText(
                canvas,
                label,
                (badge_x1 + 5, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

        return canvas

