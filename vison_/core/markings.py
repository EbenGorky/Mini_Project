"""
Markings and Sign Detection module for Helmet Vision (Milestone 4).
Detects:
1. Yellow/black diagonal hazard tape and ground boundary markings via HSV segmentation.
2. Traffic cones and safety barriers via color and geometry.
3. Construction warning, danger, and no-entry signs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection, HazardCategory


def compute_marking_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection-over-Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class MarkingType(str, Enum):
    HAZARD_TAPE = "hazard_tape"
    FLOOR_MARKING = "floor_marking"
    CONE = "cone"
    BARRIER = "barrier"
    SIGN_NO_ENTRY = "sign_no_entry"
    SIGN_DANGER = "sign_danger"
    SIGN_CAUTION = "sign_caution"


@dataclass
class MarkingDetection:
    """Represents a detected construction marking, tape, or sign."""
    marking_type: MarkingType
    box: Tuple[int, int, int, int]           # (x1, y1, x2, y2)
    polygon: np.ndarray                      # (N, 2) contour points
    confidence: float
    label: str
    is_proximity_hazard: bool = False        # True if close to wearer path (bottom of frame)


@dataclass
class TrackedMarking:
    """Internal candidate with temporal confirmation across consecutive frames."""
    track_id: int
    box: Tuple[int, int, int, int]
    polygon: np.ndarray
    marking_type: MarkingType
    confidence: float
    label: str
    is_proximity_hazard: bool
    hit_streak: int = 1
    missed_frames: int = 0

    def to_detection(self) -> MarkingDetection:
        return MarkingDetection(
            marking_type=self.marking_type,
            box=self.box,
            polygon=self.polygon,
            confidence=self.confidence,
            label=self.label,
            is_proximity_hazard=self.is_proximity_hazard,
        )


class MarkingDetector:
    """
    Classical Computer Vision detector complementing YOLO:
    - HSV color thresholding for yellow/black striped hazard tape.
    - Safety orange cone/barrier contour analysis.
    - Sign geometry & color segmentation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        m_cfg = config.get("markings", {})
        self.enabled = bool(m_cfg.get("enabled", False))
        tape_cfg = m_cfg.get("hazard_tape", {})

        # Yellow HSV range for hazard tape / safety markings
        self.yellow_lower = np.array(tape_cfg.get("yellow_hsv_lower", [18, 90, 80]), dtype=np.uint8)
        self.yellow_upper = np.array(tape_cfg.get("yellow_hsv_upper", [36, 255, 255]), dtype=np.uint8)

        # Orange HSV range for traffic cones / bollards
        self.orange_lower = np.array([5, 140, 100], dtype=np.uint8)
        self.orange_upper = np.array([17, 255, 255], dtype=np.uint8)

        # Red HSV ranges for Stop / Danger / No-Entry signs (wraps around 0/180)
        self.red1_lower = np.array([0, 120, 70], dtype=np.uint8)
        self.red1_upper = np.array([10, 255, 255], dtype=np.uint8)
        self.red2_lower = np.array([170, 120, 70], dtype=np.uint8)
        self.red2_upper = np.array([180, 255, 255], dtype=np.uint8)

        self.min_tape_area = int(tape_cfg.get("min_contour_area", 400))
        self.min_aspect_ratio = float(tape_cfg.get("min_aspect_ratio", 2.0))
        self.proximity_ratio = float(tape_cfg.get("proximity_y_ratio", 0.60))

        # Morphological kernels
        self.kernel_tape = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        self.kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Temporal Confirmation Filter (kills random 1-frame glitches)
        temp_cfg = config.get("temporal_filtering", {})
        self.min_hits = int(temp_cfg.get("marking_min_hits", 2))
        self.tracked_markings: List[TrackedMarking] = []
        self._next_track_id = 1

    def detect(self, frame: np.ndarray) -> List[MarkingDetection]:
        """
        Run marking and sign detection on frame.
        Applies temporal confirmation so random 1-frame glitches are silently dropped.
        """
        if not self.enabled or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        # Ignore dark frames during scene transitions
        if float(frame.mean()) < 5.0:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_candidates: List[MarkingDetection] = []

        # 1. Detect Yellow/Black Hazard Tape
        raw_candidates.extend(self._detect_hazard_tape(hsv, w, h))

        # 2. Detect Traffic Cones / Barriers
        raw_candidates.extend(self._detect_cones_and_barriers(hsv, w, h))

        # 3. Detect Signs (Red Stop/No-Entry, Caution)
        raw_candidates.extend(self._detect_safety_signs(hsv, frame, w, h))

        # 4. Temporal Confirmation: requires candidate to repeat in consecutive frames
        self._update_candidate_tracks(raw_candidates)

        # Return ONLY confirmed markings (hit_streak >= min_hits)
        confirmed = [
            t.to_detection()
            for t in self.tracked_markings
            if t.hit_streak >= self.min_hits and t.missed_frames == 0
        ]
        return confirmed

    def _update_candidate_tracks(self, raw_candidates: List[MarkingDetection]) -> None:
        """
        Match incoming candidates with temporal tracks.
        Eliminates single-frame blips and smooths real markings.
        """
        unmatched_new = set(range(len(raw_candidates)))
        unmatched_existing = set(range(len(self.tracked_markings)))

        matches = []
        for e_idx, existing in enumerate(self.tracked_markings):
            best_iou = 0.0
            best_n_idx = -1
            for n_idx in unmatched_new:
                cand = raw_candidates[n_idx]
                if cand.marking_type == existing.marking_type:
                    iou = compute_marking_iou(existing.box, cand.box)
                    if iou > best_iou:
                        best_iou = iou
                        best_n_idx = n_idx

            if best_iou >= 0.15 and best_n_idx != -1:
                matches.append((e_idx, best_n_idx))
                unmatched_existing.discard(e_idx)
                unmatched_new.discard(best_n_idx)

        # Update matched candidates
        for e_idx, n_idx in matches:
            t = self.tracked_markings[e_idx]
            cand = raw_candidates[n_idx]
            # Smooth coordinates
            t.box = (
                int(0.70 * cand.box[0] + 0.30 * t.box[0]),
                int(0.70 * cand.box[1] + 0.30 * t.box[1]),
                int(0.70 * cand.box[2] + 0.30 * t.box[2]),
                int(0.70 * cand.box[3] + 0.30 * t.box[3]),
            )
            t.polygon = cand.polygon
            t.confidence = max(t.confidence, cand.confidence)
            t.is_proximity_hazard = cand.is_proximity_hazard
            t.hit_streak += 1
            t.missed_frames = 0

        # Unmatched existing
        for e_idx in unmatched_existing:
            t = self.tracked_markings[e_idx]
            t.missed_frames += 1

        # Spawn new candidates (tentative state with hit_streak=1)
        for n_idx in unmatched_new:
            cand = raw_candidates[n_idx]
            self.tracked_markings.append(
                TrackedMarking(
                    track_id=self._next_track_id,
                    box=cand.box,
                    polygon=cand.polygon,
                    marking_type=cand.marking_type,
                    confidence=cand.confidence,
                    label=cand.label,
                    is_proximity_hazard=cand.is_proximity_hazard,
                    hit_streak=1,
                    missed_frames=0,
                )
            )
            self._next_track_id += 1

        # Prune candidates:
        # If tentative candidate (hit_streak < min_hits) is missed: DROP IMMEDIATELY (1-frame glitch!)
        # If confirmed candidate (hit_streak >= min_hits): allow up to 4 missed frames
        survivors = []
        for t in self.tracked_markings:
            if t.hit_streak < self.min_hits:
                if t.missed_frames == 0:
                    survivors.append(t)
            else:
                if t.missed_frames <= 4:
                    survivors.append(t)
        self.tracked_markings = survivors

    def _detect_hazard_tape(self, hsv: np.ndarray, w: int, h: int) -> List[MarkingDetection]:
        """Segment elongated yellow hazard tape boundaries."""
        results: List[MarkingDetection] = []
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Connect horizontal / diagonal tape strips
        morphed = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, self.kernel_tape, iterations=2)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, self.kernel_clean)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(1200, self.min_tape_area):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Aspect ratio check: tape is an elongated ribbon
            aspect = max(bw / max(1, bh), bh / max(1, bw))
            if aspect < 4.0 or max(bw, bh) < 130:
                continue

            # Check alternating stripes (tape must have alternating stripes, not solid dirt or paint)
            roi_v = hsv[y : y + bh, x : x + bw, 2]
            if roi_v.size == 0:
                continue

            prof = roi_v.mean(axis=0) if bw >= bh else roi_v.mean(axis=1)
            diff = np.diff(prof > prof.mean())
            transitions = np.count_nonzero(diff)
            if transitions < 3:
                continue

            is_near = (y + bh) > (h * self.proximity_ratio)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            poly = approx.reshape(-1, 2)

            results.append(
                MarkingDetection(
                    marking_type=MarkingType.HAZARD_TAPE,
                    box=(x, y, x + bw, y + bh),
                    polygon=poly,
                    confidence=min(0.95, 0.60 + aspect * 0.05),
                    label="HAZARD TAPE",
                    is_proximity_hazard=is_near,
                )
            )

        return results

    def _detect_cones_and_barriers(self, hsv: np.ndarray, w: int, h: int) -> List[MarkingDetection]:
        """Detect safety orange traffic cones and barricades with strict geometric filtering."""
        results: List[MarkingDetection] = []
        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        morphed = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, self.kernel_clean)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, self.kernel_clean, iterations=2)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(1, bw)

            if aspect >= 1.3 and bh >= 35:
                # Triangular tapering: top 25% width must be significantly narrower than base
                roi_mask = orange_mask[y : y + bh, x : x + bw]
                top_slice = roi_mask[: max(1, bh // 4), :]
                bot_slice = roi_mask[max(0, 3 * bh // 4) :, :]
                top_w = np.count_nonzero(top_slice.sum(axis=0))
                bot_w = np.count_nonzero(bot_slice.sum(axis=0))
                if bot_w > 0 and (top_w / bot_w) < 0.65:
                    is_near = (y + bh) > (h * self.proximity_ratio)
                    results.append(
                        MarkingDetection(
                            marking_type=MarkingType.CONE,
                            box=(x, y, x + bw, y + bh),
                            polygon=cnt.reshape(-1, 2),
                            confidence=0.80,
                            label="TRAFFIC CONE",
                            is_proximity_hazard=is_near,
                        )
                    )
            elif bw >= 140 and area > 1600:
                # Horizontal orange barricade
                is_near = (y + bh) > (h * self.proximity_ratio)
                results.append(
                    MarkingDetection(
                        marking_type=MarkingType.BARRIER,
                        box=(x, y, x + bw, y + bh),
                        polygon=cnt.reshape(-1, 2),
                        confidence=0.75,
                        label="SAFETY BARRIER",
                        is_proximity_hazard=is_near,
                    )
                )

        return results

    def _detect_safety_signs(
        self,
        hsv: np.ndarray,
        bgr: np.ndarray,
        w: int,
        h: int,
    ) -> List[MarkingDetection]:
        """Detect circular/octagonal red signs with internal white text contrast."""
        results: List[MarkingDetection] = []

        m1 = cv2.inRange(hsv, self.red1_lower, self.red1_upper)
        m2 = cv2.inRange(hsv, self.red2_lower, self.red2_upper)
        red_mask = cv2.bitwise_or(m1, m2)
        red_morphed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel_clean, iterations=2)

        contours, _ = cv2.findContours(red_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000 or area > (w * h * 0.30):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            ratio = bw / max(1, bh)
            if 0.80 <= ratio <= 1.25 and bw >= 32:
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / max(1.0, (perimeter * perimeter))

                if circularity > 0.60:
                    # Check internal contrast: real signs have white text/symbol in center
                    roi_hsv = hsv[y + bh // 4 : y + 3 * bh // 4, x + bw // 4 : x + 3 * bw // 4]
                    if roi_hsv.size > 0:
                        white_pixels = np.logical_and(roi_hsv[:, :, 2] > 180, roi_hsv[:, :, 1] < 50).sum()
                        white_ratio = white_pixels / (roi_hsv.shape[0] * roi_hsv.shape[1])
                        if white_ratio >= 0.04:
                            results.append(
                                MarkingDetection(
                                    marking_type=MarkingType.SIGN_NO_ENTRY,
                                    box=(x, y, x + bw, y + bh),
                                    polygon=cnt.reshape(-1, 2),
                                    confidence=0.88,
                                    label="NO ENTRY / DANGER SIGN",
                                    is_proximity_hazard=True,
                                )
                            )

        return results

    def to_generic_detections(self, markings: List[MarkingDetection]) -> List[Detection]:
        """Convert MarkingDetections to standard Detection objects for unified HUD display."""
        generic_dets: List[Detection] = []
        for m in markings:
            if m.marking_type == MarkingType.HAZARD_TAPE:
                cat = HazardCategory.BARRIER
                color = (0, 215, 255)  # Bright amber/yellow
            elif m.marking_type in (MarkingType.CONE, MarkingType.BARRIER):
                cat = HazardCategory.BARRIER
                color = (0, 140, 255)  # Orange
            else:
                cat = HazardCategory.SIGN
                color = (50, 50, 240)  # Red

            generic_dets.append(
                Detection(
                    box=m.box,
                    confidence=m.confidence,
                    class_id=-1,
                    class_name=m.label,
                    category=cat,
                    color=color,
                )
            )
        return generic_dets

