"""
Hazard Analysis and Risk Engine for Helmet Vision.
Computes dynamic danger zones, crane overhead suspended load drop zones,
machinery proximity zones, and temporal smoothing to prevent flicker.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection, HazardCategory


class RiskLevel(str, Enum):
    """Safety risk levels."""
    LOW = "Low"          # Green - Normal operation
    MEDIUM = "Medium"    # Amber - Caution / Hazard nearby
    HIGH = "High"        # Red - Imminent danger / Evacuate


@dataclass
class HazardZone:
    """Represents a computed spatial hazard zone with ground projection."""
    zone_id: str
    zone_type: str                   # "suspended_drop_zone", "crane_zone", "machinery_zone"
    risk_level: RiskLevel
    ground_polygon: np.ndarray       # (N, 2) array of ground footprint vertices
    source_box: Tuple[int, int, int, int] # (x1, y1, x2, y2) of the source object
    vertical_tracers: List[Tuple[Tuple[int, int], Tuple[int, int]]] # Projection lines
    title: str
    message: str
    is_breached: bool                # True if a worker or wearer is in the drop zone
    breached_workers: List[Detection]


class HazardEngine:
    """
    Evaluates detections against construction safety geometry rules:
    - Calculates the vertical 'drop zone' beneath cranes and suspended loads.
    - Projects perspective hazard polygons onto the ground.
    - Performs temporal smoothing across frames to eliminate flickering.
    - Detects worker intrusion into danger zones and escalates risk.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        hz_cfg = config.get("hazards", {})

        self.smoothing_frames = int(hz_cfg.get("smoothing_frames", 5))
        self.drop_margin = float(hz_cfg.get("suspended_load_safety_margin", 1.8))
        self.ground_horizon = float(hz_cfg.get("ground_horizon_ratio", 0.75))
        self.ground_bottom = float(hz_cfg.get("ground_bottom_ratio", 0.95))
        self.zone_hold_frames = int(hz_cfg.get("zone_hold_frames", 8))

        # Temporal smoothing history & zone persistence across momentary misses
        self._poly_history: Dict[str, deque] = {}
        self._last_active_zones: List[HazardZone] = []
        self._cached_zones: Dict[str, HazardZone] = {}
        self._zone_decay_counters: Dict[str, int] = {}

    def process(
        self,
        frame_shape: Tuple[int, int, int],
        detections: List[Detection],
    ) -> Tuple[List[HazardZone], RiskLevel, Optional[str]]:
        """
        Analyze detections and generate active hazard zones and risk assessment.

        Returns:
            (hazard_zones, max_risk_level, top_suggestion_message)
        """
        h, w = frame_shape[:2]
        active_zones: List[HazardZone] = []

        # Separate detections by category
        workers = [d for d in detections if d.category == HazardCategory.WORKER]
        cranes = [d for d in detections if d.category in (HazardCategory.CRANE, HazardCategory.SUSPENDED_LOAD)]
        machinery = [d for d in detections if d.category == HazardCategory.MACHINERY]
        barriers = [d for d in detections if d.category == HazardCategory.BARRIER]
        signs = [d for d in detections if d.category == HazardCategory.SIGN]
        no_ppe = [d for d in detections if d.category == HazardCategory.NO_PPE]

        # 1. Process Cranes & Suspended Loads (Overhead Drop Zone Threat)
        # Deduplicate overlapping sub-boxes (e.g. hook box inside suspended container box)
        unique_cranes: List[Detection] = []
        for c in sorted(cranes, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]), reverse=True):
            cx1, cy1, cx2, cy2 = c.box
            is_sub = False
            for uc in unique_cranes:
                ux1, uy1, ux2, uy2 = uc.box
                if cx1 >= ux1 - 15 and cy1 >= uy1 - 15 and cx2 <= ux2 + 15 and cy2 <= uy2 + 15:
                    is_sub = True
                    break
            if not is_sub:
                unique_cranes.append(c)

        crane_zones_found = False
        for idx, crane in enumerate(unique_cranes):
            zone = self._create_suspended_load_zone(crane, idx, w, h, workers)
            if zone:
                active_zones.append(zone)
                self._cached_zones[zone.zone_id] = zone
                self._zone_decay_counters[zone.zone_id] = 0
                crane_zones_found = True

        # Temporal persistence: If crane was briefly missed for 1-8 frames, hold the drop zone steady
        if not crane_zones_found and self._cached_zones:
            for zone_id, cached_zone in list(self._cached_zones.items()):
                counter = self._zone_decay_counters.get(zone_id, 0) + 1
                self._zone_decay_counters[zone_id] = counter
                if counter <= self.zone_hold_frames:
                    # Re-check worker intrusion against cached zone polygon
                    breached_workers = [
                        wk for wk in workers
                        if cv2.pointPolygonTest(cached_zone.ground_polygon, (float(wk.bottom_center[0]), float(wk.bottom_center[1])), False) >= 0
                    ]
                    cached_zone.breached_workers = breached_workers
                    cached_zone.is_breached = len(breached_workers) > 0
                    if cached_zone.is_breached:
                        cached_zone.risk_level = RiskLevel.HIGH
                    active_zones.append(cached_zone)
                else:
                    self._cached_zones.pop(zone_id, None)
                    self._zone_decay_counters.pop(zone_id, None)
                    self._poly_history.pop(zone_id, None)

        # 2. Process Ground Machinery Proximity Zones
        for idx, mach in enumerate(machinery):
            zone = self._create_machinery_zone(mach, idx, w, h, workers)
            if zone:
                active_zones.append(zone)

        # 3. Process Barriers, Cones & Hazard Tape Zones
        for idx, bar in enumerate(barriers):
            zone = self._create_barrier_zone(bar, idx, w, h, workers)
            if zone:
                active_zones.append(zone)

        # 4. Process Safety Signs (Danger, No Entry, Caution)
        for idx, sign in enumerate(signs):
            zone = self._create_sign_zone(sign, idx, w, h)
            if zone:
                active_zones.append(zone)

        # 5. Process Missing PPE Violations
        if no_ppe:
            ppe_zone = self._create_ppe_zone(no_ppe, w, h)
            if ppe_zone:
                active_zones.append(ppe_zone)

        # Determine overall risk level and primary suggestion message
        overall_risk = RiskLevel.LOW
        primary_msg: Optional[str] = None

        if active_zones:
            for z in active_zones:
                if z.risk_level == RiskLevel.HIGH:
                    overall_risk = RiskLevel.HIGH
                    primary_msg = z.message
                    break
                elif z.risk_level == RiskLevel.MEDIUM and overall_risk != RiskLevel.HIGH:
                    overall_risk = RiskLevel.MEDIUM
                    primary_msg = z.message

        self._last_active_zones = active_zones
        return active_zones, overall_risk, primary_msg

    def _create_suspended_load_zone(
        self,
        crane: Detection,
        idx: int,
        w: int,
        h: int,
        workers: List[Detection],
    ) -> Optional[HazardZone]:
        """
        Project vertical drop zone directly below the suspended load/crane.
        """
        x1, y1, x2, y2 = crane.box
        center_x = (x1 + x2) // 2
        load_w = max(40, x2 - x1)

        # Ground projection coordinates (perspective trapezoid directly beneath suspended load)
        horizon_y = max(int(h * self.ground_horizon), y2 + 25)
        footprint_depth = min(int(load_w * 0.40), 100)
        floor_y = min(h - 15, horizon_y + max(40, footprint_depth))

        if horizon_y >= floor_y:
            horizon_y = floor_y - 40

        # Safety width with swing buffer
        span_top = int(max(load_w * self.drop_margin * 0.7, 80))
        span_bot = int(max(load_w * self.drop_margin * 1.1, 120))

        # Ground trapezoid polygon: (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        pt_top_left = [max(10, center_x - span_top // 2), horizon_y]
        pt_top_right = [min(w - 10, center_x + span_top // 2), horizon_y]
        pt_bot_right = [min(w - 10, center_x + span_bot // 2), floor_y]
        pt_bot_left = [max(10, center_x - span_bot // 2), floor_y]

        raw_poly = np.array([pt_top_left, pt_top_right, pt_bot_right, pt_bot_left], dtype=np.float32)

        # Temporal smoothing
        zone_id = f"crane_drop_{idx}"
        smoothed_poly = self._smooth_polygon(zone_id, raw_poly)
        int_poly = smoothed_poly.astype(np.int32)

        # Vertical tracer lines (from bottom corners of load to ground polygon)
        tracers = [
            ((x1, y2), (int_poly[0][0], int_poly[0][1])),  # Left tracer
            ((x2, y2), (int_poly[1][0], int_poly[1][1])),  # Right tracer
            ((center_x, y2), (center_x, int_poly[0][1])), # Center plumb line
        ]

        # Check breach: Worker or camera wearer inside ground drop zone
        breached_workers: List[Detection] = []
        for worker in workers:
            # Check worker feet / bottom center
            wx, wy = worker.bottom_center
            if cv2.pointPolygonTest(int_poly, (float(wx), float(wy)), False) >= 0:
                breached_workers.append(worker)

        # Check camera wearer (screen bottom center represents forward walking path)
        wearer_pt = (w // 2, int(h * 0.92))
        wearer_in_zone = cv2.pointPolygonTest(int_poly, (float(wearer_pt[0]), float(wearer_pt[1])), False) >= 0

        is_breached = len(breached_workers) > 0 or wearer_in_zone

        if is_breached:
            risk = RiskLevel.HIGH
            title = "CRITICAL: DROP ZONE BREACHED"
            if wearer_in_zone:
                msg = "CRITICAL: You are walking under a suspended load! Move clear immediately!"
            else:
                msg = f"CRITICAL: {len(breached_workers)} worker(s) in suspended load drop zone!"
        else:
            risk = RiskLevel.MEDIUM
            title = "CRANE SUSPENDED LOAD THREAT"
            msg = "Crane load overhead — keep drop zone clear of all personnel"

        return HazardZone(
            zone_id=zone_id,
            zone_type="suspended_drop_zone",
            risk_level=risk,
            ground_polygon=int_poly,
            source_box=crane.box,
            vertical_tracers=tracers,
            title=title,
            message=msg,
            is_breached=is_breached,
            breached_workers=breached_workers,
        )

    def _create_machinery_zone(
        self,
        mach: Detection,
        idx: int,
        w: int,
        h: int,
        workers: List[Detection],
    ) -> Optional[HazardZone]:
        """
        Create expanded ground perimeter zone around heavy machinery.
        """
        x1, y1, x2, y2 = mach.box
        mw = x2 - x1
        mh = y2 - y1

        # Expand perimeter
        pad_x = int(mw * 0.30)
        pad_y = int(mh * 0.20)

        gx1 = max(10, x1 - pad_x)
        gy1 = max(10, y1 + int(mh * 0.6))  # Bottom portion of machine
        gx2 = min(w - 10, x2 + pad_x)
        gy2 = min(h - 10, y2 + pad_y)

        raw_poly = np.array([
            [gx1, gy1],
            [gx2, gy1],
            [gx2, gy2],
            [gx1, gy2],
        ], dtype=np.float32)

        zone_id = f"mach_{idx}"
        smoothed_poly = self._smooth_polygon(zone_id, raw_poly)
        int_poly = smoothed_poly.astype(np.int32)

        # Check worker proximity
        breached_workers: List[Detection] = []
        for worker in workers:
            wx, wy = worker.bottom_center
            if cv2.pointPolygonTest(int_poly, (float(wx), float(wy)), False) >= 0:
                breached_workers.append(worker)

        # Check camera wearer
        wearer_pt = (w // 2, int(h * 0.90))
        wearer_in_zone = cv2.pointPolygonTest(int_poly, (float(wearer_pt[0]), float(wearer_pt[1])), True) >= -20

        is_breached = len(breached_workers) > 0 or wearer_in_zone

        if is_breached:
            risk = RiskLevel.HIGH
            title = "MACHINERY COLLISION RISK"
            msg = f"{mach.class_name.upper()} in close proximity! Keep clear of equipment!"
        else:
            risk = RiskLevel.MEDIUM
            title = f"{mach.class_name.upper()} OPERATING"
            msg = f"{mach.class_name.capitalize()} operating ahead. Maintain safe distance."

        return HazardZone(
            zone_id=zone_id,
            zone_type="machinery_zone",
            risk_level=risk,
            ground_polygon=int_poly,
            source_box=mach.box,
            vertical_tracers=[],
            title=title,
            message=msg,
            is_breached=is_breached,
            breached_workers=breached_workers,
        )

    def _create_barrier_zone(
        self,
        barrier: Detection,
        idx: int,
        w: int,
        h: int,
        workers: List[Detection],
    ) -> Optional[HazardZone]:
        """
        Create restricted perimeter zone for hazard tape, cones, and barricades.
        """
        x1, y1, x2, y2 = barrier.box
        bw = max(40, x2 - x1)
        bh = max(20, y2 - y1)

        # Extend slightly into ground area behind barrier/tape
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.40)

        gx1 = max(5, x1 - pad_x)
        gy1 = max(5, y1)
        gx2 = min(w - 5, x2 + pad_x)
        gy2 = min(h - 5, y2 + pad_y)

        raw_poly = np.array([
            [gx1, gy1],
            [gx2, gy1],
            [gx2, gy2],
            [gx1, gy2],
        ], dtype=np.float32)

        zone_id = f"barrier_{idx}"
        smoothed_poly = self._smooth_polygon(zone_id, raw_poly)
        int_poly = smoothed_poly.astype(np.int32)

        # Proximity check: Is wearer directly approaching boundary?
        mid_x = (x1 + x2) // 2
        in_path = abs(mid_x - w // 2) < (w * 0.35)
        is_close_path = (y2 > h * 0.80) and in_path
        wearer_pt = (w // 2, int(h * 0.90))
        wearer_breach = cv2.pointPolygonTest(int_poly, (float(wearer_pt[0]), float(wearer_pt[1])), True) >= -10

        is_high = is_close_path or wearer_breach
        risk = RiskLevel.HIGH if is_high else RiskLevel.MEDIUM
        title = "RESTRICTED BOUNDARY" if "tape" in barrier.class_name.lower() else "HAZARD PERIMETER"
        msg = "Hazard tape / boundary ahead — DO NOT CROSS!" if is_high else "Safety barrier / marking detected ahead."

        return HazardZone(
            zone_id=zone_id,
            zone_type="hazard_tape_zone",
            risk_level=risk,
            ground_polygon=int_poly,
            source_box=barrier.box,
            vertical_tracers=[],
            title=title,
            message=msg,
            is_breached=wearer_breach,
            breached_workers=[],
        )

    def _create_sign_zone(
        self,
        sign: Detection,
        idx: int,
        w: int,
        h: int,
    ) -> Optional[HazardZone]:
        """
        Project directional hazard zone ahead of safety sign.
        """
        x1, y1, x2, y2 = sign.box
        name = sign.class_name.lower()

        # Is this an absolute stop / no entry / danger sign?
        is_danger = any(k in name for k in ("no_entry", "danger", "stop", "prohibited"))

        # Projection ground trapezoid in front of sign
        sw = max(30, x2 - x1)
        cx = (x1 + x2) // 2
        top_y = min(h - 80, y2 + 10)
        bot_y = min(h - 20, top_y + 120)

        raw_poly = np.array([
            [max(10, cx - sw), top_y],
            [min(w - 10, cx + sw), top_y],
            [min(w - 10, cx + sw * 2), bot_y],
            [max(10, cx - sw * 2), bot_y],
        ], dtype=np.float32)

        zone_id = f"sign_{idx}"
        smoothed_poly = self._smooth_polygon(zone_id, raw_poly)
        int_poly = smoothed_poly.astype(np.int32)

        risk = RiskLevel.HIGH if is_danger else RiskLevel.MEDIUM
        title = "RESTRICTED ACCESS" if is_danger else "CAUTION SIGN"
        msg = f"{sign.class_name.upper()} ahead — stop and check clearance!" if is_danger else "Caution sign ahead — proceed with vigilance."

        return HazardZone(
            zone_id=zone_id,
            zone_type="sign_zone",
            risk_level=risk,
            ground_polygon=int_poly,
            source_box=sign.box,
            vertical_tracers=[],
            title=title,
            message=msg,
            is_breached=False,
            breached_workers=[],
        )

    def _create_ppe_zone(
        self,
        violations: List[Detection],
        w: int,
        h: int,
    ) -> Optional[HazardZone]:
        """
        Create warning zone for workers detected without required PPE.
        """
        first = violations[0]
        x1, y1, x2, y2 = first.box
        poly = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.int32)

        return HazardZone(
            zone_id="ppe_violation",
            zone_type="ppe_violation",
            risk_level=RiskLevel.MEDIUM,
            ground_polygon=poly,
            source_box=first.box,
            vertical_tracers=[],
            title="PPE COMPLIANCE ALERT",
            message="Person without required PPE (hard hat/vest) in view",
            is_breached=False,
            breached_workers=violations,
        )

    def _smooth_polygon(self, zone_id: str, poly: np.ndarray) -> np.ndarray:
        """
        Exponential moving average smoothing over recent frames to eliminate jitter.
        """
        if zone_id not in self._poly_history:
            self._poly_history[zone_id] = deque(maxlen=self.smoothing_frames)

        history = self._poly_history[zone_id]
        history.append(poly)

        # Calculate weighted moving average
        weights = np.linspace(0.5, 1.0, len(history))
        weights /= weights.sum()

        smoothed = np.zeros_like(poly)
        for w, p in zip(weights, history):
            smoothed += w * p

        return smoothed

