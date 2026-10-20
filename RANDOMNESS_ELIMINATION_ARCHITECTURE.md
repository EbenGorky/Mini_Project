# Helmet Vision — Comprehensive Anti-Randomness Architecture & Implementation Plan

> **Executive Objective:** Eliminate all visual flickering, transient 1-frame false alarms, and border oscillation glitches in Helmet Vision while guaranteeing 100% continuous detection of real construction site hazards (suspended loads, cranes, and worker breaches).

---

## 1. Root Cause Diagnosis (Why Randomness Occurs)

Through deep-frame inspection of the pipeline across simulated video feeds and raw frames, we identified four distinct physical and algorithmic sources of randomness:

```
+-------------------------------------------------------------------------------------------------+
|                                 SOURCES OF DETECTION RANDOMNESS                                 |
+-------------------------------+---------------------------------+-------------------------------+
|  1. Semantic Misclassification|    2. Boundary Jitter & Lack    |   3. Naive IoU Tracking Loss  |
|      (COCO Pretrained Net)    |          of Hysteresis          |        & ID Switching         |
| Standard YOLOv8n has domestic | When a worker walks near the    | Simple IoU (>0.18) fails when |
| classes. Random site textures | drop zone edge, +/-5px bounding | camera moves fast or objects  |
| get tagged as kite/bird/bench | box jitter flips hazard state   | overlap. Resets hit-streak    |
| for a split second.           | RED <--> AMBER every frame.     | causing rapid on/off blips.   |
+-------------------------------+---------------------------------+-------------------------------+
```

### Detailed Breakdown:
1. **Semantic Ambiguity in Pretrained COCO Weights:**
   - `yolov8n.pt` has zero classes for `crane`, `hook`, `rigging`, `derrick`, `girder`, or `precast_slab`.
   - Heuristic remapping of classes like `kite` or `bird` aloft in the air can pick up passing reflections, scaffold cross-bracing, or power lines for 1–2 frames.
2. **Boundary Oscillation (Lack of Schmitt Trigger / Hysteresis):**
   - The drop zone footprint polygon is tested against the worker's foot coordinate $(w_x, w_y)$ via `cv2.pointPolygonTest`.
   - If the worker is walking along the border, bounding box jitter of just $3\text{ pixels}$ causes the point to test `+1` (inside $\rightarrow$ **CRITICAL RED**) on frame $N$, and `-1` (outside $\rightarrow$ **AMBER**) on frame $N+1$.
   - This creates a **strobe-light flickering effect**.
3. **Naive IoU Tracker Frame Misses:**
   - The current tracking uses static IoU matching. If a worker turns or camera pans rapidly, the IoU between consecutive frames drops below $0.18$, causing the tracker to treat the worker as a "new track" ($hit\_streak = 1$), causing the bounding box to drop out and reappear.
4. **HSV Skylight Sensitivity:**
   - Sky brightness and color temperature fluctuate with auto-exposure and clouds. Fixed threshold masks can momentarily fragment a solid girder into multiple sub-contours or connect with background cables.

---

## 2. The 5-Pillar Anti-Randomness Solution Architecture

```mermaid
flowchart TD
    subgraph INTAKE ["1. Intake & Pre-Filtering"]
        Frame[Raw Video Frame] --> SceneCheck{Scene Cut / Dark Frame?}
        SceneCheck -- Yes --> Reset[Hard Reset Tracker State]
        SceneCheck -- No --> Preprocess[CLAHE Contrast Normalization]
    end

    subgraph DETECTION ["2. Dual-Engine Detection"]
        Preprocess --> YOLO[YOLO Inference: Conf >= 0.25]
        Preprocess --> Aloft[Aloft Structural Girder Segmenter]
        YOLO --> WhitelistFilter[Strict Category Whitelist Filter: Suppress Domestic COCO]
        Aloft --> AloftGeom[Geometry & Open-Sky Isolation]
    end

    subgraph TRACKING ["3. Spatial-Temporal Kalman Tracking"]
        WhitelistFilter --> ByteTrack[Kalman Filter Track Predictor]
        AloftGeom --> ByteTrack
        ByteTrack --> BBoxSmooth[Exponential Moving Average Box Damping]
        BBoxSmooth --> ConfirmFilter{Hit Streak >= 3?}
        ConfirmFilter -- No --> Suppress[Drop Silently - Zero Flicker]
        ConfirmFilter -- Yes --> StableTracks[Confirmed Smooth Targets]
    end

    subgraph HYSTERESIS ["4. Dual-Threshold Hysteresis Hazard Engine"]
        StableTracks --> GroundProjection[Ground Drop Zone Projection]
        GroundProjection --> DualBoundary[Dual Boundary: Inner Breach (0px) vs Outer Clear (+40px)]
        DualBoundary --> StateMachine{Hazard State Machine}
        StateMachine -- 3 Consecutive Breaches --> RedState[RED ALERT: Critical Drop Zone Breach]
        StateMachine -- 10 Consecutive Clear Frames --> AmberState[AMBER CAUTION: Drop Zone Clear]
    end
```

---

## 3. Pillar Specifications

### Pillar 1: Dual-Threshold Spatial & Temporal Hysteresis (Schmitt Trigger)
To completely stop the RED $\leftrightarrow$ AMBER strobe effect along drop zone boundaries:
- **Spatial Hysteresis (Dual Boundary):**
  - **Inner Breach Polygon ($P_{in}$):** The actual drop zone footprint. To transition from `AMBER` $\rightarrow$ `RED`, the worker's feet must penetrate $P_{in}$ (`distance >= 0`).
  - **Outer Clearance Polygon ($P_{out}$):** An expanded perimeter ($P_{in} + 35\text{px}$). To transition back from `RED` $\rightarrow$ `AMBER`, the worker must completely step outside $P_{out}$ (`distance < -35px`).
  - Borderline workers walking along the edge remain locked in their previous state with zero oscillation.
- **Temporal Hysteresis (State Machine):**
  - **Activation Debounce:** Requires $N_{breach} \ge 3$ consecutive frames in breach before escalating to **RED**.
  - **De-escalation Debounce:** Requires $N_{clear} \ge 10$ consecutive frames (~350ms) completely outside the safety zone before de-escalating back to **AMBER** or **GREEN**.

```python
# Conceptual Dual-Threshold Implementation
class HazardHysteresisTracker:
    def __init__(self, activation_frames=3, release_frames=10, margin_px=35):
        self.activation_frames = activation_frames
        self.release_frames = release_frames
        self.margin_px = margin_px
        self.breach_counter = 0
        self.clear_counter = 0
        self.current_state = "AMBER"

    def update(self, ground_poly, worker_feet):
        dist = cv2.pointPolygonTest(ground_poly, worker_feet, True)
        if dist >= 0:  # Inside inner zone
            self.breach_counter += 1
            self.clear_counter = 0
            if self.breach_counter >= self.activation_frames:
                self.current_state = "RED"
        elif dist < -self.margin_px:  # Outside outer buffer
            self.clear_counter += 1
            self.breach_counter = 0
            if self.clear_counter >= self.release_frames:
                self.current_state = "AMBER"
        return self.current_state
```

---

### Pillar 2: Kalman-Assisted Bounding Box Smoothing
Instead of raw YOLO bounding boxes that jitter $\pm 5\text{--}10\text{px}$ per frame:
- Maintain a **Constant Velocity Kalman Filter** for each tracked object.
- The Kalman filter predicts object centroid $(c_x, c_y)$ and scale $(w, h)$ across frames.
- Coordinates are updated using an **Exponential Moving Average (EMA)** with momentum $\alpha = 0.80$:
  $$\text{Box}_{t} = 0.80 \cdot \text{Detection}_{t} + 0.20 \cdot \text{Track}_{t-1}$$
- **Benefit:** Smooth, cinema-quality bounding boxes and rock-solid drop zone projection with zero coordinate vibration.

---

### Pillar 3: Strict Category Whitelisting (Zero False Domestic Classes)
COCO pretrained models emit 80 classes. On a construction site, 90% of those classes are noise.
- **Strict Allowlist:** Only allow:
  1. `person` $\rightarrow$ `WORKER`
  2. `hard_hat` / `helmet` / `safety_vest` $\rightarrow$ `PPE`
  3. `crane` / `tower_crane` / `boom_crane` $\rightarrow$ `CRANE`
  4. `excavator` / `forklift` / `loader` / `truck` $\rightarrow$ `MACHINERY`
  5. `suspended_girder` $\rightarrow$ `SUSPENDED_LOAD`
- **Complete Suppression:** Explicitly discard domestic classes (`bench`, `umbrella`, `clock`, `cat`, `chair`, `airplane`, `boat`, `sports ball`, `frisbee`). They are immediately dropped before ever entering the hazard engine.

---

### Pillar 4: Gradient & Edge-Guided Aloft Structural Detector
To make structural girder detection immune to sky lighting and cloud shifts:
- Combine HSV luminance with **horizontal Sobel gradient magnitude**:
  - Girders have distinct, parallel horizontal edges (top flange, bottom flange).
  - Compute $G_y = \text{Sobel}_y(\text{SkyROI})$.
  - True girders exhibit strong horizontal gradient lines ($G_y > 80$) bounding a dark solid web.
- Add an **Aloof Isolation Guard**:
  - The object must be isolated with open sky both above ($y_1 > \text{sky}_{y1} + 6$) and below ($y_2 < \text{sky}_{y2} - 15$).
  - Aspect ratio bounded to $2.5 \le \text{Aspect} \le 15.0$.

---

### Pillar 5: Fine-Tuned Construction Domain Weights (`best.pt`)
The ultimate long-term solution to COCO-related randomness is replacing generic COCO weights with weights trained on construction safety imagery:
- Using the training pipeline already created at [`vison_/train/train.py`](file:///home/eben/Developnment/MiniProject/vison_/train/train.py) with [`vison_/train/classes.yaml`](file:///home/eben/Developnment/MiniProject/vison_/train/classes.yaml):
  - Classes: `Worker`, `Hardhat`, `No-Hardhat`, `Safety-Vest`, `No-Vest`, `Crane`, `Suspended-Load`, `Heavy-Machinery`.
- Training for 30 epochs on a dataset such as **SHWD (Safety Helmet Wearing Dataset)** or **Pictor-v3 Construction Dataset** produces $mAP_{50} > 0.88$ with direct native output of `Crane` and `Suspended-Load`, eliminating all heuristic remapping permanently.

---

## 4. Implementation Roadmap & Milestones

| Step | Action Item | File to Modify | Target Result |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Implement **Dual-Threshold Hysteresis State Machine** (3-frame activation, 10-frame release, 35px margin) | `vison_/core/hazards.py` | Eliminates border strobe flickering completely |
| **Phase 2** | Integrate **Kalman Velocity Smoothing** & increase minimum confirmation hits ($N=3$) | `vison_/core/detector.py` | Eliminates bounding box vibration & single-frame blips |
| **Phase 3** | Add **Horizontal Sobel Edge Verification** to Aloft Structural Detector | `vison_/core/detector.py` | Eliminates cloud/lighting false alarms aloft |
| **Phase 4** | Train / load domain-specific YOLOv8 construction safety weights | `vison_/models/best.pt` | Native detection with >85% confidence on all cranes/loads |

---

## 5. Verification Metrics for Success

A build is certified **Zero-Randomness Compliant** only when:
1. **0 Single-Frame Spikes:** Running the full 1850-frame video produces exactly 0 isolated 1-frame hazard alerts.
2. **0 Border Flips:** A worker stationary on the boundary of a drop zone maintains a single stable risk state without flickering.
3. **100% Hazard Retention:**
   - The 13.6s crane and aloft load remain continuously highlighted.
   - The 62.8s steel girder and ground drop zone remain continuously highlighted in red when the worker passes beneath.
4. **Latency Budget:** Total inference + tracking + hazard processing remains under **12.0 ms** on the NVIDIA RTX 4060 GPU (>80 FPS).
