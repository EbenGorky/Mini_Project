# Helmet Vision — Real-Time Construction Safety Monitoring System

**Helmet Vision** is an edge-AI computer vision system designed for construction site safety monitoring. It connects to a helmet-mounted ESP32-CAM (or simulated HTTP MJPEG video stream), runs GPU-accelerated deep learning detection (Ultralytics YOLO) on an NVIDIA RTX 4060 GPU, projects dynamic 3D perspective ground danger zones beneath overhead crane loads and machinery, detects yellow/black hazard tape boundaries and safety signs, and provides instant audio alarms and incident logging.

---

## Architecture Overview

```
[ ESP32-CAM on Helmet ]  ── Wi-Fi MJPEG ──>  [ Laptop / Host (RTX 4060 GPU) ]
(http://<ip>:81/stream)                          │
                                                 ├── 1. capture.py (Threaded latest-frame grabber)
                                                 ├── 2. detector.py (YOLOv8 + IOU temporal tracking)
                                                 ├── 3. markings.py (HSV hazard tape & signs)
                                                 ├── 4. hazards.py (3D drop zones & risk engine)
                                                 ├── 5. alerts.py (Audio beeps & CSV/JSON logging)
                                                 └── 6. overlay.py (Interactive HUD & Controls)
```

---

## Quick Start & Running Scripts

Helmet Vision uses independent modular bash scripts to start and stop services:

### 1. Test Mode (Mock Streamer Server)
Runs an interactive MJPEG test streamer on `http://localhost:4000/stream` with a web controls interface at `http://localhost:4000/`:

```bash
# Start streamer in background
./run_stream.sh

# Stop streamer
./stop_stream.sh
```

*(Web browser interface at `http://localhost:4000/` provides a **`[⏪ Reset Crane (13.6s)]`** button and keyboard shortcut `r` to instantly replay the crane lift sequence).*

### 2. Vision App (Live AI Safety Monitor)
Launches the Helmet Vision application:

```bash
# Start Helmet Vision monitor
./run_vision.sh

# Connect to specific ESP32-CAM hardware by IP
./run_vision.sh --ip 192.168.1.50

# Stop Helmet Vision monitor
./stop_vision.sh
```

### 3. Emergency Stop (All Services)
```bash
./stop_all.sh
```

---

## Hardware Setup: ESP32-CAM

1. **Firmware:** Flash the standard `CameraWebServer` example from the Arduino ESP32 library (`Examples > ESP32 > Camera > CameraWebServer`).
2. **Camera Model:** Select `#define CAMERA_MODEL_AI_THINKER`.
3. **Resolution:** Configure resolution to QVGA (`320x240`) or VGA (`640x480`) for optimal low latency and stable frame rate.
4. **Find IP Address:**
   - Open Arduino IDE Serial Monitor at `115200` baud while the ESP32-CAM boots.
   - Look for: `Camera Ready! Use 'http://192.168.X.X' to connect`.
   - The MJPEG stream endpoint is automatically located at `http://<ip>:81/stream`.
5. **In Helmet Vision:** Enter the IP into the on-screen input box and click **CONNECT** (or pass `--ip <ip>`).

---

## Key Safety Capabilities

### 1. Overhead Cranes & Suspended Loads
- **Continuous Tracking:** Detects crane masts, booms, and suspended loads aloft in mid-air.
- **Dynamic Ground Drop Zones:** Projects trapezoid exclusion footprints onto the ground beneath suspended loads with vertical laser plumb lines and safety buffers.
- **Worker Breach Detection:** Evaluates worker feet coordinates against the ground footprint, escalating immediately to **`HIGH RISK`** with evacuation alerts.

### 2. Construction Markings & Signs (Milestone 4)
- **Yellow/Black Hazard Tape:** High-saturation HSV color segmentation detects restricted perimeter tape and floor lines. Proximity to the wearer's path raises immediate alerts.
- **Traffic Cones & Barricades:** Detects orange traffic cones and barriers defining restricted zones.
- **Safety Signs:** Classifies `NO ENTRY`, `DANGER`, `STOP`, and `CAUTION` signs, projecting directional warning zones ahead of flagged areas.

### 3. Suggestions, Audio Alarms & Incident Logging (Milestone 5)
- **Audio Beep Alarms:** Plays non-blocking alert tones (`880Hz`) through system audio (`pw-play` / `paplay`) during High risk events.
- **Structured Incident Logs:** Records all risk escalations in `vison_/logs/events.csv` and `vison_/logs/events.jsonl`.
- **Snapshot Captures:** Automatically saves timestamped, annotated screenshot images to `vison_/logs/snapshots/` whenever safety violations occur.

---

## Custom Model Training (Milestone 6)

A dedicated fine-tuning pipeline is located in `vison_/train/`:

```bash
cd vison_

# 1. Generate template dataset configuration
./venv/bin/python train/train.py --generate-sample-config

# 2. Fine-tune YOLO on custom construction safety dataset
./venv/bin/python train/train.py \
    --data dataset.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --export-to-app
```

- **Taxonomy Mapping (`vison_/train/classes.yaml`):** Maps 16 custom classes (`worker`, `hard_hat`, `no_hard_hat`, `safety_vest`, `crane`, `suspended_load`, `hazard_tape`, etc.) to spatial safety risk rules.

---

## Configuration Reference (`config.yaml`)

| Setting | Default | Description |
| :--- | :--- | :--- |
| `camera.port` | `81` | Default ESP32-CAM stream port |
| `camera.reconnect_interval_sec` | `3.0` | Auto-reconnect polling interval |
| `detection.confidence_threshold`| `0.15` | Minimum YOLO detection confidence |
| `detection.max_missed_frames` | `15` | Temporal tracker frame-loss tolerance (~600ms) |
| `hazards.suspended_load_safety_margin` | `1.8` | Overhead drop zone ground multiplier |
| `hazards.zone_hold_frames` | `15` | Ground footprint persistence buffer |
| `markings.hazard_tape.yellow_hsv_lower` | `[18, 90, 80]` | HSV segmentation lower threshold |
| `alerts.beep_on_high_risk` | `true` | Audio alarms for high-risk threats |
| `alerts.save_snapshots` | `true` | Save annotated violation snapshots to `logs/snapshots/` |
