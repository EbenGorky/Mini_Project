# Project Brief: Helmet-Mounted Vision System for Construction Worker Safety

Paste this into Antigravity as the starting context. Build in the milestone order at the bottom and test each stage before moving on.

## 1. Goal

A helmet-mounted ESP32-CAM streams live video over Wi-Fi to a laptop. Software on the laptop analyses the video in real time and:

1. Detects workers, vehicles/machinery, PPE, and construction hazards.
2. Highlights danger areas directly on the video.
3. Reacts to construction symbols and markings (safety signs, cones, barriers, hazard tape, floor markings).
4. Gives short, plain-language safety suggestions and alerts.

## 2. Architecture

```
[ESP32-CAM on helmet] --Wi-Fi MJPEG--> [Laptop: capture -> detect -> hazard logic -> overlay] --> [Live view + alerts + log]
```

- **Camera node:** ESP32-CAM running the standard CameraWebServer-style firmware, MJPEG stream at `http://<esp32-ip>:81/stream`. Keep resolution at QVGA/VGA (320x240 to 640x480) for a stable frame rate.
- **Laptop (GPU if available), Python:**
  - `capture.py`: read the stream with OpenCV in a separate thread, always keep only the latest frame (drop old frames to avoid lag), auto-reconnect if the stream drops.
  - `detector.py`: YOLO (Ultralytics) inference. Run on the GPU, with a configurable confidence threshold.
  - `hazards.py`: rule engine that turns detections into danger zones and suggestions.
  - `overlay.py`: draws boxes, semi-transparent danger zones, labels, and a status banner.
  - `alerts.py`: on-screen banner, optional sound, event log (CSV/JSON with timestamp and snapshot).
  - `app.py`: ties it together. Start with an OpenCV window. Later option: FastAPI + WebSocket web dashboard.
- Config in one `config.yaml` (stream URL, model path, thresholds, zone rules).

## 3. Detection classes

A stock COCO YOLO model does not know construction signs or markings. Plan for a custom-trained model (fine-tune YOLOv8/YOLO11 on public construction-safety datasets, e.g. from Roboflow Universe, plus own frames from the ESP32-CAM).

Target classes:

- **People and PPE:** person, helmet, no-helmet, safety vest, no-vest
- **Machinery:** excavator, truck, loader, crane, forklift, other vehicle
- **Signs and markings:** warning/danger sign, no-entry sign, caution sign, traffic cone, barrier, hazard tape (yellow/black), painted floor markings, open hole/trench edge
- Start with a small first version (person, vehicle/machinery, cone, barrier, hazard tape, warning sign) and expand.

Complement the model with classical vision where it helps:

- Yellow/black hazard tape and painted markings via HSV colour segmentation + contour/line detection.
- Optional OCR (EasyOCR/PaddleOCR) for sign text like "DANGER", "NO ENTRY", "HIGH VOLTAGE", applied only to detected sign regions.

## 4. Danger-area highlighting

Draw semi-transparent red/orange polygons on the video for:

- **Marked zones:** the region bounded by hazard tape, cones, barriers or floor markings.
- **Machinery zones:** an enlarged region around each detected vehicle/machine. The zone grows with the box size (a closer machine means a bigger zone in the image).
- **Worker-to-machine proximity:** if the wearer's view shows a machine whose zone overlaps the centre/bottom of the frame (where the wearer is heading), escalate the warning.
- **Sign-triggered zones:** when a danger/no-entry sign is detected, highlight the area ahead of it.

Risk levels: **Low (green), Medium (amber), High (red)**. Use temporal smoothing over several frames so the highlight does not flicker.

Note: with a single camera, distances are approximate and in image space. Use box size and position as a proxy for distance, and calibrate thresholds from real test footage.

## 5. Suggestions and alerts

The rule engine outputs short messages, shown on screen and logged, for example:

- Vehicle ahead, move clear of its path
- Hazard tape detected, do not cross
- No-entry sign ahead, stop and check
- Person without helmet/vest in view
- Cone line ahead, area restricted

Rules should be a simple, editable table in config (condition -> risk level -> message). An LLM-generated explanation can be a later add-on, but the rules must work without internet.

## 6. Non-functional requirements

- Target at least 10-15 FPS end to end on the laptop; show live FPS and latency on screen.
- Frame skipping and a lower inference size if the GPU is weak.
- Graceful handling of stream loss (reconnect and show a "camera offline" state).
- Clear modular code, type hints, README with setup steps, `requirements.txt`.
- Test mode: run the same pipeline on a recorded video file so development doesn't need the hardware.

## 7. Milestones (test each before the next)

1. **Stream:** view the ESP32-CAM stream in an OpenCV window with reconnect and FPS counter.
2. **Detection:** run a pretrained YOLO on the stream; show boxes for person and vehicles.
3. **Custom model:** fine-tune on construction classes (PPE, cones, barriers, signs, machinery); evaluate on held-out frames.
4. **Danger zones:** implement overlays for machinery zones, then hazard tape/cone-bounded zones.
5. **Signs and markings:** add sign classes, HSV tape/marking detection, optional OCR.
6. **Rules and alerts:** suggestion engine, on-screen banner, sound, event log.
7. **Polish:** web dashboard (optional), config file, README, demo video recording.

## 8. Deliverables

Working code in a repo, `README.md`, `requirements.txt`, `config.yaml`, a short test video with overlays, and a note on model accuracy per class.
