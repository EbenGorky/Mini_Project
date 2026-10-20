"""
Generates a realistic synthetic test video simulating a construction site
with a moving vehicle, worker with hard hat, and hazard tape markings.
Used for offline testing without physical ESP32-CAM hardware.
"""

import math
import os
import cv2
import numpy as np


def create_sample_video(
    output_path: str = "test_media/sample_site.mp4",
    duration_sec: int = 10,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_frames = duration_sec * fps

    # Try MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[ERROR] Could not open VideoWriter for {output_path}")
        return ""

    print(f"[INFO] Generating {total_frames} frames ({duration_sec}s @ {fps}fps) to '{output_path}'...")

    for i in range(total_frames):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Sky & Ground background
        frame[: height // 2] = [180, 160, 140]  # Dusty overcast sky
        frame[height // 2 :] = [90, 95, 100]    # Concrete/gravel floor

        # Construction scaffolding grid in background
        for x in range(50, width, 120):
            cv2.line(frame, (x, 80), (x, height // 2 + 50), (130, 130, 130), 3)
        for y in range(80, height // 2 + 50, 45):
            cv2.line(frame, (30, y), (width - 30, y), (130, 130, 130), 2)

        # 2. Yellow & Black hazard tape barrier line
        tape_y = int(height * 0.72)
        tape_h = 24
        cv2.rectangle(frame, (0, tape_y), (width, tape_y + tape_h), (0, 215, 255), -1)
        stripe_width = 30
        for sx in range(-stripe_width, width + stripe_width, stripe_width * 2):
            pts = np.array(
                [
                    [sx, tape_y],
                    [sx + stripe_width, tape_y],
                    [sx + stripe_width - 15, tape_y + tape_h],
                    [sx - 15, tape_y + tape_h],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [pts], (20, 20, 20))

        # 3. Construction Sign (Warning Sign on post)
        sign_cx = 120
        sign_cy = int(height * 0.50)
        # Sign pole
        cv2.line(frame, (sign_cx, sign_cy), (sign_cx, tape_y), (80, 80, 80), 6)
        # Diamond warning sign
        diamond_pts = np.array(
            [
                [sign_cx, sign_cy - 40],
                [sign_cx + 45, sign_cy],
                [sign_cx, sign_cy + 40],
                [sign_cx - 45, sign_cy],
            ],
            np.int32,
        )
        cv2.fillPoly(frame, [diamond_pts], (0, 215, 255))
        cv2.polylines(frame, [diamond_pts], True, (0, 0, 0), 3)
        cv2.putText(
            frame,
            "!",
            (sign_cx - 8, sign_cy + 15),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        # 4. Worker silhouette (standing with helmet and vest)
        worker_x = 480
        worker_y = int(height * 0.42)
        # Body / Torso (Safety vest: Orange)
        cv2.rectangle(frame, (worker_x - 20, worker_y + 40), (worker_x + 20, worker_y + 110), (0, 140, 255), -1)
        # Reflective stripes on vest
        cv2.line(frame, (worker_x - 18, worker_y + 65), (worker_x + 18, worker_y + 65), (230, 230, 230), 4)
        cv2.line(frame, (worker_x - 18, worker_y + 85), (worker_x + 18, worker_y + 85), (230, 230, 230), 4)
        # Legs
        cv2.rectangle(frame, (worker_x - 18, worker_y + 110), (worker_x - 3, worker_y + 175), (50, 40, 30), -1)
        cv2.rectangle(frame, (worker_x + 3, worker_y + 110), (worker_x + 18, worker_y + 175), (50, 40, 30), -1)
        # Head
        cv2.circle(frame, (worker_x, worker_y + 20), 16, (170, 190, 215), -1)
        # Hard hat (Yellow safety helmet)
        cv2.ellipse(frame, (worker_x, worker_y + 12), (20, 14), 0, 180, 360, (0, 220, 255), -1)
        cv2.rectangle(frame, (worker_x - 22, worker_y + 11), (worker_x + 22, worker_y + 15), (0, 200, 240), -1)

        # 5. Moving Vehicle / Machinery (e.g. Loader or Truck)
        veh_x = int(180 + 160 * math.sin(t * 0.8))
        veh_y = int(height * 0.52)
        veh_w = 170
        veh_h = 95
        # Vehicle chassis (Yellow construction machinery)
        cv2.rectangle(frame, (veh_x, veh_y), (veh_x + veh_w, veh_y + veh_h), (0, 180, 240), -1)
        # Cabin
        cv2.rectangle(frame, (veh_x + 30, veh_y - 45), (veh_x + veh_w - 40, veh_y), (50, 50, 50), -1)
        cv2.rectangle(frame, (veh_x + 35, veh_y - 40), (veh_x + veh_w - 45, veh_y - 5), (220, 220, 220), -1)
        # Wheels
        cv2.circle(frame, (veh_x + 35, veh_y + veh_h), 22, (20, 20, 20), -1)
        cv2.circle(frame, (veh_x + 35, veh_y + veh_h), 10, (120, 120, 120), -1)
        cv2.circle(frame, (veh_x + veh_w - 35, veh_y + veh_h), 22, (20, 20, 20), -1)
        cv2.circle(frame, (veh_x + veh_w - 35, veh_y + veh_h), 10, (120, 120, 120), -1)

        # 6. Subtle timestamp & frame counter on top
        cv2.putText(
            frame,
            f"SIMULATED SITE FEED - T+{t:05.2f}s [FRAME {i:04d}]",
            (20, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()
    print(f"[SUCCESS] Sample test video saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_video()

