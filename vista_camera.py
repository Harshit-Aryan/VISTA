"""
VISTA — Vision-based Inertial-free Spatial Tracking Architecture
Module: vista_camera.py
Phase 1 & 2: Video Acquisition and Shi-Tomasi Feature Detection Baseline
"""

import cv2
import numpy as np
import time

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Camera not detected. Please verify webcam connection and permissions.")
        return

    # Set camera resolution (standard 720p or 640x480 fallback)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Shi-Tomasi corner detection parameters
    feature_params = {
        "maxCorners": 150,
        "qualityLevel": 0.01,
        "minDistance": 10,
        "blockSize": 7
    }

    prev_time = time.perf_counter()
    fps = 0.0

    print("[VISTA] Starting Feature Detection Baseline...")
    print("[VISTA] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Exiting loop...")
            break

        # Calculate FPS
        curr_time = time.perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Convert frame to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Shi-Tomasi corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            **feature_params
        )

        feature_count = 0
        if corners is not None:
            corners = np.int32(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),  # Green marker
                    -1
                )
                feature_count += 1

        # Render telemetry overlay
        cv2.rectangle(frame, (10, 10), (320, 85), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (320, 85), (0, 255, 0), 1)

        cv2.putText(
            frame,
            f"VISTA: Shi-Tomasi Detector",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 200),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Detected Features: {feature_count}",
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Processing Speed: {fps:.1f} FPS",
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("VISTA - Visual Feature Detection Baseline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[VISTA] Session terminated cleanly.")

if __name__ == "__main__":
    main()
