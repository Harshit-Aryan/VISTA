"""
================================================================================
VISTA: Vision-based Inertial-free Spatial Tracking Architecture
Module: vista_optical_flow.py
Phase 3 Milestone: Lucas-Kanade Optical Flow Feature Tracking & Motion Estimation
================================================================================

Description:
    Demonstrates GPS-denied camera ego-motion tracking using:
    1. Shi-Tomasi corner detection (cv2.goodFeaturesToTrack)
    2. Lucas-Kanade Pyramidal Optical Flow (cv2.calcOpticalFlowPyrLK)
    3. Trajectory motion trails visualization with decay
    4. Dynamic feature reinitialization upon track loss
    5. Real-time telemetry: FPS, Feature Count, Retention %, Dominant Motion Direction
================================================================================
"""

import cv2
import numpy as np
import time
import os
import argparse
import csv

# -----------------------------------------------------------------------------
# Configuration & Algorithm Hyperparameters
# -----------------------------------------------------------------------------

# Shi-Tomasi Corner Detector Parameters
FEATURE_PARAMS = {
    "maxCorners": 150,      # Maximum number of corners to detect
    "qualityLevel": 0.01,   # Minimum accepted quality of corners (relative to max)
    "minDistance": 10,      # Minimum Euclidean distance between corners
    "blockSize": 7          # Size of averaging block for derivative estimation
}

# Lucas-Kanade Pyramidal Optical Flow Parameters
LK_PARAMS = {
    "winSize": (21, 21),    # Search window size at each pyramid level
    "maxLevel": 3,          # Multi-scale pyramid levels (handles larger motion)
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
}

# Feature Reinitialization & Tracking Thresholds
MIN_FEATURE_THRESHOLD = 35   # Trigger reinitialization when tracked features fall below this
TRAIL_DECAY_RATE = 0.94      # Trail decay factor per frame (0.90 to 0.98 creates smooth fading)


def detect_features(gray_frame, feature_params):
    """
    Detects prominent Shi-Tomasi corner points in a grayscale image frame.
    Returns:
        np.ndarray: Array of shape (N, 1, 2) of float32 feature coordinates,
                    or None if no features found.
    """
    corners = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
    return corners


def estimate_dominant_motion(displacements):
    """
    Estimates dominant image-plane motion vector and direction category from
    feature displacement vectors [dx, dy].

    Returns:
        tuple: (median_dx, median_dy, direction_label)
    """
    if len(displacements) == 0:
        return 0.0, 0.0, "STATIONARY"

    disp_arr = np.asarray(displacements).reshape(-1, 2)
    med_dx = float(np.median(disp_arr[:, 0]))
    med_dy = float(np.median(disp_arr[:, 1]))
    mag = np.hypot(med_dx, med_dy)

    # Threshold for motion detection (in pixels)
    if mag < 0.8:
        direction = "STATIONARY"
    else:
        dirs = []
        # In image coordinates: +Y is downward, +X is rightward
        if med_dy < -0.8:
            dirs.append("UP")
        elif med_dy > 0.8:
            dirs.append("DOWN")

        if med_dx < -0.8:
            dirs.append("LEFT")
        elif med_dx > 0.8:
            dirs.append("RIGHT")

        direction = "-".join(dirs) if dirs else "STATIONARY"

    return med_dx, med_dy, direction


def draw_hud(frame, fps, detected_cnt, tracked_cnt, retention_pct, med_dx, med_dy, motion_dir):
    """
    Renders an engineering telemetry Heads-Up Display (HUD) overlay on the video frame.
    """
    h, w = frame.shape[:2]

    # Create semi-transparent telemetry background card
    overlay = frame.copy()
    card_w, card_h = 360, 165
    cv2.rectangle(overlay, (12, 12), (12 + card_w, 12 + card_h), (18, 18, 22), -1)
    cv2.rectangle(overlay, (12, 12), (12 + card_w, 12 + card_h), (60, 60, 75), 1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent indicator bar (Emerald Green)
    cv2.rectangle(frame, (12, 12), (17, 12 + card_h), (0, 230, 115), -1)

    # Header
    cv2.putText(frame, "VISTA | Optical Flow Tracker", (26, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # Metrics
    # 1. Processing Speed
    fps_color = (0, 255, 128) if fps >= 15.0 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (26, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, fps_color, 1, cv2.LINE_AA)

    # 2. Detected & Tracked Count
    cv2.putText(frame, f"Features: {tracked_cnt}/{detected_cnt} tracked", (26, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

    # 3. Retention Rate
    ret_color = (0, 255, 128) if retention_pct >= 70.0 else ((0, 215, 255) if retention_pct >= 40.0 else (0, 70, 255))
    cv2.putText(frame, f"Retention: {retention_pct:.1f}%", (26, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, ret_color, 1, cv2.LINE_AA)

    # 4. Image-Plane Relative Motion
    cv2.putText(frame, f"Motion: {motion_dir} (dx:{med_dx:+.1f}, dy:{med_dy:+.1f}px)", (26, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 225, 255), 1, cv2.LINE_AA)

    # 5. Controls Guide
    cv2.putText(frame, "[Q] Quit  |  [R] Reinit  |  [C] Clear  |  [S] Save", (26, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="VISTA Lucas-Kanade Optical Flow Feature Tracker")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or path to video file")
    parser.add_argument("--csv", default="results/test_results.csv", help="Path to save experimental metrics CSV")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without cv2.imshow GUI")
    args = parser.parse_args()

    print("=" * 70)
    print("VISTA: GPS-Denied Visual Navigation Research Prototype")
    print("Milestone: Lucas-Kanade Optical Flow Feature Tracking")
    print("=" * 70)

    # Determine video source (integer index or file path)
    source = int(args.source) if args.source.isdigit() else args.source

    # Prepare CSV telemetry logging
    csv_file = None
    csv_writer = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) if os.path.dirname(args.csv) else ".", exist_ok=True)
        csv_file = open(args.csv, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_idx", "timestamp", "fps", "detected_count",
            "tracked_count", "retention_pct", "median_dx", "median_dy", "motion_dir"
        ])
        print(f"[VISTA] Telemetry logging enabled -> {args.csv}")

    # Initialize video capture
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Video source '{source}' could not be opened.")
        print("[INFO] Please verify webcam permissions, device index, or file path.")
        return

    # Request camera resolution if capturing live stream
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read the first reference frame
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("[ERROR] Failed to read initial frame from camera.")
        cap.release()
        return

    # Convert initial frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial Shi-Tomasi feature points
    p0 = detect_features(prev_gray, FEATURE_PARAMS)
    initial_detected_count = len(p0) if p0 is not None else 0

    if p0 is None or initial_detected_count == 0:
        print("[WARN] Low texture scene: No features detected in initial frame.")
        p0 = np.empty((0, 1, 2), dtype=np.float32)

    # Create a canvas mask for drawing persistent motion trails
    trail_mask = np.zeros_like(prev_frame)

    # Color palette for tracks (cycling HSV color spectrum)
    color_palette = np.random.randint(0, 255, (FEATURE_PARAMS["maxCorners"], 3))

    # Real-time performance tracking
    prev_time = time.perf_counter()
    fps = 30.0
    frame_idx = 0

    print(f"[VISTA] Initialized with {initial_detected_count} detected corner features.")
    print("[VISTA] Tracking active. Press 'q' to quit, 'r' to reinitialize features.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Video stream ended or frame dropped.")
            break

        frame_idx += 1
        curr_time = time.perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Convert incoming frame to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracked_count = 0
        displacements = []
        med_dx, med_dy, motion_dir = 0.0, 0.0, "STATIONARY"

        # Apply trail decay for phosphorescent trailing effect
        trail_mask = (trail_mask * TRAIL_DECAY_RATE).astype(np.uint8)

        # If we have points to track, compute Lucas-Kanade optical flow
        if p0 is not None and len(p0) > 0:
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                p0,
                None,
                **LK_PARAMS
            )

            # Filter valid tracked points (status == 1)
            if p1 is not None and status is not None:
                good_curr = p1[status.ravel() == 1]
                good_prev = p0[status.ravel() == 1]
                tracked_count = len(good_curr)

                # Draw motion trails and vectors
                for i, (curr_pt, prev_pt) in enumerate(zip(good_curr, good_prev)):
                    cx, cy = curr_pt.ravel()
                    px, py = prev_pt.ravel()

                    # Calculate per-feature displacement
                    dx = cx - px
                    dy = cy - py
                    displacements.append([dx, dy])

                    # Select color from palette
                    color = color_palette[i % len(color_palette)].tolist()

                    # Draw line trail on persistent mask
                    cv2.line(
                        trail_mask,
                        (int(px), int(py)),
                        (int(cx), int(cy)),
                        color,
                        2,
                        cv2.LINE_AA
                    )

                    # Draw current tracked feature point on frame
                    cv2.circle(
                        frame,
                        (int(cx), int(cy)),
                        4,
                        (0, 255, 0),  # Crisp green point
                        -1
                    )

                if len(displacements) > 0:
                    displacements_np = np.array(displacements)
                    med_dx, med_dy, motion_dir = estimate_dominant_motion(displacements_np)

                # Update tracked points for next frame iteration
                p0 = good_curr.reshape(-1, 1, 2)
            else:
                p0 = np.empty((0, 1, 2), dtype=np.float32)
        else:
            p0 = np.empty((0, 1, 2), dtype=np.float32)

        # Compute feature retention rate
        retention_pct = (tracked_count / initial_detected_count * 100.0) if initial_detected_count > 0 else 0.0

        # Automatic Reinitialization check
        # If tracked points drop below minimum threshold, re-detect features
        if tracked_count < MIN_FEATURE_THRESHOLD:
            new_corners = detect_features(curr_gray, FEATURE_PARAMS)
            if new_corners is not None and len(new_corners) > 0:
                p0 = new_corners
                initial_detected_count = len(p0)
                tracked_count = initial_detected_count
                retention_pct = 100.0

        # Composite motion trails onto current video frame
        output_display = cv2.add(frame, trail_mask)

        # Render telemetry HUD
        draw_hud(
            output_display,
            fps,
            initial_detected_count,
            tracked_count,
            retention_pct,
            med_dx,
            med_dy,
            motion_dir
        )

        # Log metrics to CSV if enabled
        if csv_writer:
            csv_writer.writerow([
                frame_idx,
                f"{time.time():.4f}",
                f"{fps:.2f}",
                initial_detected_count,
                tracked_count,
                f"{retention_pct:.2f}",
                f"{med_dx:.2f}",
                f"{med_dy:.2f}",
                motion_dir
            ])

        # Display output window unless headless
        if not args.headless:
            try:
                cv2.imshow("VISTA - Lucas-Kanade Optical Flow Feature Tracker", output_display)
            except cv2.error:
                pass

        # Update previous frame grayscale buffer
        prev_gray = curr_gray.copy()

        # Keyboard interaction handler
        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[VISTA] Quit commanded by user.")
                break
            elif key == ord('r'):
                # Manual reinitialization
                print("[VISTA] Manual feature reinitialization triggered.")
                new_corners = detect_features(curr_gray, FEATURE_PARAMS)
                if new_corners is not None and len(new_corners) > 0:
                    p0 = new_corners
                    initial_detected_count = len(p0)
                trail_mask = np.zeros_like(frame)
            elif key == ord('c'):
                # Clear motion trails
                print("[VISTA] Trails cleared.")
                trail_mask = np.zeros_like(frame)
            elif key == ord('s'):
                # Save screenshot for laboratory report / TRL 4 validation
                os.makedirs("results/screenshots", exist_ok=True)
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join("results/screenshots", f"vista_track_{timestamp_str}.png")
                cv2.imwrite(screenshot_path, output_display)
                print(f"[VISTA] Screenshot saved to: {screenshot_path}")

    # Release resources
    if csv_file:
        csv_file.close()
        print(f"[VISTA] Telemetry CSV written with {frame_idx} frames of data.")

    cap.release()
    cv2.destroyAllWindows()
    print("[VISTA] Camera released and windows closed. System shut down cleanly.")


if __name__ == "__main__":
    main()
