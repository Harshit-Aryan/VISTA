"""
================================================================================
VISTA: Vision-based Inertial-free Spatial Tracking Architecture
Module: app.py
Phase 5: Real-Time Visualization Dashboard for GPS-Denied Visual Navigation
================================================================================
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import pandas as pd
import tempfile

# Page configuration
st.set_page_config(
    page_title="VISTA | GPS-Denied Navigation Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-tech telemetry dashboard theme
st.markdown("""
<style>
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #58a6ff;
        font-size: 1.6rem;
        font-weight: 700;
        font-family: monospace;
    }
    .status-ok { color: #3fb950 !important; }
    .status-warn { color: #d29922 !important; }
    .status-alert { color: #f85149 !important; }
</style>
""", unsafe_allow_html=True)


def estimate_dominant_motion(displacements):
    """
    Estimates dominant image-plane motion vector and direction category.
    """
    if len(displacements) == 0:
        return 0.0, 0.0, "STATIONARY"

    disp_arr = np.asarray(displacements).reshape(-1, 2)
    med_dx = float(np.median(disp_arr[:, 0]))
    med_dy = float(np.median(disp_arr[:, 1]))
    mag = np.hypot(med_dx, med_dy)

    if mag < 0.8:
        direction = "STATIONARY"
    else:
        dirs = []
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


def main():
    # Header Banner
    st.title("🛰️ VISTA — Spatial Tracking Dashboard")
    st.caption("Vision-based Inertial-free Spatial Tracking Architecture • TRL 4 Research Prototype")

    # -------------------------------------------------------------------------
    # Sidebar Configuration
    # -------------------------------------------------------------------------
    st.sidebar.header("🕹️ Video Input Source")
    input_source = st.sidebar.radio(
        "Select Source",
        ["Live Webcam", "Sample Flight Video", "Upload Video File"],
        index=0
    )

    custom_video_path = None
    if input_source == "Live Webcam":
        camera_idx = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
    elif input_source == "Sample Flight Video":
        sample_path = "data/test_videos/sample_flight.mp4"
        if not os.path.exists(sample_path):
            st.sidebar.error("Sample flight video not found. Generating default...")
        custom_video_path = sample_path
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Drone Flight MP4/AVI", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            custom_video_path = tfile.name

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Feature Detector (Shi-Tomasi)")
    max_corners = st.sidebar.slider("Max Corners (maxCorners)", 30, 300, 150, step=10)
    quality_level = st.sidebar.slider("Quality Level", 0.001, 0.1, 0.01, step=0.005, format="%.3f")
    min_distance = st.sidebar.slider("Min Distance (px)", 5, 30, 10, step=1)

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Optical Flow (Lucas-Kanade)")
    win_size_val = st.sidebar.selectbox("Window Size", [15, 21, 31], index=1)
    pyramid_levels = st.sidebar.slider("Pyramid Levels (maxLevel)", 1, 4, 3, step=1)
    reinit_threshold = st.sidebar.slider("Reinit Threshold (Min Features)", 15, 60, 35, step=5)
    trail_decay = st.sidebar.slider("Trail Decay Rate", 0.85, 0.99, 0.94, step=0.01)

    # -------------------------------------------------------------------------
    # Dashboard Layout: Video Stream & Telemetry
    # -------------------------------------------------------------------------
    col_video, col_telemetry = st.columns([7, 3])

    with col_video:
        video_placeholder = st.empty()
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        run_tracking = col_btn1.button("▶️ Start Tracking", key="start_btn", type="primary")
        stop_tracking = col_btn2.button("⏹️ Stop Tracking", key="stop_btn")
        reinit_trigger = col_btn3.button("🔄 Force Reinit", key="reinit_btn")

    with col_telemetry:
        st.subheader("📊 Live Telemetry")
        metric_fps = st.empty()
        metric_features = st.empty()
        metric_retention = st.empty()
        metric_motion = st.empty()
        metric_vector = st.empty()

    st.markdown("---")
    st.subheader("📈 Flight Trajectory & Performance Analytics")
    chart_col1, chart_col2 = st.columns(2)
    chart_retention = chart_col1.empty()
    chart_motion = chart_col2.empty()

    # Session State tracking flags
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "force_reinit" not in st.session_state:
        st.session_state.force_reinit = False

    if run_tracking:
        st.session_state.is_running = True
    if stop_tracking:
        st.session_state.is_running = False
    if reinit_trigger:
        st.session_state.force_reinit = True

    # Run Loop if active
    if st.session_state.is_running:
        # Determine VideoCapture target
        if input_source == "Live Webcam":
            cap = cv2.VideoCapture(int(camera_idx))
        else:
            if not custom_video_path or not os.path.exists(custom_video_path):
                st.error("Selected video file does not exist.")
                st.session_state.is_running = False
                return
            cap = cv2.VideoCapture(custom_video_path)

        if not cap.isOpened():
            st.error("Unable to open selected video source. Please check camera permissions or video file.")
            st.session_state.is_running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        feature_params = {
            "maxCorners": int(max_corners),
            "qualityLevel": float(quality_level),
            "minDistance": int(min_distance),
            "blockSize": 7
        }

        lk_params = {
            "winSize": (int(win_size_val), int(win_size_val)),
            "maxLevel": int(pyramid_levels),
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        }

        ret, prev_frame = cap.read()
        if not ret or prev_frame is None:
            st.warning("Failed to grab initial frame.")
            cap.release()
            st.session_state.is_running = False
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        initial_detected = len(p0) if p0 is not None else 0
        trail_mask = np.zeros_like(prev_frame)
        color_palette = np.random.randint(50, 255, (feature_params["maxCorners"], 3))

        history_df = pd.DataFrame(columns=["Frame", "FPS", "Retention", "dx", "dy"])
        frame_idx = 0
        prev_time = time.perf_counter()
        fps = 30.0

        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video if pre-recorded
                if input_source != "Live Webcam":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_idx += 1
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt)

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            trail_mask = (trail_mask * float(trail_decay)).astype(np.uint8)

            tracked_count = 0
            displacements = []
            med_dx, med_dy, motion_dir = 0.0, 0.0, "STATIONARY"

            # Check if forced reinit was requested
            if st.session_state.force_reinit:
                p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
                initial_detected = len(p0) if p0 is not None else 0
                trail_mask = np.zeros_like(frame)
                st.session_state.force_reinit = False

            if p0 is not None and len(p0) > 0:
                p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
                if p1 is not None and status is not None:
                    good_curr = p1[status.ravel() == 1]
                    good_prev = p0[status.ravel() == 1]
                    tracked_count = len(good_curr)

                    for i, (cp, pp) in enumerate(zip(good_curr, good_prev)):
                        cx, cy = cp.ravel()
                        px, py = pp.ravel()
                        displacements.append([cx - px, cy - py])
                        col = color_palette[i % len(color_palette)].tolist()
                        cv2.line(trail_mask, (int(px), int(py)), (int(cx), int(cy)), col, 2, cv2.LINE_AA)
                        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

                    if len(displacements) > 0:
                        med_dx, med_dy, motion_dir = estimate_dominant_motion(displacements)

                    p0 = good_curr.reshape(-1, 1, 2)
                else:
                    p0 = np.empty((0, 1, 2), dtype=np.float32)
            else:
                p0 = np.empty((0, 1, 2), dtype=np.float32)

            retention_pct = (tracked_count / initial_detected * 100.0) if initial_detected > 0 else 0.0

            # Auto reinit if dropped below threshold
            if tracked_count < int(reinit_threshold):
                new_p = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
                if new_p is not None and len(new_p) > 0:
                    p0 = new_p
                    initial_detected = len(p0)
                    tracked_count = initial_detected
                    retention_pct = 100.0

            # Composite image
            display_frame = cv2.add(frame, trail_mask)
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Update live video canvas
            video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

            # Update Telemetry Display
            fps_color = "status-ok" if fps >= 15 else "status-warn"
            ret_color = "status-ok" if retention_pct >= 70 else ("status-warn" if retention_pct >= 40 else "status-alert")

            metric_fps.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Processing Throughput</div>
                <div class="metric-val {fps_color}">{fps:.1f} <span style="font-size:1rem;color:#888;">FPS</span></div>
            </div>
            """, unsafe_allow_html=True)

            metric_features.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tracked Landmarks</div>
                <div class="metric-val">{tracked_count} <span style="font-size:1rem;color:#888;">/ {initial_detected}</span></div>
            </div>
            """, unsafe_allow_html=True)

            metric_retention.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Feature Retention</div>
                <div class="metric-val {ret_color}">{retention_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            metric_motion.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dominant Image Motion</div>
                <div class="metric-val" style="color:#d2a8ff;">{motion_dir}</div>
            </div>
            """, unsafe_allow_html=True)

            metric_vector.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Median Displacement</div>
                <div class="metric-val" style="font-size:1.2rem;">dx: {med_dx:+.1f}px &nbsp;|&nbsp; dy: {med_dy:+.1f}px</div>
            </div>
            """, unsafe_allow_html=True)

            # Record telemetry point
            new_row = pd.DataFrame([{
                "Frame": frame_idx,
                "FPS": round(fps, 1),
                "Retention": round(retention_pct, 1),
                "dx": round(med_dx, 2),
                "dy": round(med_dy, 2)
            }])
            history_df = pd.concat([history_df, new_row], ignore_index=True)

            # Render charts periodically
            if frame_idx % 10 == 0 and len(history_df) > 1:
                recent_history = history_df.tail(60).set_index("Frame")
                chart_retention.line_chart(recent_history[["Retention"]], color="#3fb950")
                chart_motion.line_chart(recent_history[["dx", "dy"]], color=["#58a6ff", "#f0883e"])

            prev_gray = curr_gray.copy()
            time.sleep(0.01)

        cap.release()
    else:
        video_placeholder.info("Click **▶️ Start Tracking** to begin real-time feature tracking and telemetry stream.")


if __name__ == "__main__":
    main()
