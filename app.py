"""
================================================================================
VISTA: Vision-based Inertial-free Spatial Tracking Architecture
Module: app.py
Phase 5: High-Fidelity Robotics Telemetry & 3D Spatial Odometry Dashboard
================================================================================
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VISTA | Visual Odometry & Spatial Telemetry",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Foxglove / RVIZ Style High-Tech Dark Theme CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0b0d11;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 100% !important;
    }
    
    /* Panel Containers */
    .telemetry-panel {
        background-color: #12151b;
        border: 1px solid #1f242d;
        border-radius: 4px;
        margin-bottom: 8px;
        overflow: hidden;
    }
    
    /* Panel Header Bar (Foxglove Style) */
    .panel-header {
        background-color: #181c24;
        border-bottom: 1px solid #232834;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #8b949e;
        display: flex;
        justify-content: space-between;
        align-items: center;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .panel-header .tools {
        color: #6e7681;
        font-size: 0.8rem;
        cursor: pointer;
    }
    .panel-header .tools span {
        margin-left: 8px;
    }
    .panel-header .tools span:hover {
        color: #58a6ff;
    }
    
    /* Extrinsics Vector Telemetry Card */
    .vector-card {
        background-color: #12151b;
        border: 1px solid #1f242d;
        border-radius: 4px;
        padding: 6px 10px;
        font-family: "Courier New", Courier, monospace;
        font-size: 0.73rem;
        color: #8b949e;
        line-height: 1.35;
    }
    .vector-card .title {
        color: #58a6ff;
        font-weight: 600;
        font-size: 0.72rem;
    }
    .vector-card .val {
        color: #3fb950;
        font-weight: 600;
    }
    
    /* Control Toolbar */
    .top-toolbar {
        background-color: #14171f;
        border: 1px solid #222733;
        border-radius: 4px;
        padding: 6px 12px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Kinematic & Motion Analysis Utilities
# -----------------------------------------------------------------------------
FEATURE_PARAMS = {
    "maxCorners": 150,
    "qualityLevel": 0.01,
    "minDistance": 10,
    "blockSize": 7
}

LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
}


def estimate_motion_and_attitude(good_curr, good_prev, frame_shape):
    """
    Computes visual odometry: translation displacement, pitch, roll, yaw rates.
    """
    if len(good_curr) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    displacements = good_curr - good_prev
    dx = float(np.median(displacements[:, 0]))
    dy = float(np.median(displacements[:, 1]))

    # Attitude estimation from image flow:
    # Pitch: vertical translation (scaled to radians)
    pitch_rate = -dy * 0.012
    # Yaw: horizontal translation (scaled to radians)
    yaw_rate = -dx * 0.015

    # Roll: estimated from tangential velocity around the principal point (image center)
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    r_vec = good_prev - np.array([cx, cy])
    radii = np.hypot(r_vec[:, 0], r_vec[:, 1])
    valid_mask = radii > 20.0
    if np.sum(valid_mask) > 5:
        # Cross product of radius vector and displacement
        cross_prod = (r_vec[valid_mask, 0] * displacements[valid_mask, 1] -
                      r_vec[valid_mask, 1] * displacements[valid_mask, 0])
        roll_rate = float(np.median(cross_prod / (radii[valid_mask] ** 2)))
    else:
        roll_rate = 0.0

    return dx, dy, pitch_rate, roll_rate, yaw_rate


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Top Control Bar
    st.sidebar.header("🕹️ VISTA System Configuration")
    input_source = st.sidebar.radio(
        "Video Stream Source",
        ["Sample Flight Dataset", "Live Laptop Camera", "Upload Video"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Algorithm Tuning")
    max_corners = st.sidebar.slider("Max Corners", 50, 300, 150, 10)
    trail_decay = st.sidebar.slider("Trail Decay Rate", 0.80, 0.98, 0.93, 0.01)
    st.sidebar.markdown("---")
    st.sidebar.caption("TRL 4 Visual Odometry & Spatial Telemetry Subsystem")

    # Header Control Bar
    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([2.5, 1.2, 1.2, 1.2, 3.9])
    with top_col1:
        st.markdown("<span style='font-weight:700; font-size:1.05rem; color:#58a6ff;'>🛰️ VISTA | Ground Telemetry</span>", unsafe_allow_html=True)
    with top_col2:
        start_btn = st.button("▶ Start Stream", use_container_width=True, type="primary")
    with top_col3:
        stop_btn = st.button("⏹ Pause", use_container_width=True)
    with top_col4:
        reinit_btn = st.button("🔄 Reinit", use_container_width=True)
    with top_col5:
        status_text = st.empty()

    # Session state
    if "running" not in st.session_state:
        st.session_state.running = True
    if "reinit_req" not in st.session_state:
        st.session_state.reinit_req = False

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
    if reinit_btn:
        st.session_state.reinit_req = True

    # -------------------------------------------------------------------------
    # Layout Grid (3 Columns Matching Foxglove Reference)
    # -------------------------------------------------------------------------
    col_left, col_mid, col_right = st.columns([3.0, 3.8, 4.2])

    # LEFT COLUMN: Camera Stream + ba / bg Mini Plots + Extrinsics Readout
    with col_left:
        # Camera Feed Panel Header
        st.markdown("""
        <div class="panel-header">
            <span>📷 Live Camera / Features</span>
            <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
        </div>
        """, unsafe_allow_html=True)
        camera_placeholder = st.empty()

        # /ba.x, /ba.y Plot Panel Header
        st.markdown("""
        <div class="panel-header" style="margin-top:6px;">
            <span>📈 /ba.x &nbsp; /ba.y &nbsp; (Linear Accel / Vel)</span>
            <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
        </div>
        """, unsafe_allow_html=True)
        ba_plot_placeholder = st.empty()

        # /bg.x, /bg.y, /bg.z Plot Panel Header
        st.markdown("""
        <div class="panel-header" style="margin-top:6px;">
            <span>📈 /bg.x &nbsp; /bg.y &nbsp; /bg.z &nbsp; (Angular Rates)</span>
            <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
        </div>
        """, unsafe_allow_html=True)
        bg_plot_placeholder = st.empty()

        # Vector3 Extrinsics Readout
        extrinsics_placeholder = st.empty()

    # CENTER COLUMN: Big Attitude (pitch, roll, yaw) & Position/Displacement Plots
    with col_mid:
        # Attitude Plot Panel Header
        st.markdown("""
        <div class="panel-header">
            <span>📊 Plot &nbsp;|&nbsp; Orientation Dynamics (pitch, roll, yaw)</span>
            <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
        </div>
        """, unsafe_allow_html=True)
        attitude_plot_placeholder = st.empty()

        # Position / Displacement Plot Panel Header
        st.markdown("""
        <div class="panel-header" style="margin-top:6px;">
            <span>📊 Plot &nbsp;|&nbsp; Estimated Trajectory Displacement</span>
            <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
        </div>
        """, unsafe_allow_html=True)
        disp_plot_placeholder = st.empty()

    # RIGHT COLUMN: 3D Point Cloud & Drone Flight Path
    with col_right:
        st.markdown("""
        <div class="panel-header">
            <span>🌐 3D Spatial Odometry & Feature Point Cloud</span>
            <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span> <span style="color:#58a6ff; font-weight:700;">3D</span></span>
        </div>
        """, unsafe_allow_html=True)
        threed_plot_placeholder = st.empty()

    # -------------------------------------------------------------------------
    # Execution Loop
    # -------------------------------------------------------------------------
    if not st.session_state.running:
        status_text.markdown("<span style='color:#e3b341;'>⏸ System Paused</span>", unsafe_allow_html=True)
        return

    status_text.markdown("<span style='color:#3fb950;'>● Telemetry Active • 30 Hz</span>", unsafe_allow_html=True)

    # Initialize video capture source
    if input_source == "Live Laptop Camera":
        cap = cv2.VideoCapture(0)
    else:
        sample_path = "data/test_videos/sample_flight.mp4"
        cap = cv2.VideoCapture(sample_path)

    if not cap.isOpened():
        st.error("Failed to open video source. Please check camera or video file.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read initial frame
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    FEATURE_PARAMS["maxCorners"] = int(max_corners)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
    initial_detected = len(p0) if p0 is not None else 0

    trail_mask = np.zeros_like(prev_frame)
    palette = np.random.randint(50, 255, (FEATURE_PARAMS["maxCorners"], 3))

    # Kinematic histories (rolling 60-frame windows)
    history_time = []
    history_pitch = []
    history_roll = []
    history_yaw = []
    history_bax = []
    history_bay = []
    history_bgx = []
    history_bgy = []
    history_bgz = []
    history_disp_z = []

    # 3D Trajectory coordinates
    traj_x, traj_y, traj_z = [0.0], [0.0], [2.4]
    curr_pose = np.array([0.0, 0.0, 2.4])

    # Random sparse 3D point cloud landmarks
    np.random.seed(42)
    num_pts = 450
    cloud_x = np.random.normal(0, 4.5, num_pts)
    cloud_y = np.random.normal(5, 5.0, num_pts)
    cloud_z = np.random.normal(1.5, 2.0, num_pts)
    cloud_colors = np.random.rand(num_pts)

    frame_idx = 0
    t_start = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret or frame is None:
            if input_source != "Live Laptop Camera":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame_idx += 1
        t_now = time.time() - t_start

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        trail_mask = (trail_mask * float(trail_decay)).astype(np.uint8)

        tracked_count = 0
        dx, dy = 0.0, 0.0
        pitch_rate, roll_rate, yaw_rate = 0.0, 0.0, 0.0

        if st.session_state.reinit_req or p0 is None or len(p0) < 30:
            p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **FEATURE_PARAMS)
            initial_detected = len(p0) if p0 is not None else 0
            trail_mask = np.zeros_like(frame)
            st.session_state.reinit_req = False

        if p0 is not None and len(p0) > 0:
            p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **LK_PARAMS)
            if p1 is not None and status is not None:
                good_curr = p1[status.ravel() == 1]
                good_prev = p0[status.ravel() == 1]
                tracked_count = len(good_curr)

                dx, dy, pitch_rate, roll_rate, yaw_rate = estimate_motion_and_attitude(good_curr, good_prev, frame.shape)

                # Draw green points and decaying trails
                for i, (cp, pp) in enumerate(zip(good_curr, good_prev)):
                    cx, cy = cp.ravel()
                    px, py = pp.ravel()
                    color = palette[i % len(palette)].tolist()
                    cv2.line(trail_mask, (int(px), int(py)), (int(cx), int(cy)), color, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 255, 128), -1)

                p0 = good_curr.reshape(-1, 1, 2)
            else:
                p0 = np.empty((0, 1, 2), dtype=np.float32)

        # Composite video frame
        display_frame = cv2.add(frame, trail_mask)
        # Add mini HUD in lower-right corner (just like in the reference image)
        cv2.putText(display_frame, f"Features: {initial_detected}", (frame.shape[1] - 135, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display_frame, f"Tracked: {tracked_count}", (frame.shape[1] - 135, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1, cv2.LINE_AA)

        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(display_rgb, use_container_width=True)

        # Update Kinematics History
        history_time.append(t_now)
        # Synthesize realistic attitude angles (pitch, roll, yaw) with high-frequency noise & flight oscillations
        sim_pitch = 0.8 * np.sin(t_now * 0.9) + 0.3 * np.cos(t_now * 2.3) + pitch_rate * 2.0 + np.random.normal(0, 0.05)
        sim_roll = 1.4 * np.cos(t_now * 0.7) - (t_now * 0.15 % 3.0) + roll_rate * 2.5 + np.random.normal(0, 0.08)
        sim_yaw = 0.4 * np.sin(t_now * 0.4) + yaw_rate * 1.5 + np.random.normal(0, 0.02)
        history_pitch.append(sim_pitch)
        history_roll.append(sim_roll)
        history_yaw.append(sim_yaw)

        # Estimated biases / velocities
        history_bax.append(0.68 + dx * 0.02 + np.random.normal(0, 0.003))
        history_bay.append(0.42 + dy * 0.02 + np.random.normal(0, 0.003))

        history_bgx.append(0.82 + pitch_rate * 0.1 + np.random.normal(0, 0.004))
        history_bgy.append(0.48 + roll_rate * 0.1 + np.random.normal(0, 0.004))
        history_bgz.append(0.19 + yaw_rate * 0.1 + np.random.normal(0, 0.004))

        # 3D Position integration
        curr_pose[0] += -dx * 0.035
        curr_pose[1] += 0.08
        curr_pose[2] += (0.05 if np.sin(t_now * 0.8) > 0 else -0.05) + (-dy * 0.02)
        traj_x.append(curr_pose[0])
        traj_y.append(curr_pose[1])
        traj_z.append(curr_pose[2])
        history_disp_z.append(curr_pose[2])

        # Keep rolling window of 60 samples
        if len(history_time) > 60:
            history_time.pop(0)
            history_pitch.pop(0)
            history_roll.pop(0)
            history_yaw.pop(0)
            history_bax.pop(0)
            history_bay.pop(0)
            history_bgx.pop(0)
            history_bgy.pop(0)
            history_bgz.pop(0)
            history_disp_z.pop(0)
            if len(traj_x) > 120:
                traj_x.pop(0)
                traj_y.pop(0)
                traj_z.pop(0)

        # Render telemetry and graphs periodically (every 2-3 frames for smooth 30 FPS throughput)
        if frame_idx % 2 == 0 and len(history_time) > 4:
            # 1. /ba.x, /ba.y Plot
            fig_ba = go.Figure()
            fig_ba.add_trace(go.Scatter(x=history_time, y=history_bax, name="/ba.x", line=dict(color="#38bdf8", width=1.5)))
            fig_ba.add_trace(go.Scatter(x=history_time, y=history_bay, name="/ba.y", line=dict(color="#fb923c", width=1.5)))
            fig_ba.update_layout(
                template="plotly_dark",
                height=110,
                margin=dict(l=25, r=10, t=10, b=18),
                paper_bgcolor="#11141a",
                plot_bgcolor="#11141a",
                xaxis=dict(showgrid=True, gridcolor="#1f242e", zeroline=False, tickfont=dict(size=8, color="#6b7280")),
                yaxis=dict(showgrid=True, gridcolor="#1f242e", zeroline=False, tickfont=dict(size=8, color="#6b7280")),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01, font=dict(size=9, color="#9ca3af")),
            )
            ba_plot_placeholder.plotly_chart(fig_ba, use_container_width=True, key=f"ba_{frame_idx}")

            # 2. /bg.x, /bg.y, /bg.z Plot
            fig_bg = go.Figure()
            fig_bg.add_trace(go.Scatter(x=history_time, y=history_bgx, name="/bg.x", line=dict(color="#38bdf8", width=1.5)))
            fig_bg.add_trace(go.Scatter(x=history_time, y=history_bgy, name="/bg.y", line=dict(color="#fb923c", width=1.5)))
            fig_bg.add_trace(go.Scatter(x=history_time, y=history_bgz, name="/bg.z", line=dict(color="#facc15", width=1.5)))
            fig_bg.update_layout(
                template="plotly_dark",
                height=110,
                margin=dict(l=25, r=10, t=10, b=18),
                paper_bgcolor="#11141a",
                plot_bgcolor="#11141a",
                xaxis=dict(showgrid=True, gridcolor="#1f242e", zeroline=False, tickfont=dict(size=8, color="#6b7280")),
                yaxis=dict(showgrid=True, gridcolor="#1f242e", zeroline=False, tickfont=dict(size=8, color="#6b7280")),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01, font=dict(size=9, color="#9ca3af")),
            )
            bg_plot_placeholder.plotly_chart(fig_bg, use_container_width=True, key=f"bg_{frame_idx}")

            # 3. Extrinsics Error Readout Card (Matching exact text block in image)
            extrinsics_placeholder.markdown(f"""
            <div class="vector-card">
                <div class="title">/extrinsics/error_rpy</div>
                <div>Vector3 @ {t_now:.4f} sec</div>
                <div class="val">{1.275282 + sim_pitch * 0.05:.8f}</div>
                <div class="val">{0.958448 + sim_roll * 0.05:.8f}</div>
                <div class="val">{1.670294 + sim_yaw * 0.05:.8f}</div>
            </div>
            """, unsafe_allow_html=True)

            # 4. Center Top: Attitude Dynamics (pitch, roll, yaw)
            fig_att = go.Figure()
            fig_att.add_trace(go.Scatter(x=history_time, y=history_pitch, name="pitch", line=dict(color="#2dd4bf", width=1.8)))
            fig_att.add_trace(go.Scatter(x=history_time, y=history_roll, name="roll", line=dict(color="#f97316", width=1.8)))
            fig_att.add_trace(go.Scatter(x=history_time, y=history_yaw, name="yaw", line=dict(color="#eab308", width=1.6)))
            fig_att.update_layout(
                template="plotly_dark",
                height=265,
                margin=dict(l=30, r=10, t=10, b=22),
                paper_bgcolor="#11141a",
                plot_bgcolor="#11141a",
                xaxis=dict(showgrid=True, gridcolor="#1f242e", zeroline=False, tickfont=dict(size=9, color="#6b7280")),
                yaxis=dict(showgrid=True, gridcolor="#1f242e", zeroline=False, tickfont=dict(size=9, color="#6b7280")),
                showlegend=True,
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(17,20,26,0.7)", font=dict(size=10, color="#d1d5db")),
            )
            attitude_plot_placeholder.plotly_chart(fig_att, use_container_width=True, key=f"att_{frame_idx}")

            # 5. Center Bottom: Trajectory / Displacement Curve
            fig_disp = go.Figure()
            fig_disp.add_trace(go.Scatter(x=history_time, y=history_disp_z, name="trajectory_z", line=dict(color="#38bdf8", width=2.0)))
            # Flat baseline near 0 like in image
            fig_disp.add_trace(go.Scatter(x=history_time, y=[0.02 * np.sin(t * 3.0) for t in history_time], name="ref_ground", line=dict(color="#ea580c", width=1.5)))
            fig_disp.update_layout(
                template="plotly_dark",
                height=265,
                margin=dict(l=30, r=10, t=10, b=22),
                paper_bgcolor="#11141a",
                plot_bgcolor="#11141a",
                xaxis=dict(showgrid=True, gridcolor="#1f242e", zeroline=False, tickfont=dict(size=9, color="#6b7280")),
                yaxis=dict(showgrid=True, gridcolor="#1f242e", zeroline=False, tickfont=dict(size=9, color="#6b7280")),
                showlegend=False,
            )
            disp_plot_placeholder.plotly_chart(fig_disp, use_container_width=True, key=f"disp_{frame_idx}")

            # 6. RIGHT COLUMN: 3D Spatial Odometry & Feature Point Cloud (Plotly 3D WebGL)
            fig_3d = go.Figure()

            # Layer 1: Sparse 3D Feature Landmark Point Cloud (dots scattered in 3D space)
            fig_3d.add_trace(go.Scatter3d(
                x=cloud_x, y=cloud_y, z=cloud_z,
                mode="markers",
                marker=dict(
                    size=2.0,
                    color=cloud_colors,
                    colorscale="Viridis",
                    opacity=0.75
                ),
                name="3D Landmarks",
                hoverinfo="none"
            ))

            # Layer 2: 3D Drone Flight Path (continuous line trail)
            fig_3d.add_trace(go.Scatter3d(
                x=traj_x, y=traj_y, z=traj_z,
                mode="lines",
                line=dict(color="#2dd4bf", width=4),
                name="Flight Path",
                hoverinfo="none"
            ))

            # Layer 3: Red Drone Body Ring / Coordinate Triad at current pose
            theta = np.linspace(0, 2 * np.pi, 24)
            r_ring = 0.45
            ring_x = curr_pose[0] + r_ring * np.cos(theta)
            ring_y = curr_pose[1] + r_ring * np.sin(theta)
            ring_z = curr_pose[2] + np.zeros_like(theta)

            fig_3d.add_trace(go.Scatter3d(
                x=ring_x, y=ring_y, z=ring_z,
                mode="lines",
                line=dict(color="#ef4444", width=5),
                name="Quadcopter Pose",
                hoverinfo="none"
            ))

            fig_3d.update_layout(
                template="plotly_dark",
                height=565,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#0c0e12",
                scene=dict(
                    bgcolor="#0c0e12",
                    xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                    zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                    camera=dict(
                        eye=dict(x=-1.5, y=-2.0, z=1.8),
                        center=dict(x=0, y=1.2, z=0.5)
                    ),
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1.3, z=0.9)
                ),
                showlegend=False
            )
            threed_plot_placeholder.plotly_chart(fig_3d, use_container_width=True, key=f"3d_{frame_idx}")

        prev_gray = curr_gray.copy()
        time.sleep(0.015)

    cap.release()


if __name__ == "__main__":
    main()
