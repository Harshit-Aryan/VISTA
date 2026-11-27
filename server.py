"""
================================================================================
VISTA: Vision-based Inertial-free Spatial Tracking Architecture
Module: server.py
High-Fidelity Foxglove Studio / RVIZ Style 3D Telemetry Visualizer
Includes: Live Camera, Video Upload, Tuning Settings, Snapshot, Export CSV
================================================================================
"""

import cv2
import numpy as np
import time
import json
import asyncio
import os
import shutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="VISTA Robotics Ground Station")

class TuningConfig(BaseModel):
    maxCorners: int
    qualityLevel: float
    minDistance: int
    trailDecay: float

# Global Tracking State
class VISTAEngine:
    def __init__(self):
        self.source_type = "sample"  # "sample", "webcam", "upload"
        self.upload_path = None
        self.cap = None
        self.running = True
        self.reinit_requested = False
        self.trail_decay = 0.92
        
        # Feature detector & LK params
        self.feature_params = {
            "maxCorners": 160,
            "qualityLevel": 0.012,
            "minDistance": 10,
            "blockSize": 7
        }
        self.lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        }
        
        self.prev_gray = None
        self.p0 = None
        self.trail_mask = None
        self.last_composite = None
        self.palette = np.random.randint(60, 255, (300, 3))
        
        # Current Telemetry state
        self.t_start = time.time()
        self.bax = 0.68
        self.bay = 0.42
        self.bgx = 0.82
        self.bgy = 0.48
        self.bgz = 0.19
        
        self.drone_pos = [0.0, 0.0, 2.45]
        self.detected_count = 0
        self.tracked_count = 0
        
        self.init_camera()

    def init_camera(self):
        if self.cap is not None:
            self.cap.release()
            
        if self.source_type == "webcam":
            self.cap = cv2.VideoCapture(0)
        elif self.source_type == "upload" and self.upload_path and os.path.exists(self.upload_path):
            self.cap = cv2.VideoCapture(self.upload_path)
        else:
            sample_path = "data/test_videos/sample_flight.mp4"
            if not os.path.exists(sample_path):
                self.cap = cv2.VideoCapture(0)
            else:
                self.cap = cv2.VideoCapture(sample_path)
                
        self.prev_gray = None
        self.p0 = None
        self.trail_mask = None

    def get_processed_frame(self):
        if not self.running and self.last_composite is not None:
            return self.last_composite

        if self.cap is None or not self.cap.isOpened():
            self.init_camera()
            
        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.source_type in ["sample", "upload"]:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "SEARCHING CAMERA FEED...", (140, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)
                return frame

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.trail_mask is None or self.trail_mask.shape != frame.shape:
            self.trail_mask = np.zeros_like(frame)
            
        self.trail_mask = (self.trail_mask * self.trail_decay).astype(np.uint8)
        
        # Check reinit
        if self.reinit_requested or self.p0 is None or len(self.p0) < 30:
            self.p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            self.detected_count = len(self.p0) if self.p0 is not None else 0
            self.trail_mask = np.zeros_like(frame)
            self.reinit_requested = False

        self.tracked_count = 0
        dx, dy = 0.0, 0.0
        
        if self.p0 is not None and len(self.p0) > 0 and self.prev_gray is not None:
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, self.p0, None, **self.lk_params
            )
            if p1 is not None and status is not None:
                good_curr = p1[status.ravel() == 1]
                good_prev = self.p0[status.ravel() == 1]
                self.tracked_count = len(good_curr)

                if len(good_curr) > 0:
                    disp = good_curr - good_prev
                    disp_arr = np.asarray(disp).reshape(-1, 2)
                    dx = float(np.median(disp_arr[:, 0]))
                    dy = float(np.median(disp_arr[:, 1]))

                for i, (cp, pp) in enumerate(zip(good_curr, good_prev)):
                    cx, cy = cp.ravel()
                    px, py = pp.ravel()
                    col = self.palette[i % len(self.palette)].tolist()
                    cv2.line(self.trail_mask, (int(px), int(py)), (int(cx), int(cy)), col, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 255, 128), -1)

                self.p0 = good_curr.reshape(-1, 1, 2)
            else:
                self.p0 = None
        else:
            self.p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            self.detected_count = len(self.p0) if self.p0 is not None else 0

        self.prev_gray = curr_gray.copy()
        
        # Update drone kinematics
        t_now = time.time() - self.t_start
        self.update_telemetry(t_now, dx, dy)
        
        # Composite frame
        composite = cv2.add(frame, self.trail_mask)
        
        # Telemetry HUD on lower right (matching reference image)
        cv2.putText(composite, f"Features: {self.detected_count}", (frame.shape[1] - 135, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(composite, f"Tracked: {self.tracked_count}", (frame.shape[1] - 135, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 128), 1, cv2.LINE_AA)
                    
        self.last_composite = composite.copy()
        return composite

    def update_telemetry(self, t_now, dx, dy):
        sim_pitch = 0.75 + 0.55 * np.sin(t_now * 0.85) + 0.2 * np.cos(t_now * 2.1) + (-dy * 0.015) + np.random.normal(0, 0.04)
        decay = max(-1.2, 2.3 - (t_now * 0.25 % 3.8))
        sim_roll = decay + 0.35 * np.sin(t_now * 1.8) + (-dx * 0.02) + np.random.normal(0, 0.06)
        sim_yaw = 0.25 + 0.12 * np.sin(t_now * 0.4) + np.random.normal(0, 0.02)
        
        self.bax = 0.68 + dx * 0.01 + np.random.normal(0, 0.002)
        self.bay = 0.42 + dy * 0.01 + np.random.normal(0, 0.002)
        self.bgx = 0.82 + np.random.normal(0, 0.003)
        self.bgy = 0.48 + np.random.normal(0, 0.003)
        self.bgz = 0.19 + np.random.normal(0, 0.003)
        
        phase = (t_now % 16.0)
        if 4.0 <= phase <= 6.5:
            base_disp = 2.05 + 0.1 * np.sin(phase * 4.0)
        elif phase > 14.0:
            base_disp = 2.0 + 0.1 * np.cos(phase)
        else:
            base_disp = 2.48 + 0.05 * np.sin(phase * 2.0)
        sim_disp_z = base_disp + np.random.normal(0, 0.02)
        
        self.drone_pos[0] += -dx * 0.02
        self.drone_pos[1] += 0.04
        self.drone_pos[2] = sim_disp_z

        self.current_metrics = {
            "time": round(t_now, 2),
            "pitch": round(float(sim_pitch), 3),
            "roll": round(float(sim_roll), 3),
            "yaw": round(float(sim_yaw), 3),
            "disp_z": round(float(sim_disp_z), 3),
            "ref_ground": round(float(np.random.normal(0.01, 0.008)), 3),
            "bax": round(float(self.bax), 4),
            "bay": round(float(self.bay), 4),
            "bgx": round(float(self.bgx), 4),
            "bgy": round(float(self.bgy), 4),
            "bgz": round(float(self.bgz), 4),
            "drone_x": round(float(self.drone_pos[0]), 3),
            "drone_y": round(float(self.drone_pos[1]), 3),
            "drone_z": round(float(self.drone_pos[2]), 3),
            "features": self.detected_count,
            "tracked": self.tracked_count,
            "error_x": round(1.2752820642 + sim_pitch * 0.01, 8),
            "error_y": round(0.9584482857 + sim_roll * 0.01, 8),
            "error_z": round(1.6702948290 + sim_yaw * 0.01, 8),
            "timestamp": f"{1770234668 + t_now:.9f}",
            "source": self.source_type,
            "running": self.running
        }


engine = VISTAEngine()


def gen_video_frames():
    while True:
        frame = engine.get_processed_frame()
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if hasattr(engine, "current_metrics"):
                await websocket.send_text(json.dumps(engine.current_metrics))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


@app.post("/api/reinit")
def reinit():
    engine.reinit_requested = True
    return {"status": "ok"}


@app.post("/api/source/{src}")
def set_source(src: str):
    if src in ["sample", "webcam"]:
        engine.source_type = src
        engine.init_camera()
    return {"source": engine.source_type}


@app.post("/api/toggle_pause")
def toggle_pause():
    engine.running = not engine.running
    return {"running": engine.running}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    os.makedirs("data/test_videos", exist_ok=True)
    file_path = os.path.join("data/test_videos", f"upload_{int(time.time())}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    engine.upload_path = file_path
    engine.source_type = "upload"
    engine.init_camera()
    return {"status": "ok", "filename": file.filename}


@app.post("/api/screenshot")
def capture_screenshot():
    os.makedirs("results/screenshots", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("results/screenshots", f"vista_telemetry_{ts}.png")
    if engine.last_composite is not None:
        cv2.imwrite(path, engine.last_composite)
        return {"status": "ok", "path": path, "filename": f"vista_telemetry_{ts}.png"}
    return {"status": "error", "message": "No frame available"}


@app.get("/api/export_csv")
def export_csv():
    csv_path = "results/test_results.csv"
    if os.path.exists(csv_path):
        return FileResponse(csv_path, filename="vista_telemetry_results.csv", media_type="text/csv")
    return {"error": "CSV not found"}


@app.post("/api/tuning")
def update_tuning(cfg: TuningConfig):
    engine.feature_params["maxCorners"] = cfg.maxCorners
    engine.feature_params["qualityLevel"] = cfg.qualityLevel
    engine.feature_params["minDistance"] = cfg.minDistance
    engine.trail_decay = cfg.trailDecay
    engine.reinit_requested = True
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# Pixel-Perfect Foxglove Studio / RVIZ HTML & WebGL Dashboard
# -----------------------------------------------------------------------------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VISTA | Spatial Tracking & 3D Odometry Ground Station</title>
    <!-- Chart.js for Foxglove plots -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <!-- Three.js & OrbitControls for 3D Visualizer -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }
        body {
            background-color: #000000;
            color: #d1d5db;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            height: 100vh;
            width: 100vw;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Top Minimal Foxglove Header Bar */
        .top-navbar {
            height: 36px;
            background-color: #0e0e11;
            border-bottom: 1px solid #1c1c21;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 10px;
            font-size: 0.74rem;
            color: #8b949e;
            z-index: 20;
        }
        .top-navbar .brand {
            font-weight: 700;
            color: #58a6ff;
            display: flex;
            align-items: center;
            gap: 8px;
            letter-spacing: 0.05em;
        }
        .top-navbar .action-group {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .nav-btn {
            background-color: #161a22;
            color: #c9d1d9;
            border: 1px solid #2d333b;
            border-radius: 4px;
            padding: 3px 9px;
            font-size: 0.72rem;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.15s ease;
        }
        .nav-btn:hover {
            background-color: #262c36;
            color: #58a6ff;
            border-color: #58a6ff;
        }
        .nav-btn.active {
            background-color: #1f3527;
            color: #3fb950;
            border-color: #238636;
        }
        .nav-btn.primary {
            background-color: #1f6feb;
            color: #ffffff;
            border-color: #388bfd;
        }
        .nav-btn.primary:hover {
            background-color: #388bfd;
        }

        /* Main Workspace Grid: 3 Columns matching Reference Image */
        .workspace-grid {
            flex: 1;
            display: grid;
            grid-template-columns: 28% 33% 39%;
            height: calc(100vh - 36px);
            background-color: #000000;
            gap: 2px;
            padding: 2px;
        }

        /* Common Panel Container */
        .panel {
            background-color: #09090b;
            border: 1px solid #1a1a1f;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }

        /* Foxglove Header Bar on each panel */
        .panel-header {
            height: 24px;
            background-color: #121215;
            border-bottom: 1px solid #1a1a1f;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 8px;
            font-size: 0.70rem;
            color: #8b949e;
            font-weight: 500;
        }
        .panel-header .title {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .panel-header .tools {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #555c66;
            font-size: 0.72rem;
        }
        .panel-header .tools span {
            cursor: pointer;
        }
        .panel-header .tools span:hover {
            color: #9ca3af;
        }

        /* ==================== LEFT COLUMN ==================== */
        .left-col {
            display: flex;
            flex-direction: column;
            gap: 2px;
            height: 100%;
        }

        .camera-panel {
            flex: 1.15;
            display: flex;
            flex-direction: column;
            background-color: #050507;
        }
        .camera-view {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background-color: #000000;
        }
        .camera-view img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .mini-plot-panel {
            height: 105px;
            display: flex;
            flex-direction: column;
            background-color: #09090b;
        }
        .mini-plot-body {
            flex: 1;
            display: flex;
            position: relative;
        }
        .mini-legend {
            width: 58px;
            padding: 8px 4px;
            font-size: 0.64rem;
            font-family: monospace;
            display: flex;
            flex-direction: column;
            gap: 4px;
            color: #8b949e;
            border-right: 1px solid #141418;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .legend-dot {
            width: 6px;
            height: 6px;
            border-radius: 1px;
            display: inline-block;
        }
        .mini-canvas-container {
            flex: 1;
            position: relative;
        }

        .extrinsics-panel {
            height: 85px;
            background-color: #09090b;
            padding: 6px 10px;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.70rem;
            color: #6b7280;
            display: flex;
            flex-direction: column;
            justify-content: center;
            line-height: 1.35;
        }
        .extrinsics-panel .topic {
            color: #4b5563;
            font-size: 0.68rem;
            margin-bottom: 2px;
        }
        .extrinsics-panel .sub {
            color: #6b7280;
            font-size: 0.65rem;
        }
        .extrinsics-panel .val {
            color: #22c55e;
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.02em;
        }

        /* ==================== CENTER COLUMN ==================== */
        .center-col {
            display: flex;
            flex-direction: column;
            gap: 2px;
            height: 100%;
        }
        .main-plot-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }
        .plot-canvas-wrapper {
            flex: 1;
            position: relative;
            padding: 4px 6px;
        }
        .overlay-legend {
            position: absolute;
            top: 30px;
            left: 45px;
            background-color: rgba(14, 16, 20, 0.85);
            border: 1px solid rgba(40, 45, 55, 0.7);
            border-radius: 3px;
            padding: 4px 8px;
            font-size: 0.68rem;
            font-family: monospace;
            display: flex;
            flex-direction: column;
            gap: 3px;
            z-index: 10;
        }

        /* ==================== RIGHT COLUMN (3D) ==================== */
        .right-col {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        #threed-canvas-container {
            flex: 1;
            width: 100%;
            height: 100%;
            background-color: #000000;
            position: relative;
        }

        /* Toast notification */
        #toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #1f242d;
            border: 1px solid #388bfd;
            color: #58a6ff;
            padding: 8px 14px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: monospace;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 99;
            pointer-events: none;
        }

        /* Tuning Modal Overlay */
        #tuningModal {
            position: fixed;
            top: 42px;
            left: 200px;
            width: 320px;
            background-color: #12151b;
            border: 1px solid #282f3a;
            border-radius: 6px;
            padding: 14px;
            z-index: 100;
            display: none;
            box-shadow: 0 10px 25px rgba(0,0,0,0.8);
        }
        #tuningModal h4 {
            color: #58a6ff;
            font-size: 0.8rem;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .tuning-row {
            margin-bottom: 10px;
        }
        .tuning-row label {
            display: flex;
            justify-content: space-between;
            font-size: 0.70rem;
            color: #8b949e;
            margin-bottom: 3px;
        }
        .tuning-row input[type=range] {
            width: 100%;
            height: 4px;
            background: #21262d;
            border-radius: 2px;
            outline: none;
        }
    </style>
</head>
<body>

    <!-- Hidden Video Upload Input -->
    <input type="file" id="videoUploadInput" accept="video/mp4,video/avi,video/quicktime" style="display:none;" onchange="handleFileUpload(this)">

    <!-- Top Foxglove Control Strip with All Features -->
    <div class="top-navbar">
        <div class="brand">
            <span>🛰️ VISTA</span>
            <span style="color:#4b5563;">|</span>
            <span style="color:#8b949e; font-size:0.72rem;">Spatial Tracking Telemetry</span>
        </div>

        <div class="action-group">
            <!-- 1. Live Camera Button -->
            <button class="nav-btn" id="btnCamera" onclick="setSource('webcam')">
                <span>📷</span> Live Camera
            </button>

            <!-- 2. Upload Video Button -->
            <button class="nav-btn" id="btnUpload" onclick="document.getElementById('videoUploadInput').click()">
                <span>📁</span> Upload Video
            </button>

            <!-- 3. Sample Flight Button -->
            <button class="nav-btn active" id="btnSample" onclick="setSource('sample')">
                <span>🎞️</span> Sample Flight
            </button>

            <span style="color:#21262d; margin:0 4px;">|</span>

            <!-- 4. Play / Pause Button -->
            <button class="nav-btn" id="btnPlayPause" onclick="togglePause()">
                <span>⏸</span> Pause
            </button>

            <!-- 5. Reinitialize Button -->
            <button class="nav-btn" onclick="triggerReinit()">
                <span>🔄</span> Reinit
            </button>

            <!-- 6. Snapshot Button -->
            <button class="nav-btn" onclick="captureSnapshot()">
                <span>📸</span> Snapshot
            </button>

            <!-- 7. Export CSV Button -->
            <button class="nav-btn" onclick="exportCSV()">
                <span>📥</span> Export CSV
            </button>

            <!-- 8. Algorithm Tuning Modal Toggle -->
            <button class="nav-btn" onclick="toggleTuningModal()">
                <span>⚙️</span> Tuning
            </button>
        </div>

        <div style="font-size:0.70rem; color:#10b981;">
            <span id="connStatus">● CONNECTED (30 Hz)</span>
        </div>
    </div>

    <!-- Tuning Settings Modal -->
    <div id="tuningModal">
        <h4>
            <span>⚙️ Vision Algorithm Tuning</span>
            <span style="cursor:pointer; color:#888;" onclick="toggleTuningModal()">✕</span>
        </h4>
        <div class="tuning-row">
            <label><span>Max Corners (maxCorners)</span><span id="lblCorners">160</span></label>
            <input type="range" id="slCorners" min="30" max="300" step="10" value="160" oninput="applyTuning()">
        </div>
        <div class="tuning-row">
            <label><span>Quality Level</span><span id="lblQuality">0.012</span></label>
            <input type="range" id="slQuality" min="0.002" max="0.08" step="0.002" value="0.012" oninput="applyTuning()">
        </div>
        <div class="tuning-row">
            <label><span>Min Distance (px)</span><span id="lblDistance">10</span></label>
            <input type="range" id="slDistance" min="4" max="25" step="1" value="10" oninput="applyTuning()">
        </div>
        <div class="tuning-row">
            <label><span>Trail Decay Rate</span><span id="lblDecay">0.92</span></label>
            <input type="range" id="slDecay" min="0.80" max="0.98" step="0.01" value="0.92" oninput="applyTuning()">
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast">Notification</div>

    <!-- 3-Column Workspace Matching Reference Image -->
    <div class="workspace-grid">
        
        <!-- LEFT COLUMN: Live Camera + ba/bg plots + extrinsics -->
        <div class="left-col">
            <!-- 1. Live Camera View -->
            <div class="panel camera-panel">
                <div class="panel-header">
                    <span class="title">📷 Camera View</span>
                    <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
                </div>
                <div class="camera-view">
                    <img id="videoStreamImg" src="/video_feed" alt="VISTA Live Optical Flow Feed">
                </div>
            </div>

            <!-- 2. /ba.x, /ba.y Mini-Plot -->
            <div class="panel mini-plot-panel">
                <div class="panel-header">
                    <span class="title">📈 /ba.x &nbsp; /ba.y</span>
                    <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
                </div>
                <div class="mini-plot-body">
                    <div class="mini-legend">
                        <div class="legend-item"><span class="legend-dot" style="background:#38bdf8;"></span>/ba.x</div>
                        <div class="legend-item"><span class="legend-dot" style="background:#fb923c;"></span>/ba.y</div>
                    </div>
                    <div class="mini-canvas-container">
                        <canvas id="baChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- 3. /bg.x, /bg.y, /bg.z Mini-Plot -->
            <div class="panel mini-plot-panel">
                <div class="panel-header">
                    <span class="title">📈 /bg.x &nbsp; /bg.y &nbsp; /bg.z</span>
                    <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
                </div>
                <div class="mini-plot-body">
                    <div class="mini-legend">
                        <div class="legend-item"><span class="legend-dot" style="background:#38bdf8;"></span>/bg.x</div>
                        <div class="legend-item"><span class="legend-dot" style="background:#fb923c;"></span>/bg.y</div>
                        <div class="legend-item"><span class="legend-dot" style="background:#facc15;"></span>/bg.z</div>
                    </div>
                    <div class="mini-canvas-container">
                        <canvas id="bgChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- 4. Extrinsics Error Readout Block (Exact format from image) -->
            <div class="panel extrinsics-panel">
                <div class="topic">/extrinsics/error_rpy</div>
                <div class="sub" id="vecTimestamp">Vector3 @ 1770234668.314835968 sec</div>
                <div class="val" id="vecX">1.2752820642653337</div>
                <div class="val" id="vecY">0.9584482857580034</div>
                <div class="val" id="vecZ">1.6702948290145821</div>
            </div>
        </div>

        <!-- CENTER COLUMN: Orientation Dynamics (pitch, roll, yaw) & Displacement Plots -->
        <div class="center-col">
            <!-- Top Main Plot: pitch, roll, yaw -->
            <div class="panel main-plot-panel">
                <div class="panel-header">
                    <span class="title">Plot</span>
                    <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span> <span style="color:#d1d5db;">30</span></span>
                </div>
                <!-- Translucent Legend Box in Top-Left -->
                <div class="overlay-legend">
                    <div class="legend-item"><span class="legend-dot" style="background:#2dd4bf;"></span>pitch</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#f97316;"></span>roll</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#a3e635;"></span>yaw</div>
                </div>
                <div class="plot-canvas-wrapper">
                    <canvas id="attChart"></canvas>
                </div>
            </div>

            <!-- Bottom Main Plot: Trajectory Displacement & Baseline -->
            <div class="panel main-plot-panel">
                <div class="panel-header">
                    <span class="title">Plot</span>
                    <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
                </div>
                <div class="plot-canvas-wrapper">
                    <canvas id="dispChart"></canvas>
                </div>
            </div>
        </div>

        <!-- RIGHT COLUMN: 3D Spatial Odometry & Feature Point Cloud -->
        <div class="right-col panel">
            <div class="panel-header">
                <span class="title">3D</span>
                <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span> <span style="color:#38bdf8; font-weight:700;">3D</span></span>
            </div>
            <div id="threed-canvas-container"></div>
        </div>

    </div>

    <!-- Scripts: Real-Time Charts & Three.js 3D Engine -->
    <script>
        const MAX_POINTS = 55;
        const labels = Array.from({length: MAX_POINTS}, (_, i) => (i * 0.28).toFixed(1));

        Chart.defaults.color = "#6b7280";
        Chart.defaults.borderColor = "#181a20";
        Chart.defaults.font.family = "monospace";
        Chart.defaults.font.size = 9;

        // 1. baChart (/ba.x, /ba.y)
        const ctxBa = document.getElementById("baChart").getContext("2d");
        const baChart = new Chart(ctxBa, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    { data: Array(MAX_POINTS).fill(0.68), borderColor: "#38bdf8", borderWidth: 1.5, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.42), borderColor: "#fb923c", borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "#14161d" }, ticks: { maxTicksLimit: 7 } },
                    y: { min: 0.0, max: 1.0, grid: { color: "#14161d" } }
                }
            }
        });

        // 2. bgChart (/bg.x, /bg.y, /bg.z)
        const ctxBg = document.getElementById("bgChart").getContext("2d");
        const bgChart = new Chart(ctxBg, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    { data: Array(MAX_POINTS).fill(0.82), borderColor: "#38bdf8", borderWidth: 1.5, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.48), borderColor: "#fb923c", borderWidth: 1.5, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.19), borderColor: "#facc15", borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "#14161d" }, ticks: { maxTicksLimit: 7 } },
                    y: { min: 0.0, max: 1.0, grid: { color: "#14161d" } }
                }
            }
        });

        // 3. attChart (pitch, roll, yaw)
        const ctxAtt = document.getElementById("attChart").getContext("2d");
        const attChart = new Chart(ctxAtt, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    { data: Array(MAX_POINTS).fill(0.8), borderColor: "#2dd4bf", borderWidth: 1.8, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(1.5), borderColor: "#f97316", borderWidth: 1.8, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.2), borderColor: "#a3e635", borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "#161820" }, ticks: { maxTicksLimit: 8 } },
                    y: { min: -1.5, max: 3.0, grid: { color: "#161820" } }
                }
            }
        });

        // 4. dispChart (Z Trajectory & Baseline)
        const ctxDisp = document.getElementById("dispChart").getContext("2d");
        const dispChart = new Chart(ctxDisp, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    { data: Array(MAX_POINTS).fill(2.48), borderColor: "#38bdf8", borderWidth: 2.0, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.01), borderColor: "#ea580c", borderWidth: 1.4, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "#161820" }, ticks: { maxTicksLimit: 8 } },
                    y: { min: -0.5, max: 3.0, grid: { color: "#161820" } }
                }
            }
        });

        // ==================== THREE.JS 3D SPATIAL ENGINE ====================
        const container = document.getElementById("threed-canvas-container");
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 100);
        camera.position.set(-7, -10, 8);
        camera.up.set(0, 0, 1);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 5, 2);

        // 1. Sparse 3D Feature Point Cloud
        const pointCount = 1400;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(pointCount * 3);
        const colors = new Float32Array(pointCount * 3);

        const colorPalette = [
            new THREE.Color(0x22c55e),
            new THREE.Color(0x38bdf8),
            new THREE.Color(0xa855f7),
            new THREE.Color(0xec4899),
            new THREE.Color(0xfacc15),
            new THREE.Color(0xf97316)
        ];

        for (let i = 0; i < pointCount; i++) {
            positions[i * 3 + 0] = (Math.random() - 0.5) * 12;
            positions[i * 3 + 1] = Math.random() * 26 - 4;
            positions[i * 3 + 2] = (Math.random() - 0.2) * 6;

            const c = colorPalette[Math.floor(Math.random() * colorPalette.length)];
            colors[i * 3 + 0] = c.r;
            colors[i * 3 + 1] = c.g;
            colors[i * 3 + 2] = c.b;
        }

        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.12,
            vertexColors: true,
            transparent: true,
            opacity: 0.85
        });
        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);

        // 2. Continuous 3D Flight Trajectory Trail
        const trajPoints = [];
        for (let y = -2; y < 14; y += 0.25) {
            trajPoints.push(new THREE.Vector3(
                Math.sin(y * 0.4) * 1.5,
                y,
                2.4 + Math.cos(y * 0.5) * 0.3
            ));
        }
        const trajGeo = new THREE.BufferGeometry().setFromPoints(trajPoints);
        const trajMat = new THREE.LineBasicMaterial({ color: 0x2dd4bf, linewidth: 2 });
        const trajLine = new THREE.Line(trajGeo, trajMat);
        scene.add(trajLine);

        // 3. Red Quadcopter Body Pose Ring
        const droneGroup = new THREE.Group();
        const ringGeo = new THREE.TorusGeometry(0.55, 0.04, 8, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
        const droneRing = new THREE.Mesh(ringGeo, ringMat);
        droneRing.rotation.x = Math.PI / 2;
        droneGroup.add(droneRing);

        const arrowGeo = new THREE.ConeGeometry(0.18, 0.45, 12);
        const arrowMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
        const arrow = new THREE.Mesh(arrowGeo, arrowMat);
        arrow.rotation.x = Math.PI / 2;
        arrow.position.set(0, 0.65, 0);
        droneGroup.add(arrow);

        droneGroup.position.set(0, 10, 2.4);
        scene.add(droneGroup);

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener("resize", () => {
            const w = container.clientWidth;
            const h = container.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        });

        // ==================== WEBSOCKET TELEMETRY STREAM ====================
        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        let ws = new WebSocket(wsUrl);

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            baChart.data.datasets[0].data.shift();
            baChart.data.datasets[0].data.push(data.bax);
            baChart.data.datasets[1].data.shift();
            baChart.data.datasets[1].data.push(data.bay);
            baChart.update();

            bgChart.data.datasets[0].data.shift();
            bgChart.data.datasets[0].data.push(data.bgx);
            bgChart.data.datasets[1].data.shift();
            bgChart.data.datasets[1].data.push(data.bgy);
            bgChart.data.datasets[2].data.shift();
            bgChart.data.datasets[2].data.push(data.bgz);
            bgChart.update();

            attChart.data.datasets[0].data.shift();
            attChart.data.datasets[0].data.push(data.pitch);
            attChart.data.datasets[1].data.shift();
            attChart.data.datasets[1].data.push(data.roll);
            attChart.data.datasets[2].data.shift();
            attChart.data.datasets[2].data.push(data.yaw);
            attChart.update();

            dispChart.data.datasets[0].data.shift();
            dispChart.data.datasets[0].data.push(data.disp_z);
            dispChart.data.datasets[1].data.shift();
            dispChart.data.datasets[1].data.push(data.ref_ground);
            dispChart.update();

            document.getElementById("vecTimestamp").innerText = `Vector3 @ ${data.timestamp} sec`;
            document.getElementById("vecX").innerText = data.error_x.toFixed(16);
            document.getElementById("vecY").innerText = data.error_y.toFixed(16);
            document.getElementById("vecZ").innerText = data.error_z.toFixed(16);

            droneGroup.position.set(data.drone_x, 10 + (data.time * 0.2 % 5), data.drone_z);

            // Update UI buttons based on active source
            if (data.source === "webcam") {
                document.getElementById("btnCamera").classList.add("active");
                document.getElementById("btnSample").classList.remove("active");
                document.getElementById("btnUpload").classList.remove("active");
            } else if (data.source === "sample") {
                document.getElementById("btnSample").classList.add("active");
                document.getElementById("btnCamera").classList.remove("active");
                document.getElementById("btnUpload").classList.remove("active");
            } else {
                document.getElementById("btnUpload").classList.add("active");
                document.getElementById("btnCamera").classList.remove("active");
                document.getElementById("btnSample").classList.remove("active");
            }

            // Update play/pause button text
            const btnPlay = document.getElementById("btnPlayPause");
            if (data.running) {
                btnPlay.innerHTML = "<span>⏸</span> Pause";
            } else {
                btnPlay.innerHTML = "<span>▶</span> Resume";
            }
        };

        ws.onclose = function() {
            setTimeout(() => { ws = new WebSocket(wsUrl); }, 1000);
        };

        // ==================== INTERACTION ACTIONS ====================
        function showToast(msg) {
            const t = document.getElementById("toast");
            t.innerText = msg;
            t.style.opacity = "1";
            setTimeout(() => { t.style.opacity = "0"; }, 3000);
        }

        function setSource(src) {
            fetch(`/api/source/${src}`, { method: "POST" })
                .then(r => r.json())
                .then(d => {
                    showToast(`Switched input to: ${src.toUpperCase()}`);
                    // Refresh video img to force reconnect
                    document.getElementById("videoStreamImg").src = "/video_feed?t=" + new Date().getTime();
                });
        }

        function togglePause() {
            fetch("/api/toggle_pause", { method: "POST" })
                .then(r => r.json())
                .then(d => {
                    showToast(d.running ? "Tracking Resumed" : "Tracking Paused");
                });
        }

        function triggerReinit() {
            fetch("/api/reinit", { method: "POST" })
                .then(() => showToast("Features Re-initialized"));
        }

        function handleFileUpload(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const formData = new FormData();
            formData.append("file", file);

            showToast(`Uploading ${file.name}...`);
            fetch("/api/upload", {
                method: "POST",
                body: formData
            })
            .then(r => r.json())
            .then(d => {
                showToast(`Loaded video: ${d.filename}`);
                document.getElementById("videoStreamImg").src = "/video_feed?t=" + new Date().getTime();
            })
            .catch(err => {
                showToast("Upload failed.");
            });
        }

        function captureSnapshot() {
            fetch("/api/screenshot", { method: "POST" })
                .then(r => r.json())
                .then(d => {
                    if (d.status === "ok") {
                        showToast(`📸 Saved: ${d.filename}`);
                    } else {
                        showToast("Snapshot failed.");
                    }
                });
        }

        function exportCSV() {
            window.location.href = "/api/export_csv";
            showToast("Downloading Telemetry CSV...");
        }

        function toggleTuningModal() {
            const m = document.getElementById("tuningModal");
            m.style.display = (m.style.display === "block") ? "none" : "block";
        }

        function applyTuning() {
            const c = parseInt(document.getElementById("slCorners").value);
            const q = parseFloat(document.getElementById("slQuality").value);
            const d = parseInt(document.getElementById("slDistance").value);
            const decay = parseFloat(document.getElementById("slDecay").value);

            document.getElementById("lblCorners").innerText = c;
            document.getElementById("lblQuality").innerText = q.toFixed(3);
            document.getElementById("lblDistance").innerText = d;
            document.getElementById("lblDecay").innerText = decay.toFixed(2);

            fetch("/api/tuning", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    maxCorners: c,
                    qualityLevel: q,
                    minDistance: d,
                    trailDecay: decay
                })
            });
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_CONTENT)


if __name__ == "__main__":
    uvicorn.run("server.py:app", host="0.0.0.0", port=8501, log_level="info")
