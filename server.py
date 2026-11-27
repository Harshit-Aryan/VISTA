"""
================================================================================
VISTA: Vision-based Inertial-free Spatial Tracking Architecture
Module: server.py
Foxglove Studio / RVIZ Style 3D Telemetry Visualizer powered by Real Monocular VO
================================================================================
"""

import cv2
import numpy as np
import time
import json
import asyncio
import os
import shutil
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from visual_odometry import VisualOdometry, PinholeCamera

app = FastAPI(title="VISTA Robotics Ground Station")

class TuningConfig(BaseModel):
    maxCorners: int
    qualityLevel: float
    minDistance: int
    trailDecay: float

# Global Tracking Engine using True Visual Odometry
class VISTAEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.source_type = "sample"  # "sample", "webcam", "upload"
        self.upload_path = None
        self.cap = None
        self.running = True
        self.reinit_requested = False
        
        # Instantiate true monocular visual odometry pipeline
        self.camera_model = PinholeCamera(width=640, height=480, fx=525.0, fy=525.0, cx=320.0, cy=240.0)
        self.vo = VisualOdometry(self.camera_model)
        
        self.last_composite = None
        self.current_metrics = {}
        
        self.init_camera()

    def init_camera(self):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                
            if self.source_type == "webcam":
                # Use DirectShow on Windows for fast, non-blocking camera init
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)
            elif self.source_type == "upload" and self.upload_path and os.path.exists(self.upload_path):
                self.cap = cv2.VideoCapture(self.upload_path)
            else:
                sample_path = "data/test_videos/sample_flight.mp4"
                if not os.path.exists(sample_path):
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(0)
                else:
                    self.cap = cv2.VideoCapture(sample_path)
                    
            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            self.vo.reset()

    def get_processed_frame(self):
        if not self.running and self.last_composite is not None:
            return self.last_composite

        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "INITIALIZING CAMERA...", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)
                return frame
                
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None:
                if self.source_type in ["sample", "upload"]:
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    except Exception:
                        ret, frame = False, None
                if not ret or frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "NO SIGNAL / RECONNECTING...", (130, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)
                    return frame

            if self.reinit_requested:
                self.vo.reset()
                self.reinit_requested = False

            # Run through real Visual Odometry pipeline with fallback protection
            try:
                vis_frame, telemetry = self.vo.process_frame(frame)
            except Exception as ex:
                vis_frame = frame.copy()
                telemetry = self.vo.get_current_telemetry()
                telemetry["status"] = "RECOVERING"

            telemetry["source"] = self.source_type
            telemetry["running"] = self.running
            self.current_metrics = telemetry
            
            self.last_composite = vis_frame.copy()
            return vis_frame


engine = VISTAEngine()


def gen_video_frames():
    while True:
        try:
            frame = engine.get_processed_frame()
            if frame is None or frame.size == 0:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "FRAME BUFFER EMPTY", (160, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.033)
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        except Exception:
            time.sleep(0.05)
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
            if hasattr(engine, "current_metrics") and engine.current_metrics:
                await websocket.send_text(json.dumps(engine.current_metrics))
            await asyncio.sleep(0.04)  # 25 Hz telemetry stream
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
    engine.vo.max_features = cfg.maxCorners
    engine.vo.quality_level = cfg.qualityLevel
    engine.vo.min_distance = cfg.minDistance
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
            gap: 6px;
            color: #555c66;
            font-size: 0.70rem;
        }
        .panel-header .tools span {
            cursor: pointer;
            padding: 1px 4px;
            border-radius: 2px;
        }
        .panel-header .tools span:hover {
            color: #38bdf8;
            background-color: #1a1e27;
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
            height: 100px;
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

        /* Compact Navigation Telemetry Panel (Exact requirements) */
        .extrinsics-panel {
            height: 110px;
            background-color: #09090b;
            padding: 6px 10px;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.68rem;
            color: #8b949e;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            line-height: 1.35;
        }
        .telemetry-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .telemetry-label {
            color: #6b7280;
            text-transform: uppercase;
            font-size: 0.62rem;
            letter-spacing: 0.05em;
        }
        .telemetry-val {
            color: #38bdf8;
            font-weight: 600;
        }
        .status-badge {
            padding: 1px 6px;
            border-radius: 2px;
            font-weight: 700;
            font-size: 0.62rem;
        }
        .badge-green { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid #22c55e; }
        .badge-yellow { background: rgba(234, 179, 8, 0.2); color: #eab308; border: 1px solid #eab308; }
        .badge-orange { background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid #f97316; }
        .badge-red { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
        .badge-blue { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }

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
        .threed-controls-bar {
            position: absolute;
            top: 28px;
            right: 8px;
            display: flex;
            gap: 4px;
            z-index: 10;
        }
        .threed-btn {
            background-color: rgba(22, 26, 34, 0.85);
            color: #9ca3af;
            border: 1px solid #2d333b;
            border-radius: 3px;
            padding: 2px 6px;
            font-size: 0.65rem;
            font-family: monospace;
            cursor: pointer;
        }
        .threed-btn:hover {
            color: #38bdf8;
            border-color: #38bdf8;
        }
        .threed-btn.active {
            color: #22c55e;
            border-color: #22c55e;
        }

        /* Sleek Grid Coordinates HUD Bar at Bottom of 3D Viewport */
        .grid-coords-hud {
            position: absolute;
            bottom: 6px;
            left: 8px;
            right: 8px;
            background-color: rgba(9, 11, 15, 0.88);
            border: 1px solid rgba(38, 46, 60, 0.75);
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 0.65rem;
            font-family: "Courier New", Courier, monospace;
            color: #8b949e;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
            backdrop-filter: blur(4px);
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
        .grid-coords-hud .hud-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .grid-coords-hud .hud-label {
            color: #6b7280;
            font-size: 0.60rem;
            font-weight: 600;
        }
        .grid-coords-hud .hud-val {
            color: #38bdf8;
            font-weight: 600;
        }
        .grid-coords-hud .hud-val.highlight {
            color: #22c55e;
            font-weight: 700;
        }
        .grid-coords-hud .hud-val.nearby {
            color: #facc15;
            font-weight: 700;
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

        /* Tuning Modal */
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
            <span style="color:#8b949e; font-size:0.72rem;">Monocular Visual Odometry Ground Station</span>
        </div>

        <div class="action-group">
            <button class="nav-btn" id="btnCamera" onclick="setSource('webcam')">
                <span>📷</span> Live Camera
            </button>
            <button class="nav-btn" id="btnUpload" onclick="document.getElementById('videoUploadInput').click()">
                <span>📁</span> Upload Video
            </button>
            <button class="nav-btn active" id="btnSample" onclick="setSource('sample')">
                <span>🎞️</span> Sample Flight
            </button>
            <span style="color:#21262d; margin:0 4px;">|</span>
            <button class="nav-btn" id="btnPlayPause" onclick="togglePause()">
                <span>⏸</span> Pause
            </button>
            <button class="nav-btn" onclick="triggerReinit()">
                <span>🔄</span> Reinit VO
            </button>
            <button class="nav-btn" onclick="captureSnapshot()">
                <span>📸</span> Snapshot
            </button>
            <button class="nav-btn" onclick="exportCSV()">
                <span>📥</span> Export CSV
            </button>
            <button class="nav-btn" onclick="toggleTuningModal()">
                <span>⚙️</span> Tuning
            </button>
        </div>

        <div style="font-size:0.70rem; color:#10b981;">
            <span id="connStatus">● MONOCULAR VO ENGINE ACTIVE</span>
        </div>
    </div>

    <!-- Tuning Settings Modal -->
    <div id="tuningModal">
        <h4>
            <span>⚙️ Vision Algorithm Tuning</span>
            <span style="cursor:pointer; color:#888;" onclick="toggleTuningModal()">✕</span>
        </h4>
        <div class="tuning-row">
            <label><span>Max Features (Shi-Tomasi)</span><span id="lblCorners">180</span></label>
            <input type="range" id="slCorners" min="40" max="350" step="10" value="180" oninput="applyTuning()">
        </div>
        <div class="tuning-row">
            <label><span>Corner Quality Level</span><span id="lblQuality">0.015</span></label>
            <input type="range" id="slQuality" min="0.002" max="0.08" step="0.002" value="0.015" oninput="applyTuning()">
        </div>
        <div class="tuning-row">
            <label><span>Min Distance (px)</span><span id="lblDistance">12</span></label>
            <input type="range" id="slDistance" min="4" max="25" step="1" value="12" oninput="applyTuning()">
        </div>
        <div class="tuning-row">
            <label><span>Decay Rate</span><span id="lblDecay">0.92</span></label>
            <input type="range" id="slDecay" min="0.80" max="0.98" step="0.01" value="0.92" oninput="applyTuning()">
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast">Notification</div>

    <!-- 3-Column Workspace Matching Reference Image -->
    <div class="workspace-grid">
        
        <!-- LEFT COLUMN: Live Camera + ba/bg plots + True Extrinsics Telemetry -->
        <div class="left-col">
            <!-- 1. Live Camera View -->
            <div class="panel camera-panel">
                <div class="panel-header">
                    <span class="title">📷 Camera View</span>
                    <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
                </div>
                <div class="camera-view">
                    <img id="videoStreamImg" src="/video_feed" alt="VISTA Live Optical Flow Feed" onerror="handleStreamError(this)">
                </div>
            </div>

            <!-- 2. /ba.x, /ba.y Mini-Plot (Linear Rates) -->
            <div class="panel mini-plot-panel">
                <div class="panel-header">
                    <span class="title">📈 /ba.x &nbsp; /ba.y &nbsp; (Linear Rate)</span>
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

            <!-- 3. /bg.x, /bg.y, /bg.z Mini-Plot (Angular Rates) -->
            <div class="panel mini-plot-panel">
                <div class="panel-header">
                    <span class="title">📈 /bg.x &nbsp; /bg.y &nbsp; /bg.z &nbsp; (Angular Rates)</span>
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

            <!-- 4. Real Navigation & Telemetry Readout (Mathematically Computed) -->
            <div class="panel extrinsics-panel">
                <div class="telemetry-row">
                    <span class="telemetry-label">Camera Pose [Rel Scale]</span>
                    <span class="telemetry-val" id="dispPose">X: 0.000 | Y: 0.000 | Z: 0.000</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Orientation (deg)</span>
                    <span style="color:#d1d5db;" id="dispAngles">R: 0.00° | P: 0.00° | Y: 0.00°</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Visual Landmarks</span>
                    <span style="color:#2dd4bf;" id="dispFeatures">Det: 0 | Trk: 0 | Inliers: 0</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">VO Health Status</span>
                    <span id="dispStatus" class="status-badge badge-green">STABLE</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">GPS Satellites</span>
                    <span style="color:#ef4444; font-weight:700;">UNAVAILABLE (GPS-Denied)</span>
                </div>
            </div>
        </div>

        <!-- CENTER COLUMN: Real Orientation Dynamics & Real Trajectory Plots -->
        <div class="center-col">
            <!-- Top Main Plot: Roll, Pitch, Yaw from True Rotation Matrix -->
            <div class="panel main-plot-panel">
                <div class="panel-header">
                    <span class="title">Plot | Orientation Dynamics (Roll, Pitch, Yaw)</span>
                    <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span> <span style="color:#d1d5db;">30</span></span>
                </div>
                <div class="overlay-legend">
                    <div class="legend-item"><span class="legend-dot" style="background:#2dd4bf;"></span>pitch (deg)</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#f97316;"></span>roll (deg)</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#a3e635;"></span>yaw (deg)</div>
                </div>
                <div class="plot-canvas-wrapper">
                    <canvas id="attChart"></canvas>
                </div>
            </div>

            <!-- Bottom Main Plot: Real Estimated Trajectory Coordinates -->
            <div class="panel main-plot-panel">
                <div class="panel-header">
                    <span class="title">Plot | Estimated Camera Trajectory Displacement</span>
                    <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span></span>
                </div>
                <div class="overlay-legend" style="top:25px;">
                    <div class="legend-item"><span class="legend-dot" style="background:#38bdf8;"></span>Y (forward)</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#ea580c;"></span>Z (altitude)</div>
                </div>
                <div class="plot-canvas-wrapper">
                    <canvas id="dispChart"></canvas>
                </div>
            </div>
        </div>

        <!-- RIGHT COLUMN: Upgraded 3D Spatial Odometry & Reconstructed Point Cloud -->
        <div class="right-col panel">
            <div class="panel-header">
                <span class="title">3D Spatial Odometry (SE3 Frame)</span>
                <span class="tools"><span>⊞</span><span>⏵</span><span>⊕</span><span>⚙</span> <span style="color:#38bdf8; font-weight:700;">3D</span></span>
            </div>

            <!-- Interactive 3D Viewport Toolbar -->
            <div class="threed-controls-bar">
                <button class="threed-btn" onclick="setCamView('reset')">Reset</button>
                <button class="threed-btn" onclick="setCamView('top')">Top</button>
                <button class="threed-btn" onclick="setCamView('side')">Side</button>
                <button class="threed-btn" onclick="setCamView('front')">Front</button>
                <button class="threed-btn active" id="btnFollow" onclick="toggleFollow()">Follow</button>
                <button class="threed-btn active" id="btnTraj" onclick="toggleTraj()">Traj</button>
                <button class="threed-btn active" id="btnGrid" onclick="toggleGrid()">Grid</button>
                <button class="threed-btn active" id="btnPoints" onclick="togglePoints()">Landmarks</button>
                <button class="threed-btn" id="btnAutoRot" onclick="toggleAutoRotate()">Auto Rot</button>
                <button class="threed-btn" onclick="triggerReinit()">Clear Traj</button>
            </div>

            <div id="threed-canvas-container">
                <!-- Bottom Grid Coordinates HUD Bar -->
                <div class="grid-coords-hud">
                    <div class="hud-item"><span class="hud-label">GRID EXTENTS:</span> <span class="hud-val">X: [-15, +15]m &nbsp; Y: [-15, +15]m</span></div>
                    <div class="hud-item"><span class="hud-label">ORIGIN:</span> <span class="hud-val">(0.00, 0.00, 0.00)</span></div>
                    <div class="hud-item"><span class="hud-label">STEP:</span> <span class="hud-val">1.0m</span></div>
                    <div class="hud-item"><span class="hud-label">NEARBY PTS (&le;2.5m):</span> <span class="hud-val nearby" id="hudNearbyCount">0 pts</span></div>
                    <div class="hud-item"><span class="hud-label">DRONE COORD:</span> <span class="hud-val highlight" id="hudDronePos">X: 0.000 | Y: 0.000 | Z: 0.000</span></div>
                </div>
            </div>
        </div>

    </div>

    <!-- Scripts: Real-Time Charts & Three.js 3D Engine -->
    <script>
        const MAX_POINTS = 60;
        const labels = Array.from({length: MAX_POINTS}, (_, i) => i);

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
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#38bdf8", borderWidth: 1.5, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#fb923c", borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "#14161d" }, ticks: { display: false } },
                    y: { min: -1.0, max: 1.0, grid: { color: "#14161d" } }
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
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#38bdf8", borderWidth: 1.5, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#fb923c", borderWidth: 1.5, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#facc15", borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "#14161d" }, ticks: { display: false } },
                    y: { min: -1.0, max: 1.0, grid: { color: "#14161d" } }
                }
            }
        });

        // 3. attChart (pitch, roll, yaw from true rotation matrix)
        const ctxAtt = document.getElementById("attChart").getContext("2d");
        const attChart = new Chart(ctxAtt, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#2dd4bf", borderWidth: 1.8, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#f97316", borderWidth: 1.8, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#a3e635", borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "#161820" }, ticks: { maxTicksLimit: 8 } },
                    y: { min: -25.0, max: 25.0, grid: { color: "#161820" } }
                }
            }
        });

        // 4. dispChart (Real Estimated Trajectory Components)
        const ctxDisp = document.getElementById("dispChart").getContext("2d");
        const dispChart = new Chart(ctxDisp, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#38bdf8", borderWidth: 2.0, pointRadius: 0 },
                    { data: Array(MAX_POINTS).fill(0.0), borderColor: "#ea580c", borderWidth: 1.4, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "#161820" }, ticks: { maxTicksLimit: 8 } },
                    y: { grid: { color: "#161820" } }
                }
            }
        });

        // ==================== THREE.JS 3D SPATIAL ENGINE ====================
        const container = document.getElementById("threed-canvas-container");
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.05, 200);
        camera.position.set(-5, -6, 4.5);
        camera.up.set(0, 0, 1);

        const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 2, 0);

        // 1. Fixed Reference Coordinate Grid (Z = 0)
        const gridHelper = new THREE.GridHelper(30, 30, 0x38bdf8, 0x171923);
        gridHelper.rotation.x = Math.PI / 2;
        scene.add(gridHelper);

        // 2. Fixed World Coordinate Axes at (0, 0, 0)
        const axesHelper = new THREE.AxesHelper(1.5);
        scene.add(axesHelper);

        // 3. Green Starting Origin Sphere
        const originGeo = new THREE.SphereGeometry(0.12, 16, 16);
        const originMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
        const originMarker = new THREE.Mesh(originGeo, originMat);
        originMarker.position.set(0, 0, 0);
        scene.add(originMarker);

        // 3b. 3D Spatial Grid Coordinate Labels along ground plane (Z = 0)
        function createGridLabelSprite(text, color = "#64748b", fontSize = 26) {
            const canvas = document.createElement("canvas");
            canvas.width = 128;
            canvas.height = 64;
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = color;
            ctx.font = `bold ${fontSize}px monospace`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(text, 64, 32);
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(1.4, 0.7, 1.0);
            return sprite;
        }

        const gridLabelsGroup = new THREE.Group();
        [-10, -5, 5, 10].forEach(val => {
            const lblX = createGridLabelSprite(`${val > 0 ? "+" : ""}${val}m`, "#475569", 22);
            lblX.position.set(val, -0.4, 0.02);
            gridLabelsGroup.add(lblX);

            const lblY = createGridLabelSprite(`${val > 0 ? "+" : ""}${val}m`, "#475569", 22);
            lblY.position.set(-0.6, val, 0.02);
            gridLabelsGroup.add(lblY);
        });

        const originLabel = createGridLabelSprite("(0,0) ORIGIN", "#22c55e", 24);
        originLabel.position.set(0, -0.7, 0.02);
        gridLabelsGroup.add(originLabel);

        const xEast = createGridLabelSprite("+X (East)", "#38bdf8", 24);
        xEast.position.set(15.2, 0, 0.02);
        gridLabelsGroup.add(xEast);

        const yNorth = createGridLabelSprite("+Y (North)", "#22c55e", 24);
        yNorth.position.set(0, 15.2, 0.02);
        gridLabelsGroup.add(yNorth);
        scene.add(gridLabelsGroup);

        // 4. Complete Continuous Estimated Trajectory Path (Supports Full Flight History)
        const MAX_TRAJ_PTS = 10000;
        const trajPositions = new Float32Array(MAX_TRAJ_PTS * 3);
        const trajColors = new Float32Array(MAX_TRAJ_PTS * 3);
        const trajGeo = new THREE.BufferGeometry();
        trajGeo.setAttribute("position", new THREE.BufferAttribute(trajPositions, 3));
        trajGeo.setAttribute("color", new THREE.BufferAttribute(trajColors, 3));
        trajGeo.setDrawRange(0, 1);
        const trajMat = new THREE.LineBasicMaterial({
            vertexColors: true,
            linewidth: 3
        });
        const trajectoryLine = new THREE.Line(trajGeo, trajMat);
        scene.add(trajectoryLine);

        // 5. Environmental 3D Landmarks Point Cloud (Distant / Background)
        const MAX_LANDMARKS = 2000;
        const landmarkPositions = new Float32Array(MAX_LANDMARKS * 3);
        const landmarkColors = new Float32Array(MAX_LANDMARKS * 3);
        const landmarkGeo = new THREE.BufferGeometry();
        landmarkGeo.setAttribute("position", new THREE.BufferAttribute(landmarkPositions, 3));
        landmarkGeo.setAttribute("color", new THREE.BufferAttribute(landmarkColors, 3));
        landmarkGeo.setDrawRange(0, 0);

        const landmarkMat = new THREE.PointsMaterial({
            size: 0.08,
            vertexColors: true,
            transparent: true,
            opacity: 0.65
        });
        const landmarkPoints = new THREE.Points(landmarkGeo, landmarkMat);
        scene.add(landmarkPoints);

        // 5b. High-Visibility Highlighted Nearby Points (< 2.5m from Drone)
        const MAX_NEARBY = 600;
        const nearbyPositions = new Float32Array(MAX_NEARBY * 3);
        const nearbyGeo = new THREE.BufferGeometry();
        nearbyGeo.setAttribute("position", new THREE.BufferAttribute(nearbyPositions, 3));
        nearbyGeo.setDrawRange(0, 0);

        const nearbyMat = new THREE.PointsMaterial({
            size: 0.26,
            color: 0xfacc15, // Bright Neon Gold / Yellow
            transparent: true,
            opacity: 0.95
        });
        const nearbyPoints = new THREE.Points(nearbyGeo, nearbyMat);
        scene.add(nearbyPoints);

        // 6. Camera / Drone Marker at Current Estimated Pose
        const droneGroup = new THREE.Group();

        // Red Center Fuselage Sphere
        const redSphereGeo = new THREE.SphereGeometry(0.15, 16, 16);
        const redSphereMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
        const centerSphere = new THREE.Mesh(redSphereGeo, redSphereMat);
        droneGroup.add(centerSphere);

        // Red Quadcopter Body Ring
        const ringGeo = new THREE.TorusGeometry(0.40, 0.03, 8, 24);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
        const droneRing = new THREE.Mesh(ringGeo, ringMat);
        droneRing.rotation.x = Math.PI / 2;
        droneGroup.add(droneRing);

        // Active Proximity Perception Range Ring (2.5m radius highlighting nearby space)
        const proxRingGeo = new THREE.RingGeometry(2.44, 2.52, 48);
        const proxRingMat = new THREE.MeshBasicMaterial({
            color: 0xfacc15,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.35
        });
        const proxRing = new THREE.Mesh(proxRingGeo, proxRingMat);
        proxRing.rotation.x = Math.PI / 2;
        droneGroup.add(proxRing);

        // Heading Direction Arrow (pointing forward along +Y)
        const arrowGeo = new THREE.ConeGeometry(0.12, 0.35, 12);
        const arrowMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
        const headingArrow = new THREE.Mesh(arrowGeo, arrowMat);
        headingArrow.rotation.x = Math.PI / 2;
        headingArrow.position.set(0, 0.45, 0);
        droneGroup.add(headingArrow);

        // Triad axes at drone body
        const droneAxes = new THREE.AxesHelper(0.6);
        droneGroup.add(droneAxes);

        droneGroup.position.set(0, 0, 0);
        scene.add(droneGroup);

        // Viewport flags
        let followDrone = true;
        let showGrid = true;
        let showLandmarks = true;
        let autoRotate = false;

        function animate() {
            requestAnimationFrame(animate);

            if (autoRotate) {
                scene.rotation.z += 0.002;
            }

            if (followDrone) {
                // Smoothly pan camera position together with target so view angle doesn't wildly tilt or swing the grid
                const prevTarget = controls.target.clone();
                controls.target.lerp(droneGroup.position, 0.05);
                const panDelta = controls.target.clone().sub(prevTarget);
                camera.position.add(panDelta);
            }

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

        // 3D Controls
        function setCamView(mode) {
            followDrone = false;
            document.getElementById("btnFollow").classList.remove("active");
            const target = droneGroup.position;

            if (mode === "top") {
                camera.position.set(target.x, target.y, target.z + 12);
                camera.up.set(0, 1, 0);
                controls.target.copy(target);
            } else if (mode === "side") {
                camera.position.set(target.x + 10, target.y, target.z + 1.0);
                camera.up.set(0, 0, 1);
                controls.target.copy(target);
            } else if (mode === "front") {
                camera.position.set(target.x, target.y + 10, target.z + 1.0);
                camera.up.set(0, 0, 1);
                controls.target.copy(target);
            } else {
                camera.position.set(target.x - 5, target.y - 6, target.z + 4.5);
                camera.up.set(0, 0, 1);
                controls.target.copy(target);
            }
        }

        function toggleFollow() {
            followDrone = !followDrone;
            document.getElementById("btnFollow").classList.toggle("active", followDrone);
        }

        function toggleGrid() {
            showGrid = !showGrid;
            gridHelper.visible = showGrid;
            if (typeof gridLabelsGroup !== "undefined") gridLabelsGroup.visible = showGrid;
            document.getElementById("btnGrid").classList.toggle("active", showGrid);
        }

        function togglePoints() {
            showLandmarks = !showLandmarks;
            landmarkPoints.visible = showLandmarks;
            if (typeof nearbyPoints !== "undefined") nearbyPoints.visible = showLandmarks;
            document.getElementById("btnPoints").classList.toggle("active", showLandmarks);
        }

        let showTraj = true;
        function toggleTraj() {
            showTraj = !showTraj;
            trajectoryLine.visible = showTraj;
            document.getElementById("btnTraj").classList.toggle("active", showTraj);
        }

        function toggleAutoRotate() {
            autoRotate = !autoRotate;
            document.getElementById("btnAutoRot").classList.toggle("active", autoRotate);
        }

        // ==================== WEBSOCKET TELEMETRY STREAM ====================
        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        let ws = new WebSocket(wsUrl);

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            // 1. Update Charts
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

            // Att chart with real Euler angles
            attChart.data.datasets[0].data.shift();
            attChart.data.datasets[0].data.push(data.pitch);
            attChart.data.datasets[1].data.shift();
            attChart.data.datasets[1].data.push(data.roll);
            attChart.data.datasets[2].data.shift();
            attChart.data.datasets[2].data.push(data.yaw);
            attChart.update();

            // Disp chart with real trajectory translation
            dispChart.data.datasets[0].data.shift();
            dispChart.data.datasets[0].data.push(data.pos_y);
            dispChart.data.datasets[1].data.shift();
            dispChart.data.datasets[1].data.push(data.pos_z);
            dispChart.update();

            // 2. Update Compact Navigation Telemetry Display
            document.getElementById("dispPose").innerText = 
                `X: ${data.pos_x.toFixed(3)} | Y: ${data.pos_y.toFixed(3)} | Z: ${data.pos_z.toFixed(3)}`;
            document.getElementById("dispAngles").innerText = 
                `R: ${data.roll.toFixed(1)}° | P: ${data.pitch.toFixed(1)}° | Y: ${data.yaw.toFixed(1)}°`;
            document.getElementById("dispFeatures").innerText = 
                `Det: ${data.features} | Trk: ${data.tracked} | Inliers: ${data.inliers} (${data.inlier_ratio}%)`;

            const statusEl = document.getElementById("dispStatus");
            statusEl.innerText = data.status;
            statusEl.className = "status-badge " + (
                data.status === "STABLE" ? "badge-green" :
                (data.status === "DEGRADED" ? "badge-yellow" :
                (data.status === "UNSTABLE" ? "badge-orange" :
                (data.status === "REINITIALIZING" ? "badge-blue" : "badge-red")))
            );

            // 3. Update 3D Drone Marker to Real Position & Real Attitude Orientation
            droneGroup.position.set(data.pos_x, data.pos_y, data.pos_z);
            // Apply Euler rotation: X=pitch, Y=roll, Z=yaw
            const r_rad = THREE.MathUtils.degToRad(data.roll);
            const p_rad = THREE.MathUtils.degToRad(data.pitch);
            const y_rad = THREE.MathUtils.degToRad(data.yaw);
            droneGroup.rotation.set(p_rad, r_rad, y_rad, 'ZYX');

            // 4. Update Complete Continuous Flight Path Trajectory Ribbon
            if (data.trajectory && data.trajectory.length > 0) {
                const trajLen = data.trajectory.length;
                const posArr = trajGeo.attributes.position;
                const colArr = trajGeo.attributes.color;

                for (let i = 0; i < trajLen && i < MAX_TRAJ_PTS; i++) {
                    posArr.setXYZ(i, data.trajectory[i][0], data.trajectory[i][1], data.trajectory[i][2]);
                    
                    // Trajectory color gradient: Green (Start) -> Cyan (Path) -> Bright Red (Current Head)
                    const progress = i / Math.max(1, trajLen - 1);
                    if (i === 0) {
                        colArr.setXYZ(i, 0.13, 0.77, 0.36); // Emerald start
                    } else if (progress > 0.96) {
                        colArr.setXYZ(i, 0.95, 0.27, 0.27); // Vivid red tip
                    } else {
                        colArr.setXYZ(i, 0.18 + 0.12 * progress, 0.85, 0.75 - 0.2 * progress); // Cyan gradient
                    }
                }
                posArr.needsUpdate = true;
                colArr.needsUpdate = true;
                trajGeo.setDrawRange(0, Math.min(trajLen, MAX_TRAJ_PTS));
            }

            // 5. Update Real Triangulated 3D Landmarks & Highlight Nearby Points
            let nearbyCount = 0;
            if (data.landmarks_3d && data.landmarks_3d.length > 0) {
                const count = data.landmarks_3d.length;
                const ptsArr = landmarkGeo.attributes.position;
                const colArr = landmarkGeo.attributes.color;
                const nearArr = nearbyGeo.attributes.position;

                for (let i = 0; i < count && i < MAX_LANDMARKS; i++) {
                    const lm = data.landmarks_3d[i];
                    ptsArr.setXYZ(i, lm.x, lm.y, lm.z);

                    // 3D Euclidean distance to current drone position
                    const dist3d = Math.hypot(lm.x - data.pos_x, lm.y - data.pos_y, lm.z - data.pos_z);
                    
                    if (dist3d <= 2.5 && nearbyCount < MAX_NEARBY) {
                        // Nearby Point: Add to highlighted points buffer (rendered as large, glowing neon points)
                        nearArr.setXYZ(nearbyCount, lm.x, lm.y, lm.z);
                        nearbyCount++;
                    }

                    // Distant/Background landmarks color coded by depth
                    if (dist3d < 4.0) {
                        colArr.setXYZ(i, 0.22, 0.74, 0.97); // Cyan
                    } else if (dist3d < 9.0) {
                        colArr.setXYZ(i, 0.50, 0.45, 0.90); // Lavender
                    } else {
                        colArr.setXYZ(i, 0.60, 0.25, 0.80); // Purple
                    }
                }
                ptsArr.needsUpdate = true;
                colArr.needsUpdate = true;
                landmarkGeo.setDrawRange(0, Math.min(count, MAX_LANDMARKS));

                nearArr.needsUpdate = true;
                nearbyGeo.setDrawRange(0, nearbyCount);
            }

            // Update HUD Coordinates Bar at the bottom of the 3D Grid
            const hudNearby = document.getElementById("hudNearbyCount");
            if (hudNearby) {
                hudNearby.innerText = `${nearbyCount} pts (dist \u2264 2.5m)`;
            }
            const hudPos = document.getElementById("hudDronePos");
            if (hudPos) {
                hudPos.innerText = `X: ${data.pos_x.toFixed(3)} | Y: ${data.pos_y.toFixed(3)} | Z: ${data.pos_z.toFixed(3)}`;
            }

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
            btnPlay.innerHTML = data.running ? "<span>⏸</span> Pause" : "<span>▶</span> Resume";
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
                    document.getElementById("videoStreamImg").src = "/video_feed?t=" + new Date().getTime();
                });
        }

        function togglePause() {
            fetch("/api/toggle_pause", { method: "POST" })
                .then(r => r.json())
                .then(d => {
                    showToast(d.running ? "VO Tracking Resumed" : "VO Tracking Paused");
                });
        }

        function triggerReinit() {
            fetch("/api/reinit", { method: "POST" })
                .then(() => {
                    trajGeo.setDrawRange(0, 1);
                    if (typeof landmarkGeo !== "undefined") landmarkGeo.setDrawRange(0, 0);
                    if (typeof nearbyGeo !== "undefined") nearbyGeo.setDrawRange(0, 0);
                    showToast("VO Engine & Trajectory Re-initialized");
                });
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

        // Auto stream reconnection and stall watchdog
        let streamReconnectTimer = null;
        function handleStreamError(img) {
            clearTimeout(streamReconnectTimer);
            streamReconnectTimer = setTimeout(() => {
                img.src = "/video_feed?t=" + Date.now();
            }, 600);
        }

        // Watchdog: If camera stream freezes or disconnects, auto-reconnect
        setInterval(() => {
            const img = document.getElementById("videoStreamImg");
            if (img && (!img.complete || img.naturalWidth === 0)) {
                img.src = "/video_feed?t=" + Date.now();
            }
        }, 3500);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_CONTENT)


if __name__ == "__main__":
    uvicorn.run("server.py:app", host="0.0.0.0", port=8501, log_level="info")
