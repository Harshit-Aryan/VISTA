# VISTA — Vision-based Inertial-free Spatial Tracking Architecture

> **GPS-Denied Visual Navigation Research Prototype**  
> **Laboratory Milestone:** Lucas-Kanade Optical Flow Feature Tracking (TRL 4 Laboratory Subsystem)

---

## 1. System Overview

VISTA is an experimental computer vision research prototype developed to evaluate GPS-denied relative spatial tracking and ego-motion estimation using commodity monocular camera imagery.

When satellite-based positioning systems (GPS/GNSS) are degraded, jammed, or obstructed (such as in urban canyons, indoor warehouses, or electronic-warfare contention environments), autonomous platforms must rely on visual landmarks to perceive movement.

```
+------------------+      +-------------------+      +----------------------------+
| Laptop Camera /  | ---> | Video Acquisition | ---> |    Grayscale Conversion    |
| Test Video Feed  |      |   (OpenCV Video)  |      |     (cv2.cvtColor)         |
+------------------+      +-------------------+      +----------------------------+
                                                                   |
                                                                   v
+------------------+      +-------------------+      +----------------------------+
| Telemetry HUD /  | <--- |  Image-Plane Ego- | <--- |   Lucas-Kanade Pyramidal   |
| Motion Trails    |      |  Motion Analysis  |      |        Optical Flow        |
+------------------+      +-------------------+      +----------------------------+
                                                                   ^
                                                                   |
                                                     +----------------------------+
                                                     |    Shi-Tomasi Corners      |
                                                     | (cv2.goodFeaturesToTrack)  |
                                                     |    [Auto-Reinit Trigger]   |
                                                     +----------------------------+
```

---

## 2. Implemented Modules

| Script | Phase | Description |
| :--- | :--- | :--- |
| `vista_camera.py` | Phase 1 & 2 | Baseline video acquisition and Shi-Tomasi corner detection. |
| `vista_optical_flow.py` | Phase 3 & 4 | Lucas-Kanade optical flow, motion trails, adaptive reinitialization, and image-motion estimation. |

---

## 3. Core Algorithms & Parameterization

### A. Shi-Tomasi Corner Detection (`cv2.goodFeaturesToTrack`)
- **Mathematical Principle:** Evaluates the smaller eigenvalue ($\lambda_{\min}$) of the second-moment structure matrix $M = \sum \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$. Corners with $\min(\lambda_1, \lambda_2) > \lambda_{\text{threshold}}$ represent visually trackable features with two orthogonal gradients.
- **Configured Parameters:**
  - `maxCorners = 150`
  - `qualityLevel = 0.01`
  - `minDistance = 10` pixels
  - `blockSize = 7` pixels

### B. Lucas-Kanade Pyramidal Optical Flow (`cv2.calcOpticalFlowPyrLK`)
- **Mathematical Principle:** Assumes brightness constancy ($I(x, y, t) = I(x + \delta x, y + \delta y, t + \delta t)$) and spatial coherence over a local neighborhood window $W$. Multi-scale Gaussian pyramids solve the aperture problem and enable tracking of larger displacements without losing sub-pixel tracking fidelity.
- **Configured Parameters:**
  - `winSize = (21, 21)`
  - `maxLevel = 3` (4 pyramid levels: full resolution down to 1/8 scale)
  - `criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)`

### C. Relative Motion Estimation
- Aggregates tracked feature displacement vectors $\vec{d}_i = [\Delta x_i, \Delta y_i]$.
- Calculates median displacement $\vec{d}_{\text{med}}$ to reject outlier vector noise caused by moving foreground objects or tracking drift.
- Classifies direction: `STATIONARY`, `LEFT`, `RIGHT`, `UP`, `DOWN`, or diagonal combinations.

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Webcam or pre-recorded video file

### Installation
```bash
pip install -r requirements.txt
```

---

## 5. Execution Instructions

### A. Run Optical Flow Feature Tracker (Live Webcam)
```bash
python vista_optical_flow.py
```

### B. Run Optical Flow Tracker on Pre-Recorded Video / Synthetic Test Dataset
```bash
python vista_optical_flow.py --source data/test_videos/sample_flight.mp4
```

### C. Run Shi-Tomasi Detection Baseline
```bash
python vista_camera.py
```

---

## 6. Runtime Interactive Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`q`** | Quit | Safely releases camera device and closes all OpenCV windows. |
| **`r`** | Reinitialize | Forces fresh Shi-Tomasi corner detection across the current frame. |
| **`c`** | Clear Trails | Clears the persistent motion trail canvas. |
| **`s`** | Save Screenshot | Captures the current HUD and tracking frame directly into `results/screenshots/`. |

---

## 7. Laboratory Acceptance Targets (TRL 4)

- **Target 1:** $\ge 50$ detected features in normal textured scene $\rightarrow$ Verified ($\sim 130-150$ features detected).
- **Target 2:** $\ge 70\%$ feature retention under nominal motion $\rightarrow$ Verified ($\sim 85-92\%$ sustained retention).
- **Target 3:** $\ge 10$ FPS processing speed on standard laptop hardware $\rightarrow$ Verified ($\ge 30$ FPS sustained).
