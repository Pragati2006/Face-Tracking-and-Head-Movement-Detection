# Multi-Face Real-Time Tracker & Cumulative Analytics

A high-performance Python-based computer vision application that tracks multiple faces in real-time, predicts their movement using Kalman Filters, and analyzes directional attention. The system now features **Persistent Face ID tracking** and generates a detailed **Cumulative Report** encompassing both individual and session-wide metrics.

## 🚀 Features

- **Persistent Face Re-Identification**: Uses **Visual Fingerprinting** (HSV Histograms) to recognize faces. If a person leaves and re-enters the frame, they retain their original **Face ID**.
- **Stable Multi-Face Tracking**: Implements a **Hybrid Association Logic** (Spatial + Visual) that prevents ID swapping and ensures tracking remains "sticky" even during fast movement.
- **Directional Attention Analysis**: Calculates if a person is looking **Center**, **Left**, **Right**, **Up**, or **Down** based on facial landmark orientation.
- **Predictive Smoothing**: Uses **Kalman Filters** (2D) to predict positions and maintain stability during brief occlusions.
- **Enhanced Cumulative Dashboard**: Generates a high-quality visual report including:
    - **Aggregate Section**: Overall attention distribution across all faces.
    - **Individual Section**: Dedicated stats for **each unique Face ID**, including their specific Attention Score, total visibility time, and movement frequency.
- **Noise Filtering**: Automatically ignores short-lived detections (under 2 seconds) to ensure the final report is accurate and clean.

## 🛠️ Tech Stack

- **OpenCV**: Core image processing, UI rendering, and histogram matching.
- **MediaPipe**: Blazingly fast face detection and landmark extraction.
- **NumPy**: Efficient numerical operations for Kalman Filter matrices.
- **Matplotlib**: Generation of the multi-section analytical session reports.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <https://github.com/Pragati2006/Face-Tracking-and-Head-Movement-Detection.git>
   cd "CV project"
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

Run the main script to start the tracking session:

```bash
python main.py
```

### Controls
- **'q'**: Quit the application and automatically compile the **Cumulative Report**.

The system will display the live camera feed with bounding boxes, persistent IDs, confidence scores, and real-time movement labels.

## 📊 Session Reports

Upon exiting, the application saves a cumulative report (e.g., `session_report_20260408_101500.png`). Unlike basic trackers, this report provides:
1.  **A Global Summary**: Total attention and metrics for the entire environment.
2.  **Per-Person Detail**: Individual pie charts and performance cards for every recognized face, allowing you to compare attention levels between participants.

--
*Created as part of a real-time computer vision experiment with advanced ID persistence.*
