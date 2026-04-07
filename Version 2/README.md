# Multi-Face Real-Time Tracker & Movement Analytics

A high-performance Python-based computer vision application that tracks multiple faces in real-time, predicts their movement using Kalman Filters, and analyzes directional attention. The system generates a consolidated visual report at the end of each session, providing insights into directional attention.

## 🚀 Features

- **Real-Time Multi-Face Tracking**: Simultaneously detects and tracks multiple individuals with stable ID assignment.
- **Directional Attention Analysis**: Calculates if a person is looking **Center**, **Left**, **Right**, **Up**, or **Down** based on facial landmark orientation.
- **Predictive Smoothing**: Uses **Kalman Filters** (2D) to predict face positions and maintain track stability even during brief occlusions or fast movements.
- **Automated Analytics Dashboard**: Generates a consolidated PNG report including:
    - **Directional Attention Distribution**: A pie chart showing time spent in each direction across all faces.
    - **Attention Score**: Percentage of time spent looking directly at the screen (**Center**).
    - **Observation Time**: Measures total time each individual was tracked.
- **Aggregate Reporting**: Saves a time-stamped visual summary combining data from all detected faces into one high-level dashboard.

## 🛠️ Tech Stack

- **OpenCV**: Core image processing and UI rendering.
- **MediaPipe**: Blazingly fast face detection and landmark extraction.
- **NumPy**: Efficient numerical operations for Kalman Filter matrices.
- **Matplotlib**: Generation of analytical charts and session reports.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
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
- **'q'**: Quit the application and automatically generate the session report.

The system will display the live camera feed with bounding boxes, face IDs, and real-time directional labels.

## 📊 Session Reports

Upon exiting, the application saves an aggregate report (e.g., `session_report_20260407_223100.png`) in the project directory. This report provides a visual breakdown of the session's overall metrics, measuring the total attention of all participants during the observation period.

---
*Created as part of a real-time computer vision experiment.*
