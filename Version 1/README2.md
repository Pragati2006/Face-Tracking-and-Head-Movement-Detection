# Lightweight Face & Landmark Detection (Haar Cascades)

This is a lightweight version of the face tracking system that uses **OpenCV's built-in Haar Cascades** instead of MediaPipe. It is designed for systems where MediaPipe might be too heavy or unavailable.

## 🚀 Features

- **Multi-Object Detection**: Detects faces, eyes, and smiles using pre-trained XML models.
- **Directional Analysis**: Computes head orientation (Left, Right, Up, Down, Center) based on the relative position of the face in the frame.
- **Dynamic Confidence**: Calculates a confidence score based on the magnitude of movement from the screen's center.
- **Visual Feedback**: Draws bounding boxes, center points, and landmarks (eyes/mouth) for real-time visualization.
- **Optimized for CPU**: Runs efficiently on almost any hardware without requiring a GPU.

## 🛠️ Tech Stack

- **OpenCV**: Handles image acquisition, grayscale conversion, cascade classification, and UI overlays.
- **Haar Cascades**: Utilizes the Viola-Jones object detection framework for specialized detection of facial features.

## 📦 Installation

1. **Install specialized dependencies**:
   ```bash
   pip install -r requirements2.txt
   ```

## 🖥️ Usage

Run the lightweight tracker:

```bash
python main2.py
```

### Controls
- **ESC**: Exit the application.



---
*Created as a lightweight alternative for basic facial feature detection.*
