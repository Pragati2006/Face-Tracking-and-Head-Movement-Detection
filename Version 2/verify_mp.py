import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")
try:
    from mediapipe.solutions import face_mesh
    print("SUCCESS: mediapipe.solutions.face_mesh is available.")
except ImportError as e:
    print(f"FAILURE: {e}")
except AttributeError as e:
    print(f"FAILURE: {e}")
