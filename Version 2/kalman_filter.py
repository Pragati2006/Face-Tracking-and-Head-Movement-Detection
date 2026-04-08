import cv2
import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=1.0):
        # State vector: [x, y, dx, dy] (position and velocity)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Measurement matrix: we only measure x and y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Transition matrix: x = x_old + dx * dt, y = y_old + dy * dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Process noise covariance (Q): uncertainty in the model
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        
        # Measurement noise covariance (R): uncertainty in the measurement
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance matrix (P): initial uncertainty
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self):
        """Predict the next state."""
        return self.kf.predict()

    def update(self, x, y):
        """Update the state with new measurement (x, y)."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        return self.kf.statePost

    def get_velocity(self):
        """Returns the velocity vector (dx, dy)."""
        return self.kf.statePost[2], self.kf.statePost[3]
