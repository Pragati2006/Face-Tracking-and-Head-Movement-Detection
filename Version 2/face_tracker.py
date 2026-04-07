import cv2
import numpy as np
import mediapipe.python.solutions.face_detection as mp_face_detection
import math
import time
from kalman_filter import KalmanFilter2D

class FaceState:
    """Represents the state of a single face, including its Kalman Filter and history."""
    def __init__(self, face_id, start_pos):
        self.face_id = face_id
        # dt=1/30.0 assuming ~30fps, adjust if needed
        self.kf = KalmanFilter2D(dt=0.033) 
        self.kf.update(start_pos[0], start_pos[1])
        
        self.missed_frames = 0
        self.session_start_time = time.time()
        
        # History for Analytics
        self.last_direction = None
        self.direction_start_time = time.time()
        self.durations = {d: 0.0 for d in ["Center", "Left", "Right", "Up", "Down"]}
        self.change_count = 0

    def predict(self):
        return self.kf.predict()

    def update(self, pos, direction):
        self.kf.update(pos[0], pos[1])
        self.missed_frames = 0
        
        now = time.time()
        if self.last_direction != direction:
            if self.last_direction is not None:
                self.durations[self.last_direction] += (now - self.direction_start_time)
                self.change_count += 1
            self.last_direction = direction
            self.direction_start_time = now

    def finalize(self):
        if self.last_direction is not None:
            self.durations[self.last_direction] += (time.time() - self.direction_start_time)
            self.last_direction = None

    def get_summary(self):
        total_time = sum(self.durations.values())
        return {
            'face_id': self.face_id,
            'durations': self.durations,
            'total_time': total_time,
            'change_count': self.change_count,
            'session_duration': time.time() - self.session_start_time
        }

class MultiFaceTracker:
    def __init__(self, min_detection_confidence=0.5):
        self.face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=0
        )
        
        self.tracked_faces = {} 
        self.next_face_id = 0
        self.max_missed_frames = 15 # Increased buffer for stability
        
        self.center_zone_x, self.center_zone_y = 0.15, 0.15
        
    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        detections = [] 
        if results.detections:
            for d in results.detections:
                bbox = d.location_data.relative_bounding_box
                score = d.score[0]
                nose = d.location_data.relative_keypoints[2]
                detections.append({
                    'bbox': (int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)),
                    'nose_pos': (int(nose.x*w), int(nose.y*h)),
                    'score': score
                })
        
        # 1. Predict all current faces to get search positions
        predictions = {}
        for fid, fs in self.tracked_faces.items():
            predictions[fid] = fs.predict()
            fs.missed_frames += 1

        # 2. Association (Detections to Tracks)
        unmatched_detections = list(range(len(detections)))
        matched_results = []
        
        # Increased matching threshold (adjustable based on resolution)
        # 20% of frame width is a safe bet for identity stability
        max_matching_dist = w * 0.2 
        
        for fid, fs in list(self.tracked_faces.items()):
            if not unmatched_detections:
                break
            
            best_idx = -1
            min_dist = max_matching_dist
            pred_pos = predictions[fid]
            
            for i in unmatched_detections:
                dist = self._calculate_distance(pred_pos, detections[i]['nose_pos'])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                det = detections[best_idx]
                # Calculate direction relative to the FACE bounding box, not frame center
                direction = self._get_direction(det['nose_pos'], det['bbox'])
                fs.update(det['nose_pos'], direction)
                unmatched_detections.remove(best_idx)
                
                matched_results.append({
                    'id': fid,
                    'bbox': det['bbox'],
                    'direction': direction,
                    'confidence': det['score']
                })

        # 3. Handle Remaining Detections (New Faces)
        for i in unmatched_detections:
            det = detections[i]
            direction = self._get_direction(det['nose_pos'], det['bbox'])
            new_fs = FaceState(self.next_face_id, det['nose_pos'])
            new_fs.update(det['nose_pos'], direction)
            self.tracked_faces[self.next_face_id] = new_fs
            
            matched_results.append({
                'id': self.next_face_id,
                'bbox': det['bbox'],
                'direction': direction,
                'confidence': det['score']
            })
            self.next_face_id += 1

        # 4. Handle Lost Tracks
        completed_analytics = []
        lost = [fid for fid, fs in self.tracked_faces.items() if fs.missed_frames > self.max_missed_frames]
        for fid in lost:
            self.tracked_faces[fid].finalize()
            completed_analytics.append(self.tracked_faces[fid].get_summary())
            del self.tracked_faces[fid]
            
        return frame, matched_results, completed_analytics

    def _get_direction(self, nose_pos, bbox):
        """
        Calculates direction relative to the face's own center.
        nose_pos: (nx, ny)
        bbox: (x, y, w, h)
        """
        bx, by, bw, bh = bbox
        nx, ny = nose_pos
        
        # Find local center of the bounding box
        center_x = bx + bw / 2
        center_y = by + bh / 2
        
        # Calculate offset from center (-1.0 to 1.0)
        # bw/2 is the distance from center to edge.
        ox = (nx - center_x) / (bw / 2) if bw > 0 else 0
        oy = (ny - center_y) / (bh / 2) if bh > 0 else 0
        
        # sensitivity for local movement
        # Lowered to 0.08 to be more responsive to head turns
        sens_x, sens_y = 0.08, 0.08 
        
        if abs(ox) < sens_x and abs(oy) < sens_y:
            return "Center"
        
        if abs(ox) > abs(oy):
            return "Right" if ox > 0 else "Left"
        else:
            return "Down" if oy > 0 else "Up"
