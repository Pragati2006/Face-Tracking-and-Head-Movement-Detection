import cv2
import numpy as np
import mediapipe.python.solutions.face_detection as mp_face_detection
import math
import time
from kalman_filter import KalmanFilter2D

class FaceState:
    """Represents the state of a single face, including its Kalman Filter and history."""
    def __init__(self, face_id, start_pos, hist=None):
        self.face_id = face_id
        # dt=1/30.0 assuming ~30fps
        self.kf = KalmanFilter2D(dt=0.033) 
        self.kf.update(start_pos[0], start_pos[1])
        
        self.missed_frames = 0
        self.session_start_time = time.time()
        self.hist = hist # Visual fingerprint
        
        # History for Analytics
        self.last_direction = None
        self.direction_start_time = time.time()
        self.durations = {d: 0.0 for d in ["Center", "Left", "Right", "Up", "Down"]}
        self.change_count = 0

    def predict(self):
        return self.kf.predict()

    def update(self, pos, direction, new_hist=None):
        self.kf.update(pos[0], pos[1])
        self.missed_frames = 0
        
        # Smoothly update histogram (visual fingerprint)
        if new_hist is not None and self.hist is not None:
            self.hist = cv2.addWeighted(self.hist, 0.9, new_hist, 0.1, 0)
        elif new_hist is not None:
            self.hist = new_hist

        now = time.time()
        if self.last_direction != direction:
            if self.last_direction is not None:
                self.durations[self.last_direction] += (now - self.direction_start_time)
                self.change_count += 1
            self.last_direction = direction
            self.direction_start_time = now

    def finalize(self):
        if self.last_direction is not None:
            now = time.time()
            self.durations[self.last_direction] += (now - self.direction_start_time)
            # Reset direction timer for next time they appear
            self.direction_start_time = now
            self.last_direction = None

    def get_summary(self):
        total_time = sum(self.durations.values())
        return {
            'face_id': self.face_id,
            'durations': self.durations.copy(),
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
        
        self.tracked_faces = {} # Active in frame: {fid: FaceState}
        self.face_database = {} # Persistent storage: {fid: {hist: array, durations: dict}}
        self.next_face_id = 0
        self.max_missed_frames = 45 # Buffer for faces turning away or fast movement
        
        # Re-ID Sensitivity (Histogram Correlation)
        # Lowered to 0.45 to be more forgiving of head rotation/lighting
        self.reid_threshold = 0.45 
        
    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _get_face_histogram(self, frame, bbox):
        """Extracts a color histogram for the face region as a fingerprint."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0: return None
        
        # Clamp coordinates to frame
        fh, fw, _ = frame.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x+w), min(fh, y+h)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0: return None
        
        hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_face], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _calculate_iou(self, bbox1, bbox2):
        """Calculates Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def process_frame(self, frame):
        h, w, _ = frame.shape
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        detections = [] 
        if results.detections:
            for d in results.detections:
                bbox_rel = d.location_data.relative_bounding_box
                score = d.score[0]
                nose = d.location_data.relative_keypoints[2]
                
                # Convert rel to abs
                bx, by = int(bbox_rel.xmin*w), int(bbox_rel.ymin*h)
                bw, bh = int(bbox_rel.width*w), int(bbox_rel.height*h)
                
                # Extract histogram fingerprint
                hist = self._get_face_histogram(frame, (bx, by, bw, bh))
                
                detections.append({
                    'bbox': (bx, by, bw, bh),
                    'nose_pos': (int(nose.x*w), int(nose.y*h)),
                    'score': score,
                    'hist': hist
                })
        
        # 1. Predict all current faces
        predictions = {}
        for fid, fs in self.tracked_faces.items():
            predictions[fid] = fs.predict()
            fs.missed_frames += 1

        # 2. Association (Detections to Active Tracks) using Hybrid Scoring
        unmatched_detections = list(range(len(detections)))
        matched_results = []
        
        potential_matches = []
        for fid, fs in self.tracked_faces.items():
            pred_pos = predictions[fid]
            for i in unmatched_detections:
                det = detections[i]
                
                # Distance score (normalized 0.0 to 1.0+)
                dist_score = self._calculate_distance(pred_pos, det['nose_pos']) / w
                
                # Visual similarity score (converted to 0.0-1.0 where 0 is better)
                sim_score = 1.0
                if fs.hist is not None and det['hist'] is not None:
                    sim_score = 1.0 - cv2.compareHist(fs.hist, det['hist'], cv2.HISTCMP_CORREL)
                
                # Total weighted score (Distance matters most for tracking, sim for validation)
                total_score = dist_score * 0.7 + sim_score * 0.3
                
                # Broad threshold to allow sticky tracking
                if dist_score < 0.45 or sim_score < 0.4:
                    potential_matches.append((total_score, fid, i))

        # Greedy match based on best scores
        potential_matches.sort(key=lambda x: x[0])
        used_fids = set()
        used_dets = set()
        
        for _, fid, i in potential_matches:
            if fid in used_fids or i in used_dets:
                continue
            
            det = detections[i]
            direction = self._get_direction(det['nose_pos'], det['bbox'])
            self.tracked_faces[fid].update(det['nose_pos'], direction, det['hist'])
            
            matched_results.append({
                'id': fid, 'bbox': det['bbox'],
                'direction': direction, 'confidence': det['score']
            })
            
            used_fids.add(fid)
            used_dets.add(i)
            unmatched_detections.remove(i)

        # 3. Persistent Re-Identification / New Faces
        for i in unmatched_detections:
            det = detections[i]
            direction = self._get_direction(det['nose_pos'], det['bbox'])
            new_hist = det['hist']
            
            best_fid = -1
            max_sim = self.reid_threshold
            
            if new_hist is not None:
                for fid, db_data in self.face_database.items():
                    if fid in self.tracked_faces: continue
                    
                    sim = cv2.compareHist(new_hist, db_data['hist'], cv2.HISTCMP_CORREL)
                    if sim > max_sim:
                        max_sim = sim
                        best_fid = fid
            
            if best_fid != -1:
                # Welcome back old friend
                fs = FaceState(best_fid, det['nose_pos'], new_hist)
                fs.durations = self.face_database[best_fid]['durations'].copy()
                fs.change_count = self.face_database[best_fid]['change_count']
                fs.update(det['nose_pos'], direction)
                self.tracked_faces[best_fid] = fs
                found_id = best_fid
            else:
                # New face detected
                new_fs = FaceState(self.next_face_id, det['nose_pos'], new_hist)
                new_fs.update(det['nose_pos'], direction)
                self.tracked_faces[self.next_face_id] = new_fs
                found_id = self.next_face_id
                
                self.face_database[found_id] = {
                    'hist': new_hist,
                    'durations': {d: 0.0 for d in ["Center", "Left", "Right", "Up", "Down"]},
                    'change_count': 0
                }
                self.next_face_id += 1
            
            matched_results.append({
                'id': found_id, 'bbox': det['bbox'],
                'direction': direction, 'confidence': det['score']
            })

        # 4. Handle Lost Tracks
        lost = [fid for fid, fs in self.tracked_faces.items() if fs.missed_frames > self.max_missed_frames]
        for fid in lost:
            fs = self.tracked_faces[fid]
            fs.finalize()
            self.face_database[fid]['durations'] = fs.durations.copy()
            self.face_database[fid]['change_count'] = fs.change_count
            self.face_database[fid]['hist'] = fs.hist 
            del self.tracked_faces[fid]
            
        return frame, matched_results, []

    def get_all_session_analytics(self):
        """Compiles analytics from both active and inactive faces in the database."""
        all_summaries = []
        
        # Capture current state of all faces ever seen
        for fid, db_data in self.face_database.items():
            if fid in self.tracked_faces:
                fs = self.tracked_faces[fid]
                fs.finalize()
                summary = fs.get_summary()
            else:
                summary = {
                    'face_id': fid,
                    'durations': db_data['durations'],
                    'total_time': sum(db_data['durations'].values()),
                    'change_count': db_data['change_count']
                }
            
            # FILTER: Only include faces that were seen for at least 2.0 seconds
            # This removes "ghost" IDs or very brief accidental detections
            if summary['total_time'] >= 2.0:
                all_summaries.append(summary)
            
        # Sort by duration to keep most prominent faces first
        all_summaries.sort(key=lambda x: x['total_time'], reverse=True)
        return all_summaries

    def _get_direction(self, nose_pos, bbox):
        bx, by, bw, bh = bbox
        nx, ny = nose_pos
        center_x = bx + bw / 2
        center_y = by + bh / 2
        
        ox = (nx - center_x) / (bw / 2) if bw > 0 else 0
        oy = (ny - center_y) / (bh / 2) if bh > 0 else 0
        
        sens_x, sens_y = 0.08, 0.08 
        
        if abs(ox) < sens_x and abs(oy) < sens_y:
            return "Center"
        
        if abs(ox) > abs(oy):
            return "Right" if ox > 0 else "Left"
        else:
            return "Down" if oy > 0 else "Up"
