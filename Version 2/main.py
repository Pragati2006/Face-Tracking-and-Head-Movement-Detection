import cv2
import numpy as np
import time
from face_tracker import MultiFaceTracker
from analytics import generate_session_report

def main():
    cap = cv2.VideoCapture(0)
    
    # Initialize Tracker (FaceDetection for live scores and stable ID tracking)
    tracker = MultiFaceTracker(min_detection_confidence=0.5)
    
    # Store analytics data for all faces across the session
    session_analytics = []
    
    # Timing for FPS calculation
    prev_time = 0
    
    print("--------------------------------------------------")
    print("Multi-Face Movement Analytics started.")
    print("Labels: Face 1, Face 2, etc. (Fixed to each person)")
    print("Commands: 'q' - Quit and Generate Report")
    print("--------------------------------------------------")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        h_f, w_f, _ = frame.shape
        
        # Process frame
        _, face_list, lost_face_analytics = tracker.process_frame(frame)
        
        # Keep track of analytics from faces that left the frame
        if lost_face_analytics:
            session_analytics.extend(lost_face_analytics)
        
        # Overlay UI for each face currently visible
        for face in face_list:
            bbox = face['bbox']
            direction = face['direction']
            fid = face['id'] 
            conf = face.get('confidence', 0.0)
            x, y, w_box, h_box = bbox
            
            # Get the live change count for this specific tracked face
            face_state = tracker.tracked_faces.get(fid)
            changes = face_state.change_count if face_state else 0
            
            # Using the direction, confidence and changes for the label
            label = f"{direction}"
            sub_label = f"ID: {fid} | C:{changes} | {conf:.2f}"
            
            # Draw movement direction on top of the bounding box
            cv2.putText(frame, label, (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, sub_label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            
            # Bounding Box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w_f - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show result
        cv2.imshow('Face Tracking: Movement Analytics', frame)
        
        # Input handling
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Collect analytics for currently tracking faces before exit
            for fid, fs in tracker.tracked_faces.items():
                fs.finalize()
                session_analytics.append(fs.get_summary())
                
            print("\n--------------------------------------------------")
            print("Session Ended. Generating report...")
            
            report_name = generate_session_report(session_analytics)
            if report_name:
                print(f"Report Generated: {report_name}")
            else:
                print("No analytics data collected to generate report.")
            print("--------------------------------------------------")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
