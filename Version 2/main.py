import cv2
import numpy as np
import time
from face_tracker import MultiFaceTracker
from analytics import generate_session_report

def main():
    cap = cv2.VideoCapture(0)
    
    # Initialize Tracker (Stable ID tracking using histograms)
    tracker = MultiFaceTracker(min_detection_confidence=0.5)
    
    # Timing for FPS calculation
    prev_time = 0
    
    print("--------------------------------------------------")
    print("Multi-Face Movement Analytics started.")
    print("Status: Persistent Face Tracking Active.")
    print("Labels: Face IDs are assigned visually and persist.")
    print("Commands: 'q' - Quit and Generate Cumulative Report")
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
        _, face_list, _ = tracker.process_frame(frame)
        
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
            
            # Label display
            label = f"{direction}"
            sub_label = f"ID: {fid} | Changes: {changes} | {conf:.2f}"
            
            # Bounding Box and Directions
            cv2.putText(frame, label, (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if direction != "Center" else (255, 0, 0), 2)
            cv2.putText(frame, sub_label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        # FPS calculation and display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w_f - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Face Tracking: Movement Analytics', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n--------------------------------------------------")
            print("Session Ended. Compiling cumulative report...")
            
            # Get all analytics from the tracker's persistent database
            all_analytics = tracker.get_all_session_analytics()
            
            report_name = generate_session_report(all_analytics)
            if report_name:
                print(f"Cumulative Report Generated: {report_name}")
            else:
                print("No analytics data collected to generate report.")
            print("--------------------------------------------------")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
