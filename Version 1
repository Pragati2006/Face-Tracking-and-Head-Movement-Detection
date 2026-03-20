import cv2


def draw_label(img, text, pos, font_scale=0.6):
    """Draw subtle, readable text with a soft outline."""
    x, y = pos
    # Shadow/outline
    cv2.putText(
        img,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    # Main text
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


# Use OpenCV's built-in detectors (Haar cascades)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)  # ignore very small detections
    )

    h, w, _ = frame.shape
    movement = "No Face"
    confidence = 0.0

    if len(faces) > 0:
        # Take the largest face (closest to camera)
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        # Face "landmarks": box corners and center
        corners = [
            (x, y),  # top-left
            (x + fw, y),  # top-right
            (x, y + fh),  # bottom-left
            (x + fw, y + fh),  # bottom-right
        ]
        for (px, py) in corners:
            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

        # Compute face center
        cx = x + fw // 2
        cy = y + fh // 2

        # Draw center point
        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)

        # Detect eyes and mouth region inside the face (extra landmarks)
        roi_gray = gray[y:y + fh, x:x + fw]
        roi_color = frame[y:y + fh, x:x + fw]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            ecx = x + ex + ew // 2
            ecy = y + ey + eh // 2
            cv2.circle(frame, (ecx, ecy), 4, (0, 0, 255), -1)

        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=25,
            minSize=(25, 25)
        )
        for (sx, sy, sw, sh) in smiles:
            mcx = x + sx + sw // 2
            mcy = y + sy + sh // 2
            cv2.circle(frame, (mcx, mcy), 4, (0, 165, 255), -1)

        # Normalize center to [0, 1]
        nx = cx / w
        ny = cy / h

        # Offsets from ideal center
        dx = nx - 0.5
        dy = ny - 0.5

        # Decide which axis dominates: horizontal vs vertical
        movement = "Center"
        center_dead_zone = 0.08  # small area counted as perfectly centered

        if abs(dx) < center_dead_zone and abs(dy) < center_dead_zone:
            movement = "Center"
        else:
            if abs(dx) > abs(dy):
                # Horizontal movement dominates
                if dx < 0:
                    movement = "Looking Left"
                else:
                    movement = "Looking Right"
            else:
                # Vertical movement dominates
                if dy < 0:
                    movement = "Looking Up"
                else:
                    movement = "Looking Down"

        # Confidence grows with distance from center (outside the dead zone)
        max_offset_for_conf = 0.3
        offset_mag = max(abs(dx), abs(dy))

        if movement == "Center":
            # Closer to exact center -> higher confidence
            confidence = 1.0 - min(offset_mag / max_offset_for_conf, 1.0)
        else:
            # Further from center (in dominant direction) -> higher confidence
            raw = (offset_mag - center_dead_zone) / (
                max_offset_for_conf - center_dead_zone
            )
            confidence = max(0.0, min(raw, 1.0))

        confidence = max(0.0, min(confidence, 1.0))
        label = f"{movement} ({confidence * 100:.0f}%)"

        # Draw label on top of the face box (subtle style)
        draw_label(
            frame,
            label,
            (x, max(y - 15, 25)),
            font_scale=0.55,
        )

    # Also show movement text at top-left, smaller and subtle
    draw_label(
        frame,
        movement,
        (20, 35),
        font_scale=0.6,
    )

    cv2.imshow("Face Tracking & Head Movement (OpenCV)", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
