import cv2
import time

# Load face and eye detection models
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

closed_start = None
DROWSY_TIME = 2.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.3, 5
    )

    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

        face_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_gray, 1.3, 5
        )

        if len(eyes) > 0:
            eyes_detected = True

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (255, 0, 0),
                    2
                )

    # Drowsiness detection
    if len(faces) > 0 and not eyes_detected:

        if closed_start is None:
            closed_start = time.time()

        elapsed = time.time() - closed_start

        if elapsed >= DROWSY_TIME:
            cv2.putText(
                frame,
                "DROWSINESS DETECTED!",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            print("\a")  # Computer alert sound

    else:
        closed_start = None

        cv2.putText(
            frame,
            "Driver Alert",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
