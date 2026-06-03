from pathlib import Path

import cv2
import joblib
import numpy as np
from pygame import mixer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cnnCat2.joblib"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.joblib"
ALARM_PATH = BASE_DIR / "alarm.wav"

print("=" * 60)
print("Driver Drowsiness Detection - Real-time Detection")
print("=" * 60)

sound = None
try:
    mixer.init()
    sound = mixer.Sound(str(ALARM_PATH))
    print("Audio enabled")
except Exception as exc:
    print(f"Warning: audio unavailable ({exc})")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

print("Haar cascade classifiers loaded")

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print("ERROR: Model files not found!")
    print("Please run model_training_sklearn.py first")
    raise SystemExit(1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot access webcam!")
    raise SystemExit(1)

print("Webcam initialized")
print("\nStarting detection...")
print("Press 'q' to quit\n")

font = cv2.FONT_HERSHEY_SIMPLEX
drowsiness_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

        cv2.rectangle(
            frame,
            (0, height - 50),
            (300, height),
            (0, 0, 0),
            thickness=cv2.FILLED,
        )

        eye_closed_count = 0

        for (ex, ey, ew, eh) in eyes:
            eye = frame[ey : ey + eh, ex : ex + ew]
            eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            eye_resized = cv2.resize(eye_gray, (24, 24))
            eye_flat = eye_resized.flatten() / 255.0

            prediction = model.predict([eye_flat])[0]
            class_name = str(label_encoder.inverse_transform([prediction])[0]).strip().lower()

            if class_name in {"close", "closed"}:
                eye_closed_count += 1
                color = (0, 0, 255)
                label = "Closed"
            else:
                color = (0, 255, 0)
                label = "Open"

            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(
                frame,
                label,
                (ex, max(20, ey - 8)),
                font,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if eye_closed_count > 0:
            drowsiness_count += 1
        else:
            drowsiness_count = max(0, drowsiness_count - 1)

        if drowsiness_count > 10:
            status = "CLOSED!"
            status_color = (0, 0, 255)
        elif drowsiness_count < 5:
            status = "OPEN"
            status_color = (0, 255, 0)
        else:
            status = "ALERT"
            status_color = (0, 255, 255)

        cv2.putText(
            frame,
            f"Eyes: {status}",
            (10, height - 20),
            font,
            0.8,
            status_color,
            2,
        )
        cv2.putText(
            frame,
            f"Score: {drowsiness_count}",
            (150, height - 20),
            font,
            0.8,
            (255, 255, 255),
            2,
        )

        if drowsiness_count > 15 and sound:
            try:
                sound.play()
            except Exception:
                pass

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nDetection stopped by user")
            break

except KeyboardInterrupt:
    print("\nDetection interrupted")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("=" * 60)
    print("Webcam released. Program ended.")
    print("=" * 60)
