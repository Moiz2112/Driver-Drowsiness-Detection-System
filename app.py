from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "DDDS_CNN" / "models" / "cnnCat2.joblib"
LABEL_ENCODER_PATH = BASE_DIR / "DDDS_CNN" / "models" / "label_encoder.joblib"


class DrowsinessWebDetector:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.eye_glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self.lock = Lock()
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.state = self._default_state()

    def _default_state(self) -> dict[str, object]:
        return {
            "running": False,
            "score": 0,
            "closed_frames": 0,
            "status": "Idle",
            "alarm": False,
            "fatigue": "Standby",
            "left_eye": "Not detected",
            "right_eye": "Not detected",
            "eyes_detected": 0,
            "face_detected": False,
        }

    def start(self) -> tuple[bool, str]:
        with self.lock:
            if self.running and self.cap and self.cap.isOpened():
                return True, "Camera is already running."

            self._release_camera()
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                self.cap = None
                self.running = False
                self.state.update(self._default_state())
                self.state["status"] = "Camera unavailable"
                return False, "Could not access the webcam."

            self.running = True
            self.state.update(self._default_state())
            self.state["running"] = True
            self.state["status"] = "Scanning"
            return True, "Camera started successfully."

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self._release_camera()
            self.state.update(self._default_state())
            self.state["status"] = "Stopped"

    def _release_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_state(self) -> dict[str, object]:
        with self.lock:
            return dict(self.state)

    def _predict_eye(self, eye_frame: np.ndarray) -> str:
        eye_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        eye_resized = cv2.resize(eye_gray, (24, 24))
        eye_flat = eye_resized.flatten() / 255.0
        prediction = self.model.predict([eye_flat])[0]
        label = str(self.label_encoder.inverse_transform([prediction])[0]).strip().lower()
        return "Closed" if label in {"close", "closed"} else "Open"

    def _fallback_eye_regions(self, face_box: tuple[int, int, int, int]) -> list[dict[str, int]]:
        x, y, w, h = face_box
        top = y + int(h * 0.18)
        eye_h = max(18, int(h * 0.18))
        eye_w = max(24, int(w * 0.22))
        left_x = x + int(w * 0.16)
        right_x = x + int(w * 0.62) - eye_w
        return [
            {"x": left_x, "y": top, "w": eye_w, "h": eye_h},
            {"x": right_x, "y": top, "w": eye_w, "h": eye_h},
        ]

    def _detect_best_eyes(self, gray: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        candidates: list[dict[str, int]] = []
        face_list = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)

        search_regions = []
        if face_list:
            x, y, w, h = face_list[0]
            upper_face = gray[y : y + int(h * 0.6), x : x + w]
            search_regions.append((x, y, upper_face))
        else:
            search_regions.append((0, 0, gray))

        for offset_x, offset_y, region in search_regions:
            for cascade in (self.eye_cascade, self.eye_glasses_cascade):
                detections = cascade.detectMultiScale(region, scaleFactor=1.1, minNeighbors=5)
                for ex, ey, ew, eh in detections:
                    candidates.append(
                        {
                            "x": int(ex + offset_x),
                            "y": int(ey + offset_y),
                            "w": int(ew),
                            "h": int(eh),
                        }
                    )

        candidates.sort(key=lambda item: item["w"] * item["h"], reverse=True)
        filtered: list[dict[str, int]] = []
        for candidate in candidates:
            overlaps = False
            for kept in filtered:
                dx = abs(candidate["x"] - kept["x"])
                dy = abs(candidate["y"] - kept["y"])
                if dx < min(candidate["w"], kept["w"]) * 0.6 and dy < min(candidate["h"], kept["h"]) * 0.6:
                    overlaps = True
                    break
            if not overlaps:
                filtered.append(candidate)
            if len(filtered) == 2:
                break

        filtered.sort(key=lambda item: item["x"])
        if face_list and len(filtered) < 2:
            fallback_regions = self._fallback_eye_regions(tuple(int(v) for v in face_list[0]))
            for fallback in fallback_regions:
                overlaps = False
                for kept in filtered:
                    dx = abs(fallback["x"] - kept["x"])
                    dy = abs(fallback["y"] - kept["y"])
                    if dx < max(fallback["w"], kept["w"]) * 0.7 and dy < max(fallback["h"], kept["h"]) * 0.7:
                        overlaps = True
                        break
                if not overlaps:
                    filtered.append(fallback)

        filtered = filtered[:2]
        filtered.sort(key=lambda item: item["x"])
        return faces, filtered

    def analyze_frame(self, frame: np.ndarray, score: int, closed_frames: int) -> tuple[np.ndarray, dict[str, object]]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, eyes = self._detect_best_eyes(gray)

        face_detected = len(faces) > 0
        if face_detected:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 191, 0), 2)

        eye_results: list[dict[str, object]] = []
        for item in eyes:
            ex, ey, ew, eh = item["x"], item["y"], item["w"], item["h"]
            eye = frame[ey : ey + eh, ex : ex + ew]
            if eye.size == 0:
                continue
            try:
                label = self._predict_eye(eye)
            except cv2.error:
                continue
            eye_results.append({"x": ex, "y": ey, "w": ew, "h": eh, "label": label})

        eye_results.sort(key=lambda item: item["x"])
        left_eye = eye_results[0]["label"] if len(eye_results) >= 1 else "Not detected"
        right_eye = eye_results[1]["label"] if len(eye_results) >= 2 else "Not detected"
        closed_count = sum(1 for item in eye_results if item["label"] == "Closed")
        both_eyes_closed = len(eye_results) == 2 and closed_count == 2
        both_eyes_open = len(eye_results) == 2 and closed_count == 0

        if both_eyes_closed:
            closed_frames += 1
            score = min(100, score + 3)
        elif closed_count == 1:
            closed_frames = max(0, closed_frames - 1)
            score = min(100, score + 1)
        elif both_eyes_open:
            closed_frames = 0
            score = max(0, score - 4)
        else:
            closed_frames = max(0, closed_frames - 1)
            score = max(0, score - 2)

        if both_eyes_closed and closed_frames >= 3:
            status = "Eyes closed"
            fatigue = "High"
            alarm = True
            accent = (0, 76, 255)
        elif closed_frames >= 2 or score >= 8:
            status = "Drowsy warning"
            fatigue = "Medium"
            alarm = False
            accent = (0, 185, 255)
        elif both_eyes_open:
            status = "Eyes open"
            fatigue = "Low"
            alarm = False
            accent = (0, 200, 120)
        elif closed_count == 1:
            status = "One eye closed"
            fatigue = "Low"
            alarm = False
            accent = (0, 185, 255)
        elif len(eye_results) == 0:
            status = "No eyes detected"
            fatigue = "Unknown"
            alarm = False
            accent = (140, 170, 210)
        else:
            status = "Scanning"
            fatigue = "Low"
            alarm = False
            accent = (200, 220, 255)

        for index, item in enumerate(eye_results[:2]):
            eye_name = "Left Eye" if index == 0 else "Right Eye"
            color = (0, 76, 255) if item["label"] == "Closed" else (0, 200, 120)
            cv2.rectangle(
                frame,
                (item["x"], item["y"]),
                (item["x"] + item["w"], item["y"] + item["h"]),
                color,
                2,
            )
            cv2.putText(
                frame,
                f"{eye_name}: {item['label']}",
                (item["x"], max(20, item["y"] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(frame, (0, height - 104), (width, height), (6, 10, 18), thickness=cv2.FILLED)
        cv2.putText(
            frame,
            f"Status: {status}",
            (18, height - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            accent,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Fatigue: {fatigue}  Score: {score}",
            (18, height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.74,
            (240, 244, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"L: {left_eye}  R: {right_eye}",
            (18, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (190, 224, 255),
            2,
            cv2.LINE_AA,
        )

        return frame, {
            "score": score,
            "closed_frames": closed_frames,
            "status": status,
            "fatigue": fatigue,
            "alarm": alarm,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "eyes_detected": len(eye_results),
            "face_detected": face_detected,
        }

    def generate_frames(self):
        while True:
            with self.lock:
                if not self.running or self.cap is None:
                    break
                cap = self.cap
                current_score = int(self.state["score"])
                current_closed_frames = int(self.state["closed_frames"])

            ok, frame = cap.read()
            if not ok:
                with self.lock:
                    self.state.update(self._default_state())
                    self.state["status"] = "Frame read failed"
                    self._release_camera()
                break

            with self.lock:
                if not self.running:
                    break

            annotated, result = self.analyze_frame(frame, current_score, current_closed_frames)

            with self.lock:
                self.state.update(result)
                self.state["running"] = True

            success, encoded = cv2.imencode(".jpg", annotated)
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            )


app = Flask(__name__)
detector = DrowsinessWebDetector()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/about")
def about():
    return render_template("about.html")


@app.post("/start")
def start_camera():
    ok, message = detector.start()
    return jsonify({"ok": ok, "message": message, **detector.get_state()}), (200 if ok else 503)


@app.post("/stop")
def stop_camera():
    detector.stop()
    return jsonify({"ok": True, "message": "Camera stopped.", **detector.get_state()})


@app.get("/status")
def status():
    return jsonify(detector.get_state())


@app.get("/video_feed")
def video_feed():
    if not detector.get_state()["running"]:
        return jsonify({"ok": False, "message": "Camera is not running."}), 409
    return Response(detector.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
