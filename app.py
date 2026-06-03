from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock, Thread

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "DDDS_CNN" / "models" / "cnnCat2.joblib"
LABEL_ENCODER_PATH = BASE_DIR / "DDDS_CNN" / "models" / "label_encoder.joblib"


class DrowsinessWebDetector:
    MODE_SETTINGS = {
        "normal": {
            "one_eye_score": 2,
            "both_eye_score": 4,
            "alarm_closed_frames": 2,
            "alarm_open_frames": 2,
        },
        "strict": {
            "one_eye_score": 3,
            "both_eye_score": 5,
            "alarm_closed_frames": 1,
            "alarm_open_frames": 2,
        },
        "very_sensitive": {
            "one_eye_score": 4,
            "both_eye_score": 6,
            "alarm_closed_frames": 1,
            "alarm_open_frames": 1,
        },
    }

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
        self.frame_lock = Lock()
        self.cap: cv2.VideoCapture | None = None
        self.capture_thread: Thread | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_id = 0
        self.last_annotated_frame: np.ndarray | None = None
        self.running = False
        self.state = self._default_state()

    def _default_state(self) -> dict[str, object]:
        return {
            "running": False,
            "score": 0,
            "closed_frames": 0,
            "open_frames": 0,
            "status": "Idle",
            "alarm": False,
            "fatigue": "Standby",
            "left_eye": "Not detected",
            "right_eye": "Not detected",
            "left_confidence": 0.0,
            "right_confidence": 0.0,
            "eyes_detected": 0,
            "face_detected": False,
            "mode": "normal",
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

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            self.running = True
            self.state.update(self._default_state())
            self.state["running"] = True
            self.state["status"] = "Scanning"
            self.latest_frame = None
            self.latest_frame_id = 0
            self.last_annotated_frame = None
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            return True, "Camera started successfully."

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self._release_camera()
            self.state.update(self._default_state())
            self.state["status"] = "Stopped"

    def _release_camera(self) -> None:
        with self.frame_lock:
            self.latest_frame = None
            self.latest_frame_id = 0
            self.last_annotated_frame = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.capture_thread = None

    def get_state(self) -> dict[str, object]:
        with self.lock:
            return dict(self.state)

    def set_mode(self, mode: str) -> tuple[bool, str]:
        normalized = mode.strip().lower().replace(" ", "_")
        if normalized not in self.MODE_SETTINGS:
            return False, "Invalid mode."
        with self.lock:
            self.state["mode"] = normalized
        return True, f"Mode set to {normalized.replace('_', ' ')}."

    def _capture_loop(self) -> None:
        while True:
            with self.lock:
                if not self.running or self.cap is None:
                    break
                cap = self.cap

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            with self.frame_lock:
                self.latest_frame = frame
                self.latest_frame_id += 1

    def _prepare_eye(self, eye_frame: np.ndarray) -> np.ndarray:
        eye_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        eye_gray = cv2.equalizeHist(eye_gray)
        eye_resized = cv2.resize(eye_gray, (24, 24))
        return (eye_resized.flatten() / 255.0).astype("float32")

    def _predict_eye(self, eye_frame: np.ndarray) -> tuple[str, float]:
        eye_flat = self._prepare_eye(eye_frame)
        prediction = int(self.model.predict([eye_flat])[0])

        confidence = 0.5
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([eye_flat])[0]
            confidence = float(np.max(proba))

        label = str(self.label_encoder.inverse_transform([prediction])[0]).strip().lower()
        return ("Closed" if label in {"close", "closed"} else "Open"), confidence

    def _fallback_eye_regions(self, face_box: tuple[int, int, int, int]) -> dict[str, dict[str, int]]:
        x, y, w, h = face_box
        top = y + int(h * 0.20)
        eye_h = max(20, int(h * 0.18))
        eye_w = max(28, int(w * 0.24))
        left_x = x + int(w * 0.13)
        right_x = x + int(w * 0.63) - eye_w
        return {
            "left": {"x": left_x, "y": top, "w": eye_w, "h": eye_h},
            "right": {"x": right_x, "y": top, "w": eye_w, "h": eye_h},
        }

    def _extract_eye_regions(
        self, frame: np.ndarray, gray: np.ndarray
    ) -> tuple[tuple[int, int, int, int] | None, dict[str, dict[str, int]]]:
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) == 0:
            return None, {}

        face_box = tuple(int(v) for v in max(faces, key=lambda box: box[2] * box[3]))
        x, y, w, h = face_box
        upper_face = gray[y : y + int(h * 0.60), x : x + w]

        regions = self._fallback_eye_regions(face_box)
        detections: list[dict[str, int]] = []
        for cascade in (self.eye_cascade, self.eye_glasses_cascade):
            found = cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=5)
            for ex, ey, ew, eh in found:
                detections.append(
                    {
                        "x": int(ex + x),
                        "y": int(ey + y),
                        "w": int(ew),
                        "h": int(eh),
                    }
                )

        left_boundary = x + w // 2
        for item in sorted(detections, key=lambda det: det["w"] * det["h"], reverse=True):
            center_x = item["x"] + item["w"] // 2
            side = "left" if center_x < left_boundary else "right"
            current = regions[side]
            current_area = current["w"] * current["h"]
            item_area = item["w"] * item["h"]
            if item_area >= current_area * 0.50:
                pad_x = int(item["w"] * 0.10)
                pad_y = int(item["h"] * 0.18)
                regions[side] = {
                    "x": max(0, item["x"] - pad_x),
                    "y": max(0, item["y"] - pad_y),
                    "w": min(frame.shape[1] - max(0, item["x"] - pad_x), item["w"] + 2 * pad_x),
                    "h": min(frame.shape[0] - max(0, item["y"] - pad_y), item["h"] + 2 * pad_y),
                }

        return face_box, regions

    def analyze_frame(self, frame: np.ndarray, prev_state: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_box, eye_regions = self._extract_eye_regions(frame, gray)

        face_detected = face_box is not None
        if face_detected:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 191, 0), 2)

        eye_results: dict[str, dict[str, object]] = {}
        for side in ("left", "right"):
            region = eye_regions.get(side)
            if region is None:
                continue

            ex, ey, ew, eh = region["x"], region["y"], region["w"], region["h"]
            eye = frame[ey : ey + eh, ex : ex + ew]
            if eye.size == 0:
                continue

            try:
                label, confidence = self._predict_eye(eye)
            except cv2.error:
                continue

            eye_results[side] = {
                "x": ex,
                "y": ey,
                "w": ew,
                "h": eh,
                "label": label,
                "confidence": confidence,
            }

        left_eye = str(eye_results.get("left", {}).get("label", "Not detected"))
        right_eye = str(eye_results.get("right", {}).get("label", "Not detected"))
        left_confidence = float(eye_results.get("left", {}).get("confidence", 0.0))
        right_confidence = float(eye_results.get("right", {}).get("confidence", 0.0))
        eyes_detected = len(eye_results)

        left_closed = left_eye == "Closed"
        right_closed = right_eye == "Closed"
        both_eyes_closed = left_closed and right_closed
        both_eyes_open = left_eye == "Open" and right_eye == "Open"
        one_eye_closed = left_closed ^ right_closed
        any_eye_closed = left_closed or right_closed

        score = int(prev_state["score"])
        closed_frames = int(prev_state["closed_frames"])
        open_frames = int(prev_state["open_frames"])
        alarm_active = bool(prev_state["alarm"])
        mode = str(prev_state.get("mode", "normal"))
        settings = self.MODE_SETTINGS.get(mode, self.MODE_SETTINGS["normal"])

        if both_eyes_closed:
            closed_frames += 1
            open_frames = 0
            score = min(100, score + int(settings["both_eye_score"]))
        elif one_eye_closed:
            closed_frames += 1
            open_frames = 0
            score = min(100, score + int(settings["one_eye_score"]))
        elif both_eyes_open:
            open_frames += 1
            closed_frames = 0
            score = max(0, score - 5)
        else:
            open_frames = 0
            closed_frames = max(0, closed_frames - 1)
            score = max(0, score - 1)

        if any_eye_closed and closed_frames >= int(settings["alarm_closed_frames"]):
            alarm_active = True
        elif both_eyes_open and open_frames >= int(settings["alarm_open_frames"]):
            alarm_active = False

        if alarm_active and both_eyes_closed:
            status = "Eyes closed"
            fatigue = "High"
            accent = (0, 76, 255)
        elif alarm_active and one_eye_closed:
            status = "Eye closed alert"
            fatigue = "Medium"
            accent = (255, 140, 0)
        elif both_eyes_closed or closed_frames >= 1:
            status = "Drowsy warning"
            fatigue = "Medium"
            accent = (0, 185, 255)
        elif both_eyes_open:
            status = "Eyes open"
            fatigue = "Low"
            accent = (0, 200, 120)
        elif one_eye_closed:
            status = "One eye closed"
            fatigue = "Low"
            accent = (0, 185, 255)
        elif eyes_detected == 0:
            status = "No eyes detected"
            fatigue = "Unknown"
            accent = (140, 170, 210)
        else:
            status = "Scanning"
            fatigue = "Low"
            accent = (200, 220, 255)

        for side in ("left", "right"):
            item = eye_results.get(side)
            if item is None:
                continue
            color = (0, 76, 255) if item["label"] == "Closed" else (0, 200, 120)
            eye_name = "Left Eye" if side == "left" else "Right Eye"
            cv2.rectangle(
                frame,
                (item["x"], item["y"]),
                (item["x"] + item["w"], item["y"] + item["h"]),
                color,
                2,
            )
            cv2.putText(
                frame,
                f"{eye_name}: {item['label']} {item['confidence']:.2f}",
                (item["x"], max(20, item["y"] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
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
            "open_frames": open_frames,
            "status": status,
            "fatigue": fatigue,
            "alarm": alarm_active,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "left_confidence": left_confidence,
            "right_confidence": right_confidence,
            "eyes_detected": eyes_detected,
            "face_detected": face_detected,
            "mode": mode,
        }

    def generate_frames(self):
        last_processed_id = -1
        skip_counter = 0
        while True:
            with self.lock:
                if not self.running or self.cap is None:
                    break
                prev_state = dict(self.state)

            with self.frame_lock:
                frame_id = self.latest_frame_id
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None or frame_id == last_processed_id:
                time.sleep(0.01)
                continue

            run_detection = skip_counter % 2 == 0 or self.last_annotated_frame is None
            skip_counter += 1

            if run_detection:
                last_processed_id = frame_id
                annotated, result = self.analyze_frame(frame, prev_state)

                with self.lock:
                    self.state.update(result)
                    self.state["running"] = True

                with self.frame_lock:
                    self.last_annotated_frame = annotated.copy()
            else:
                with self.frame_lock:
                    annotated = (
                        self.last_annotated_frame.copy()
                        if self.last_annotated_frame is not None
                        else frame
                    )

            success, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 68],
            )
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


@app.post("/mode/<mode>")
def set_mode(mode: str):
    ok, message = detector.set_mode(mode)
    return jsonify({"ok": ok, "message": message, **detector.get_state()}), (200 if ok else 400)


@app.get("/status")
def status():
    return jsonify(detector.get_state())


@app.get("/video_feed")
def video_feed():
    if not detector.get_state()["running"]:
        return jsonify({"ok": False, "message": "Camera is not running."}), 409
    return Response(detector.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(
        debug=False,
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
    )
