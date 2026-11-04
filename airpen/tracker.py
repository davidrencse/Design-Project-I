"""Camera capture (threaded, newest-frame) and MediaPipe hand landmarking."""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Hand:
    label: str          # "Left" / "Right" from the user's point of view (frame is mirrored)
    score: float
    pts: np.ndarray     # (21,3) pixels; z is MediaPipe's relative depth scaled to pixels
    norm: np.ndarray    # (21,2) normalized image coordinates in [0,1]
    world: np.ndarray | None = None  # (21,3) metres, true 3D hand shape (no perspective squash)


class Camera:
    """Reads frames on a background thread and keeps only the newest one.

    OpenCV's capture buffer otherwise queues stale frames, which adds latency you can feel.
    """

    def __init__(self, index: int | str = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 lock_fps: bool = True):
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        if lock_fps:
            self._lock_framerate(index)
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index!r}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for 720p30 on most UVC cams
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {index!r} opened but returned no frame")
        self.size = (frame.shape[1], frame.shape[0])
        self._frame, self._seq = frame, 1
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    @staticmethod
    def _lock_framerate(index: int | str) -> None:
        """Many UVC webcams halve their frame rate in room light (exposure_dynamic_framerate).
        Hand tracking cares more about frame rate than brightness, so turn that off. The
        setting lasts until the camera is replugged."""
        if not sys.platform.startswith("linux") or not shutil.which("v4l2-ctl"):
            return
        dev = f"/dev/video{index}" if isinstance(index, int) else str(index)
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", "exposure_dynamic_framerate=0"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self, last_seq: int, timeout: float = 1.0) -> tuple[int, np.ndarray]:
        """Block until a frame newer than last_seq exists, then return (seq, frame)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq, timeout):
                raise RuntimeError("camera stopped delivering frames")
            return self._seq, self._frame

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class HandTracker:
    def __init__(self, model_path: str | Path, num_hands: int = 2,
                 det_conf: float = 0.5, presence_conf: float = 0.5, track_conf: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision  # heavy import; keep it local

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} missing. Run ./run.sh once, or download hand_landmarker.task from "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
        import mediapipe as mp

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=track_conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1
        self._last_wrist: dict[str, tuple[float, np.ndarray]] = {}  # label -> (t, wrist xy norm)
        self._gamma_lut = np.array([((i / 255.0) ** 0.55) * 255 for i in range(256)], np.uint8)
        self.brightness = 128.0

    def detect(self, frame_bgr: np.ndarray, t: float) -> list[Hand]:
        """frame_bgr must already be mirrored (selfie view); MediaPipe's handedness assumes that."""
        h, w = frame_bgr.shape[:2]
        # dim room: brighten what the model sees (the palm detector struggles in the dark)
        self.brightness = 0.9 * self.brightness + 0.1 * float(frame_bgr[::8, ::8].mean())
        if self.brightness < 85:
            frame_bgr = cv2.LUT(frame_bgr, self._gamma_lut)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(t * 1000), self._last_ts + 1)  # VIDEO mode requires strictly increasing timestamps
        self._last_ts = ts
        res = self._lm.detect_for_video(image, ts)

        hands: list[Hand] = []
        worlds = res.hand_world_landmarks or [None] * len(res.hand_landmarks)
        for lms, hd, wl in zip(res.hand_landmarks, res.handedness, worlds):
            arr = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float64)
            world = np.array([(p.x, p.y, p.z) for p in wl], dtype=np.float64) if wl else None
            pts = np.empty_like(arr)
            pts[:, 0] = arr[:, 0] * w
            pts[:, 1] = arr[:, 1] * h
            pts[:, 2] = arr[:, 2] * w  # MediaPipe z uses roughly the same scale as x
            hands.append(Hand(hd[0].category_name, float(hd[0].score), pts, arr[:, :2].copy(), world))

        # Two hands with the same label happens occasionally; in the mirrored view the
        # user's right hand is on the right side of the image.
        if len(hands) == 2 and hands[0].label == hands[1].label:
            a, b = sorted(hands, key=lambda hh: hh.pts[0, 0])
            a.label, b.label = "Left", "Right"
        self._stabilize_labels(hands, t)
        return hands

    def _stabilize_labels(self, hands: list[Hand], t: float, recent: float = 0.4) -> None:
        """MediaPipe sometimes flips a single hand's Left/Right label for a frame or two, which
        would make the gesture engine think the hand vanished. If the hand sits exactly where
        the *other* label was a moment ago, and nowhere near its own label's last spot, swap."""
        if len(hands) == 1:
            h = hands[0]
            other = "Left" if h.label == "Right" else "Right"
            wrist = h.norm[0]
            prev_other = self._last_wrist.get(other)
            prev_same = self._last_wrist.get(h.label)
            near_other = prev_other is not None and t - prev_other[0] < recent \
                and np.linalg.norm(wrist - prev_other[1]) < 0.08
            same_fresh = prev_same is not None and t - prev_same[0] < recent \
                and np.linalg.norm(wrist - prev_same[1]) < 0.15
            if near_other and not same_fresh:
                h.label = other
        for h in hands:
            self._last_wrist[h.label] = (t, h.norm[0].copy())

    def close(self) -> None:
        self._lm.close()
