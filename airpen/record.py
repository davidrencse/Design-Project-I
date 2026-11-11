"""Guided capture of RAW hand landmarks, for checking the action recognizer offline.

    ./run.sh record                        # dominant hand from settings
    .venv/bin/python -m airpen.record --hand left --reps 5

Steps run in order (neutral, open, index_tap, middle_tap, fist, thumb_swipe_right,
thumb_swipe_left, thumb_tap, peace). Each step shows an instruction, counts down, then records
for a few seconds. Only the dominant hand is kept. Output:

    <out>/<YYYYMMDD-HHMMSS>/<step>.npz   t, present, pts, norm, world (NaN where absent)
    <out>/<YYYYMMDD-HHMMSS>/meta.json    hand, frame_size, fps, reps, steps, expected

Replay with `./run.sh replay` (see airpen/replay.py). q or Esc aborts; completed steps are kept.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from airpen import settings, ui
from airpen.fx import CameraOverlay
from airpen.tracker import Camera, HandTracker

ROOT = Path(__file__).resolve().parent.parent
WINDOW = "AirPen record"

ACTION_NAMES = {1: "index_tap", 2: "middle_tap", 3: "fist", 4: "thumb_swipe_right",
                5: "thumb_swipe_left", 6: "peace", 7: "thumb_tap"}

STEPS = ("neutral", "open", "index_tap", "middle_tap", "fist",
         "thumb_swipe_right", "thumb_swipe_left", "thumb_tap", "peace")

TEXT = {
    "neutral": "Relax your hand, move it around naturally. NO gestures.",
    "open": "Open palm, fingers spread. Hold still.",
    "index_tap": "Tap THUMB + INDEX together, again and again. Other fingers straight.",
    "middle_tap": "Tap THUMB + MIDDLE together, again and again.",
    "fist": "Make a fist, then open the hand. Repeat.",
    "thumb_swipe_right": "Make a fist, swipe the THUMB to YOUR right. Repeat.",
    "thumb_swipe_left": "Make a fist, swipe the THUMB to YOUR left. Repeat.",
    "thumb_tap": "Make a fist, tap the THUMB inside the fist. Repeat.",
    "peace": "Make a peace sign (index + middle up), then relax. Repeat.",
}

N_PTS = 21


class Abort(Exception):
    """User pressed q / Esc or closed the window."""


def expected_counts(step: str, reps: int) -> dict[int, int]:
    """Action counts a perfect recognizer should report for one recorded step."""
    table = {
        "neutral": {}, "open": {},
        "index_tap": {1: reps},
        "middle_tap": {2: reps},
        "fist": {3: reps},
        "thumb_swipe_right": {4: reps, 3: 1},
        "thumb_swipe_left": {5: reps, 3: 1},
        "thumb_tap": {7: reps, 3: 1},
        "peace": {6: reps},
    }
    return dict(table[step])


def build_meta(hand: str, frame_size, fps: float, reps: int, steps, tuning: dict | None = None) -> dict:
    meta = {
        "hand": hand,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "fps": round(float(fps), 2),
        "reps": int(reps),
        "steps": list(steps),
        # JSON keys are strings: {step: {"<action_id>": count}}
        "expected": {s: {str(k): v for k, v in expected_counts(s, reps).items()} for s in steps},
    }
    if tuning is not None:
        meta["tuning"] = tuning     # effective thresholds at record time (for reference; replay uses the current ones)
    return meta


def save_step(capture_dir: Path, step: str, *, t, present, pts, norm, world) -> Path:
    """Write one step in the format replay reads. Missing frames: present False, NaN coords."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n = len(t)
    present = np.asarray(present, dtype=bool).reshape(n)
    pts = np.asarray(pts, dtype=np.float64).reshape(n, N_PTS, 3)
    norm = np.asarray(norm, dtype=np.float64).reshape(n, N_PTS, 2)
    world = np.asarray(world, dtype=np.float64).reshape(n, N_PTS, 3)
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / f"{step}.npz"
    np.savez(path, t=t, present=present, pts=pts, norm=norm, world=world)
    return path


def write_meta(capture_dir: Path, meta: dict) -> Path:
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def dominant(hands, hand: str):
    for h in hands:
        if h.label == hand:
            return h
    return None


def _draw(overlay, frame, hands, hand, head: tuple[str, str, str], sub: str,
          frac: float | None):
    """Color camera with hand landmarks and the instructions needed for recording."""
    step, name, text = head
    image = overlay.render_safe(frame, hands, hand=hand)
    return ui.camera_guide(image, [f"{step}  {name}", text, sub, "Q / ESC: abort"], frac)


def record(args) -> int:
    hand = settings.norm_hand(args.hand) if args.hand else settings.load_hand()
    cam_index = int(args.camera) if str(args.camera).isdigit() else str(args.camera)
    out_root = Path(args.out) if args.out else ROOT / "captures"
    cap_dir = out_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    model = args.model or str(ROOT / "models" / "hand_landmarker.task")

    cam = tracker = None
    done: list[str] = []
    n_frames, rec_secs = 0, 0.0
    rc = 0
    try:
        cam = Camera(cam_index, args.width, args.height, args.fps)
        tracker = HandTracker(model)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, 1280, 960)
        seq = 0
        overlay = CameraOverlay()
        for idx, step in enumerate(STEPS):
            start = time.monotonic()
            rec_t0 = start + args.ready_s
            end = rec_t0 + args.record_s
            ts, present, pts, norm, world = [], [], [], [], []
            while True:
                seq, frame = cam.read(seq)
                frame = cv2.flip(frame, 1)  # same selfie view as main.py
                t = time.monotonic()
                hands = tracker.detect(frame, t)
                h = dominant(hands, hand)
                if rec_t0 <= t < end:
                    ts.append(t - rec_t0)
                    present.append(h is not None)
                    if h is not None:
                        pts.append(h.pts)
                        norm.append(h.norm)
                        world.append(h.world if h.world is not None else np.full((N_PTS, 3), np.nan))
                    else:
                        pts.append(np.full((N_PTS, 3), np.nan))
                        norm.append(np.full((N_PTS, 2), np.nan))
                        world.append(np.full((N_PTS, 3), np.nan))
                el = t - start
                head = (f"{idx + 1:02d} / {len(STEPS):02d}", step.upper().replace("_", " "), TEXT[step])
                if el < args.ready_s:
                    img = _draw(overlay, frame, hands, hand, head, f"GET READY  {args.ready_s - el:.1f} S", None)
                elif t < end:
                    frac = (t - rec_t0) / args.record_s
                    sub = f"RECORDING  {len(ts)} FRAMES" + ("" if h else f"  /  NO {hand.upper()} HAND IN VIEW")
                    img = _draw(overlay, frame, hands, hand, head, sub, frac)
                else:
                    break
                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or ui.window_closed(WINDOW):
                    raise Abort()
            save_step(cap_dir, step,
                      t=ts, present=present,
                      pts=np.asarray(pts).reshape(-1, N_PTS, 3),
                      norm=np.asarray(norm).reshape(-1, N_PTS, 2),
                      world=np.asarray(world).reshape(-1, N_PTS, 3))
            done.append(step)
            n_frames += len(ts)
            rec_secs += args.record_s
            print(f"saved {step}: {sum(present)}/{len(ts)} frames with a {hand} hand")
        print(f"all {len(STEPS)} steps recorded -> {cap_dir}")
    except Abort:
        print(f"aborted. {len(done)} completed step(s) kept in {cap_dir}" if done else "aborted; nothing saved")
        rc = 130
    except RuntimeError as exc:  # camera or model problems
        print(f"record failed: {exc}", file=sys.stderr)
        rc = 1
    finally:
        if done:
            fps = n_frames / rec_secs if rec_secs else 0.0
            frame_size = cam.size if cam else (args.width, args.height)
            from airpen.actions import effective_tuning
            write_meta(cap_dir, build_meta(hand, frame_size, fps, args.reps, done, effective_tuning()))
        if cam is not None:
            cam.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="airpen.record", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hand", choices=("left", "right", "Left", "Right"),
                   help="dominant hand to record (default: saved setting)")
    p.add_argument("--camera", default="0", help="camera index or device path (default 0)")
    p.add_argument("--out", help="output folder (default: captures/ in the project)")
    p.add_argument("--reps", type=int, default=5, help="repetitions per gesture step (default 5)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--ready-s", type=float, default=2.0, help="countdown before each step (s)")
    p.add_argument("--record-s", type=float, default=8.0, help="recording length per step (s)")
    p.add_argument("--model", help="hand landmarker model (default: models/hand_landmarker.task)")
    args = p.parse_args(argv)
    if args.reps < 1:
        p.error("--reps must be at least 1")
    return record(args)


if __name__ == "__main__":
    sys.exit(main())
