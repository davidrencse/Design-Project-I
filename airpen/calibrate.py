"""Calibration wizard: measures YOUR hand doing each gesture and fits the thresholds to it.

    ./run.sh calibrate

Each step shows an instruction, counts down, then records for a few seconds. Thresholds are
placed halfway between the measured "doing it" and "not doing it" clusters, which is far more
reliable than one-size-fits-all defaults. Results go to ~/.config/airpen/tuning.json and are
loaded automatically on every start. Delete that file to go back to the defaults.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict

import cv2
import numpy as np

from airpen import ui
from airpen.fx import CameraOverlay
from airpen.gestures import Tuning, compute_metrics
from airpen.main import ROOT
from airpen.tracker import Camera, HandTracker
from airpen.tuning_io import CONFIG, SAMPLES

STEPS = (
    ("open", "Hold your hand up, fingers spread wide"),
    ("fist", "Make a tight fist"),
    ("tap_index", "Tap THUMB + INDEX together, again and again (other fingers relaxed and straight)"),
    ("tap_middle", "Tap THUMB + MIDDLE together, again and again"),
    ("point", "Point your index finger UP and hold still"),
    ("rest", "Relax your hand where moving the cursor feels comfortable"),
)
READY_S, RECORD_S = 2.0, 4.0
FIELDS = ("tap_index", "tap_middle", "tap_ring", "tap_pinky", "thumb_mcp", "reach",
          "index_reach", "middle_reach", "scale_norm", "distance_cm", "thumb_side", "ring_reach")


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def ema(xs, a: float) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    out = np.empty_like(xs)
    for i, x in enumerate(xs):
        out[i] = x if i == 0 else a * x + (1 - a) * out[i - 1]
    return out


def split2(xs) -> tuple[float, float] | None:
    """1-D two-cluster split (Otsu). Returns (top of low cluster, bottom of high cluster)."""
    xs = np.sort(np.asarray(xs, dtype=float))
    if len(xs) < 10:
        return None
    best, k_best = -1.0, None
    for k in range(3, len(xs) - 3):
        lo, hi = xs[:k], xs[k:]
        score = len(lo) * len(hi) * (hi.mean() - lo.mean()) ** 2
        if score > best:
            best, k_best = score, k
    lo, hi = xs[:k_best], xs[k_best:]
    return float(np.percentile(lo, 90)), float(np.percentile(hi, 10))


def fit(samples: dict[str, list[dict]], tune: Tuning) -> tuple[dict, list[str]]:
    """Turn recorded samples into threshold overrides. Returns (overrides, notes)."""
    out, notes = {}, []

    def col(step, key):
        return np.array([s[key] for s in samples.get(step, [])], dtype=float)

    def bend(step, finger):
        return np.array([s["bends"][finger] for s in samples.get(step, [])], dtype=float)

    def between(name, low, high, frac=0.5, lo=None, hi=None):
        if np.isnan(low) or np.isnan(high) or high <= low:
            notes.append(f"{name}: clusters overlap ({low:.2f} vs {high:.2f}), kept default")
            return None
        v = low + frac * (high - low)
        v = max(lo, v) if lo is not None else v
        v = min(hi, v) if hi is not None else v
        return round(float(v), 3)

    for step, _ in STEPS:
        n = len(samples.get(step, []))
        if n < 15:
            notes.append(f"{step}: only {n} frames with one clear hand")

    # straight vs curled fingers: straight = all fingers while open, ring+pinky while tapping;
    # curled = all fingers in a fist, middle/ring/pinky while pointing
    straight = np.concatenate([bend("open", f) for f in range(4)]
                              + [bend("tap_index", f) for f in (1, 2, 3)]
                              + [bend("tap_middle", f) for f in (2,)]
                              + [bend("point", 0)])
    curled = np.concatenate([bend("fist", f) for f in range(4)]
                            + [bend("point", f) for f in (1, 2, 3)])
    v = between("extend_bend", _pct(straight, 85), _pct(curled, 5), lo=90, hi=200)
    if v: out["extend_bend"] = v

    v = between("fist_bend", _pct(bend("rest", 0), 90), _pct(bend("fist", 0), 10), lo=140, hi=230)
    if v: out["fist_bend"] = v

    v = between("point_bend", _pct(bend("point", 0), 90), _pct(bend("rest", 0), 10), lo=60, hi=130)
    if v: out["point_bend"] = v

    # taps: fit on the same smoothed signal the live engine sees. "touch" = the highest the
    # distance gets while touching; "release" = the lowest peak between two touches (quick
    # taps never fully open, so the release threshold must sit under the smallest peak).
    for step, key, on_k, off_k in (("tap_index", "tap_index", "tap_on", "tap_off"),
                                    ("tap_middle", "tap_middle", "right_on", "right_off")):
        xs = ema(col(step, key), tune.smooth)
        sp = split2(xs)
        if sp is None:
            notes.append(f"{on_k}: not enough frames, kept default")
            continue
        mid = (sp[0] + sp[1]) / 2
        lows = xs[xs < mid]
        peaks = [xs[i] for i in range(1, len(xs) - 1)
                 if xs[i] >= mid and xs[i] >= xs[i - 1] and xs[i] >= xs[i + 1]]
        if len(lows) < 5 or len(peaks) < 3:
            notes.append(f"{on_k}: too few taps recorded, kept default")
            continue
        touch, release = float(np.median(lows)), float(np.percentile(peaks, 10))
        if release <= touch:
            notes.append(f"{on_k}: touching and released overlap ({touch:.2f} vs {release:.2f}), kept default")
            continue
        out[on_k] = round(touch + 0.40 * (release - touch), 3)
        out[off_k] = round(touch + 0.70 * (release - touch), 3)

    fist_gap = _pct(col("fist", "tap_index"), 5)
    if not np.isnan(fist_gap):
        out["fist_gap"] = round(float(min(fist_gap * 0.85, out.get("tap_off", tune.tap_off))), 3)

    rest = col("rest", "scale_norm")
    if len(rest) >= 15:
        out["pointer_ref_scale"] = round(float(np.median(rest)), 4)
    return out, notes


def report(samples: dict[str, list[dict]], tune: Tuning) -> list[str]:
    """Replay the recorded frames through the classifier: what would each step be read as?"""
    from airpen.gestures import Debounced, Metrics, classify_pose

    lines = []
    for step, rows in samples.items():
        if not rows:
            continue
        poses = {}
        for r in rows:
            ext = tuple(b < tune.extend_bend for b in r["bends"])
            m = Metrics(scale=1, tap_index=r["tap_index"], tap_middle=r["tap_middle"],
                        tap_ring=r["tap_ring"], tap_pinky=r["tap_pinky"], reach=r["reach"],
                        extended=ext, bends=tuple(r["bends"]), palm=np.zeros(2), anchor=np.zeros(2),
                        index_tip=np.zeros(2), thumb_tip=np.zeros(2), thumb_mcp=r["thumb_mcp"],
                        index_dir=np.zeros(2), index_reach=r["index_reach"],
                        middle_reach=r["middle_reach"], scale_norm=r["scale_norm"],
                        distance_cm=r["distance_cm"])
            pz = classify_pose(m, tune)
            poses[pz] = poses.get(pz, 0) + 1
        mix = ", ".join(f"{k} {100 * v // len(rows)}%" for k, v in sorted(poses.items(), key=lambda kv: -kv[1]))
        line = f"{step:11s} read as: {mix}"
        for key, on, off, name in (("tap_index", tune.tap_on, tune.tap_off, "left taps"),
                                   ("tap_middle", tune.right_on, tune.right_off, "right taps")):
            d = Debounced(on, off, tune.confirm_frames, tune.release_frames)
            xs = ema([r[key] for r in rows], tune.smooth)
            ring = [r["bends"][2] < tune.extend_bend for r in rows]
            # like the live engine: a tap can only START while the ring finger is straight
            n = sum(1 for x, ok in zip(xs, ring) if d.update(x if (ok or d.on) else float("inf")) is True)
            if n:
                line += f" | {name} {n}"
        lines.append(line)
    return lines


def save(samples: dict, tune: Tuning) -> None:
    overrides, notes = fit(samples, tune)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(overrides, indent=2))
    SAMPLES.write_text(json.dumps(samples))
    print(f"saved {len(overrides)} personal thresholds to {CONFIG}")
    defaults = asdict(Tuning())
    for k, v in overrides.items():
        print(f"  {k:18s} {defaults.get(k, '-')!s:>8} -> {v}")
    for n in notes:
        print(f"  note: {n}")
    fitted = Tuning()
    for k, v in overrides.items():
        if hasattr(fitted, k):
            setattr(fitted, k, v)
    print("replay of your recording with these thresholds:")
    for ln in report(samples, fitted):
        print("  " + ln)


def main() -> int:
    if "--refit" in sys.argv:
        save(json.loads(SAMPLES.read_text()), Tuning())
        return 0
    tune = Tuning()
    cam = Camera(0, 640, 480)
    tracker = HandTracker(ROOT / "models" / "hand_landmarker.task")
    win = "AirPen calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, 1280, 960)
    samples: dict[str, list[dict]] = {}
    seq = 0
    overlay = CameraOverlay()
    try:
        for idx, (step, text) in enumerate(STEPS):
            samples[step] = []
            start = time.monotonic()
            while True:
                seq, frame = cam.read(seq)
                frame = cv2.flip(frame, 1)
                t = time.monotonic()
                hands = tracker.detect(frame, t)
                el = t - start
                recording = READY_S <= el < READY_S + RECORD_S
                if recording and len(hands) == 1:
                    h = hands[0]
                    m = compute_metrics(h.pts, h.norm, tune, h.world)
                    row = {k: float(getattr(m, k)) for k in FIELDS}
                    row["bends"] = [float(b) for b in m.bends]
                    samples[step].append(row)
                one = len(hands) == 1
                if el < READY_S:
                    status, ok, frac = f"GET READY  {READY_S - el:.1f} S", False, None
                else:
                    status = f"RECORDING  {len(samples[step])} FRAMES"
                    ok = one
                    frac = min(1.0, (el - READY_S) / RECORD_S)
                hint = ("SPACE SKIP  /  ESC CANCEL" if one or el < READY_S
                        else "SHOW EXACTLY ONE HAND  /  SPACE SKIP  /  ESC CANCEL")
                image = overlay.render_safe(frame, hands)
                img = ui.camera_guide(image, [f"{idx + 1} / {len(STEPS)}  {step.replace('_', ' ')}",
                                              text, status, hint], frac)
                cv2.imshow(win, img)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or ui.window_closed(win):
                    print("calibration cancelled; nothing saved")
                    return 1
                if key == ord(" ") or el >= READY_S + RECORD_S:
                    break
    finally:
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()
    save(samples, tune)
    return 0


if __name__ == "__main__":
    sys.exit(main())
