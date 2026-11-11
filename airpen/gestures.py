"""Gesture recognition on MediaPipe hand landmarks.

Input: per-frame list of hands (21 landmarks each, pixel coordinates, mirrored "selfie" view).
Output: a GestureFrame with discrete events (clicks, swipes, mode changes) and continuous
signals (pointer anchor, scroll and zoom deltas, freeze flags for the pointer).

Gesture map
-----------
  PEN GRIP (hold a pen/stylus)                    pen mode: cursor on while you hold it
    tap the pen with your index finger            left click
    tap the pen with your middle finger           right click
  FIST held ~0.8 s                                toggle mouse mode (cursor on/off)
  mouse mode:
    hand moves                                    move cursor (knuckle anchor, steady while tapping)
    tap thumb + index                             left button (hold = drag)
    tap thumb + middle                            right button
  any mode:
    POINT index up / down, held still             scroll up / down (speeds up while held)
    both hands POINT, spread / close              zoom in / out
  gesture mode (cursor off):
    OPEN palm fast swipe left / right             swipe event (workspace switch)
  zoom is two-handed only.

Thresholds come from real calibration data (./run.sh calibrate): hand *shape* is measured on
MediaPipe's metric 3D world landmarks with joint angles, distances are divided by the hand's
own size, so everything holds at any distance and rotation.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGERS = (  # (mcp, pip, dip, tip) for index, middle, ring, pinky
    (INDEX_MCP, INDEX_PIP, 7, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, 11, MIDDLE_TIP),
    (RING_MCP, RING_PIP, 15, RING_TIP),
    (PINKY_MCP, PINKY_PIP, 19, PINKY_TIP),
)
SMOOTHED = ("tap_index", "tap_middle", "tap_ring", "tap_pinky", "thumb_mcp", "reach",
            "index_reach", "middle_reach", "distance_cm", "scale_norm")
PALM_SEGMENTS = ((WRIST, INDEX_MCP), (WRIST, MIDDLE_MCP), (WRIST, PINKY_MCP), (INDEX_MCP, PINKY_MCP))
INF = float("inf")


@dataclass
class Tuning:
    """Defaults measured on a real calibration run (world landmarks, hand-scale units)."""
    # finger shape
    extend_bend: float = 131.23    # total joint bend below this = finger straight (straight ~35-90, curled ~195-270)
    fist_bend: float = 194.83      # index bend above this counts toward a fist (resting hand ~100-160)
    point_bend: float = 81.43      # a POINT needs the index this straight (pointing ~50-62, resting ~100-160)
    # left tap: thumb tip to index tip (touching 0.18-0.30, apart 0.5-1.0)
    tap_on: float = 0.44
    tap_off: float = 0.56
    # right tap: thumb tip to middle tip reads larger (touching 0.42-0.68, apart 0.85-1.2)
    right_on: float = 0.76
    right_off: float = 0.82
    confirm_frames: int = 3       # frames a touch must persist before it counts as a press
    release_frames: int = 1       # quick taps open for a single frame; the hysteresis gap is the guard
    # pen mode: hold a pen/stylus; index tap on the pen = left click, middle tap = right click
    pen_thumb_near: float = 0.45  # thumb tip this close to the index finger = gripping something
    pen_hold: float = 0.30        # grip held this long turns pen mode on
    pen_release: float = 0.70     # grip gone this long turns it off
    tap_dev: float = 0.15         # finger must move this far (hand-scales) from its resting spot
    tap_min: float = 0.05         # a tap lasts at least this long (s)...
    tap_max: float = 0.60         # ...and at most this long; longer = a pose change, not a tap
    tap_ring_still: float = 0.15  # the ring finger must stay put (whole-hand moves are not taps)
    tap_baseline: float = 1.0     # seconds of history that define a finger's resting spot
    # fist toggles mouse mode
    fist_hold: float = 0.8        # seconds
    fist_gap: float = 0.15        # thumb-index must be at least this apart (a pinch is not a fist)
    # ignore hands too close to the camera: landmarks turn to garbage when the hand fills the frame
    pointer_ref_scale: float = 0.11  # calibrated hand size for cursor sensitivity
    max_hand_size: float = 0.40   # hand length / frame width (normal ~0.2 at 35 cm)
    min_distance_cm: float = 15.0
    # smoothing / voting
    z_weight: float = 0.6         # fallback only (no world landmarks)
    camera_hfov: float = 70.0     # webcam horizontal field of view, degrees (distance estimate)
    smooth: float = 0.6           # EMA weight of the newest frame for distance features
    pose_votes: int = 3           # a new pose needs a majority of this many recent frames
    # timing
    lost_timeout: float = 0.25    # hand unseen this long = gone (buttons released)
    cursor_timeout: float = 4.0   # cursor hand gone this long = mouse mode off
    # scroll by pointing
    scroll_rate: float = 600.0    # wheel units per second (120 = one notch)
    scroll_accel: float = 3.0     # speed multiplier reached after scroll_ramp seconds
    scroll_ramp: float = 1.5
    scroll_tilt: float = 0.55     # how vertical the finger must be (0 sideways .. 1 vertical)
    scroll_still: float = 0.25    # seconds the point must be held still before scrolling
    # zoom
    zoom_gain: float = 600.0      # wheel units (120 = one zoom step) per unit of log(spread)
    zoom_deadband: float = 0.03
    zoom_engage: float = 0.22     # one-hand zoom starts once thumb-index changes this much...
    zoom_window: float = 0.40     # ...within this many seconds
    zoom_idle: float = 0.8        # ...and stops after this long without movement
    two_hand_settle: int = 4      # frames both hands must point before two-hand zoom counts
    # swipes
    swipe_window: float = 0.35
    swipe_speed: float = 3.5      # hand-scales per second
    swipe_dist: float = 1.2
    swipe_axis: float = 2.0
    swipe_cooldown: float = 1.0
    swipe_open_frames: int = 4        # palm must already be open this many frames (no accidental swipes)
    swipe_return_block: float = 1.2   # after a swipe, ignore the opposite direction this long (the hand coming back)
    # arming
    arm_hold: float = 1.5
    arm_still: float = 0.8


@dataclass
class Event:
    kind: str                 # see module docstring; e.g. press, release, swipe, cursor_on
    hand: str = ""            # "Left" / "Right"
    button: str = ""          # "left" / "right" for press/release
    dir: str = ""             # swipe direction
    dx: float = 0.0           # scroll deltas, hi-res wheel units
    dy: float = 0.0
    amount: float = 0.0       # zoom delta, hi-res wheel units (+ = zoom in)


@dataclass
class Metrics:
    scale: float
    tap_index: float          # thumb tip to index tip / scale
    tap_middle: float
    tap_ring: float
    tap_pinky: float
    reach: float              # mean of index/middle/ring tip-to-wrist / scale
    extended: tuple[bool, bool, bool, bool]
    bends: tuple[float, float, float, float]   # total joint bend per finger, degrees (0 = straight)
    palm: np.ndarray          # (2,) pixels
    anchor: np.ndarray        # (2,) normalized image coords; knuckle centroid, steady while tapping
    index_tip: np.ndarray     # (2,) pixels
    thumb_tip: np.ndarray     # (2,) pixels
    thumb_mcp: float          # thumb tip to index knuckle / scale (small = thumb tucked)
    index_dir: np.ndarray     # (2,) unit vector index knuckle -> tip (y < 0 = pointing up)
    index_reach: float        # index tip to wrist / scale
    middle_reach: float
    scale_norm: float         # hand size as a fraction of frame width (bigger = closer)
    distance_cm: float        # estimated hand-to-camera distance (0 if unknown)
    thumb_side: float = 9.9   # thumb tip to nearest point on the index finger (pen grip: small)
    ring_reach: float = 0.0   # ring tip to wrist / scale (moves when the whole hand changes pose)
    pen_anchor: np.ndarray | None = None  # (2,) normalized; thumb + index knuckle: steady while fingers tap


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2)


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def compute_metrics(pts: np.ndarray, norm: np.ndarray, tune: Tuning,
                    world: np.ndarray | None = None) -> Metrics:
    """pts: (21,3) pixel coords. norm: (21,2) in [0,1]. world: (21,3) metres, optional.

    Hand *shape* (touch distances, finger bends) is measured on MediaPipe's world landmarks
    when available: true 3D, so fingers lined up with the camera don't look like they touch.
    Hand *motion* (palm, swipes, pointer) stays in image space, where the screen is.
    """
    scale = max(float(np.linalg.norm(pts[WRIST, :2] - pts[MIDDLE_MCP, :2])), 1.0)
    shape = world if world is not None else pts * np.array([1.0, 1.0, tune.z_weight])
    s3 = max(float(np.linalg.norm(shape[WRIST] - shape[MIDDLE_MCP])), 1e-9)

    def d(i: int, j: int) -> float:
        return float(np.linalg.norm(shape[i] - shape[j])) / s3

    # All twelve joint angles in one array operation rather than twelve tiny NumPy
    # dispatches per hand, per frame. Preserve the zero-length segment convention.
    joints = shape[np.asarray(FINGERS)]
    vectors = np.diff(np.concatenate((np.broadcast_to(shape[WRIST], (4, 1, 3)), joints), axis=1), axis=1)
    u, v = vectors[:, :-1], vectors[:, 1:]
    nu, nv = np.linalg.norm(u, axis=2), np.linalg.norm(v, axis=2)
    valid = (nu >= 1e-9) & (nv >= 1e-9)
    cosine = np.ones((4, 3))
    np.divide(np.einsum("ijk,ijk->ij", u, v), nu * nv, out=cosine, where=valid)
    bends = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).sum(axis=1)
    extended = tuple(bool(b < tune.extend_bend) for b in bends)

    # frame width from any landmark (pts = norm * width)
    k = int(np.argmax(norm[:, 0]))
    frame_w = pts[k, 0] / norm[k, 0] if norm[k, 0] > 1e-3 else 640.0
    # distance: pinhole model, real palm length (world, metres) vs its image length (pixels).
    # A tilted palm looks shorter, which would read as farther, so take the least
    # foreshortened of several palm segments.
    distance_cm = 0.0
    if world is not None:
        f_px = (frame_w / 2) / np.tan(np.radians(tune.camera_hfov / 2))
        cands = []
        for i, j in PALM_SEGMENTS:
            lw = float(np.linalg.norm(world[i] - world[j]))
            lp = float(np.linalg.norm(pts[i, :2] - pts[j, :2]))
            if lp > 2 and lw > 1e-4:
                cands.append(f_px * lw / lp)
        if cands:
            distance_cm = 100.0 * min(cands)
    return Metrics(
        scale=scale,
        tap_index=d(THUMB_TIP, INDEX_TIP),
        tap_middle=d(THUMB_TIP, MIDDLE_TIP),
        tap_ring=d(THUMB_TIP, RING_TIP),
        tap_pinky=d(THUMB_TIP, PINKY_TIP),
        reach=(d(INDEX_TIP, WRIST) + d(MIDDLE_TIP, WRIST) + d(RING_TIP, WRIST)) / 3.0,
        extended=extended,  # type: ignore[arg-type]
        bends=tuple(bends),  # type: ignore[arg-type]
        palm=pts[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2].mean(axis=0),
        anchor=norm[[INDEX_MCP, MIDDLE_MCP, RING_MCP]].mean(axis=0),
        index_tip=pts[INDEX_TIP, :2].copy(),
        thumb_tip=pts[THUMB_TIP, :2].copy(),
        thumb_mcp=d(THUMB_TIP, INDEX_MCP),
        index_dir=_unit(pts[INDEX_TIP, :2] - pts[INDEX_MCP, :2]),
        thumb_side=min(d(THUMB_TIP, INDEX_PIP), d(THUMB_TIP, 7), d(THUMB_TIP, INDEX_TIP)),
        ring_reach=d(RING_TIP, WRIST),
        pen_anchor=norm[[3, THUMB_TIP, INDEX_MCP]].mean(axis=0),
        index_reach=d(INDEX_TIP, WRIST),
        middle_reach=d(MIDDLE_TIP, WRIST),
        scale_norm=scale / frame_w,
        distance_cm=distance_cm,
    )


def classify_pose(m: Metrics, tune: Tuning) -> str:
    i, mi, r, p = m.extended
    if i and mi and r and p:
        return "open"
    if not mi and not r and not p:
        if i and m.bends[0] < tune.point_bend:
            return "point"
        if m.bends[0] > tune.fist_bend and m.tap_index > tune.fist_gap:
            return "fist"
        return "curled"           # loose hand / pinch with everything folded
    if i and mi and not r and not p:
        return "peace"
    return "other"


class Debounced:
    """Touch detector with hysteresis and N-frame confirmation. Lower ratio = touching."""

    def __init__(self, on: float, off: float, frames: int, release_frames: int | None = None):
        self.on_th, self.off_th, self.frames = on, off, frames
        self.release_frames = frames if release_frames is None else release_frames
        self.on = False
        self._n = 0

    def update(self, ratio: float) -> bool | None:
        """Returns True on press, False on release, None when nothing changed."""
        crossing = ratio < self.on_th if not self.on else ratio > self.off_th
        self._n = self._n + 1 if crossing else 0
        if self._n >= (self.release_frames if self.on else self.frames):
            self.on = not self.on
            self._n = 0
            return self.on
        return None

    def force_off(self) -> bool:
        was = self.on
        self.on = False
        self._n = 0
        return was


class TapDetector:
    """A tap = one finger leaves its resting position and comes back quickly.

    Works whichever way the finger rests: on a pen (tap = lift and put back) or in the air
    (tap = touch and let go). The click fires when the finger returns, so a pose change
    (finger moves and stays) never clicks."""

    def __init__(self, tune: Tuning):
        self.tune = tune
        self.hist: deque = deque()      # (t, value) while at rest
        self.active = False
        self.start = 0.0
        self.peak = 0.0

    def baseline(self) -> float | None:
        if len(self.hist) < 5:
            return None
        return float(np.median([v for _, v in self.hist]))

    def reset(self) -> None:
        self.hist.clear()
        self.active = False

    def update(self, v: float, t: float, hand_still: bool) -> bool:
        """Returns True on the frame a tap completes."""
        tune = self.tune
        base = self.baseline()
        if not self.active:
            self.hist.append((t, v))
            while self.hist and t - self.hist[0][0] > tune.tap_baseline:
                self.hist.popleft()
            if base is not None and abs(v - base) > tune.tap_dev and hand_still:
                self.active, self.start, self.peak = True, t, abs(v - base)
            return False
        dev = abs(v - base) if base is not None else 0.0
        self.peak = max(self.peak, dev)
        if dev < tune.tap_dev * 0.5:
            if not hand_still:          # the whole hand moved back with it: not a tap
                self.reset()
                return False
            self.active = False
            return tune.tap_min <= t - self.start <= tune.tap_max
        if t - self.start > tune.tap_max or not hand_still:
            self.reset()                # held too long / whole hand moved: re-learn the resting spot
        return False


@dataclass
class HandTrack:
    label: str
    tune: Tuning
    last_seen: float = 0.0
    m: Metrics | None = None
    pose: str = "none"
    pose_frames: int = 0
    pose_since: float = 0.0
    fist_latched: bool = False
    hist: deque = field(default_factory=lambda: deque(maxlen=40))      # (t, palm_x, palm_y, scale)
    pinch_hist: deque = field(default_factory=lambda: deque(maxlen=60))  # (t, tap_index)
    raw_poses: deque = field(default_factory=lambda: deque(maxlen=5))
    swipe_ready_at: float = 0.0
    last_swipe: tuple[str, float] | None = None
    zoom_active: bool = False
    pen_since: float | None = None
    pen_seen: deque = field(default_factory=lambda: deque(maxlen=8))   # was the pen visible, last frames
    tip_now: np.ndarray | None = None      # pen tip this frame (normalized), None if not seen
    tip_offset: np.ndarray | None = None   # learned tip - grip offset, used while the pen is hidden
    ring_hist: deque = field(default_factory=lambda: deque(maxlen=45))
    zoom_prev: float | None = None
    zoom_moved_at: float = 0.0

    def __post_init__(self) -> None:
        t = self.tune
        self.left = Debounced(t.tap_on, t.tap_off, t.confirm_frames, t.release_frames)       # thumb + index
        self.right = Debounced(t.right_on, t.right_off, t.confirm_frames, t.release_frames)  # thumb + middle
        self.pen_left = TapDetector(t)    # index finger taps the pen
        self.pen_right = TapDetector(t)   # middle finger taps the pen

    def speed(self, window: float = 0.15) -> float:
        """Palm speed in hand-scales per second over the last `window` seconds."""
        if len(self.hist) < 2:
            return 0.0
        t1, x1, y1, s1 = self.hist[-1]
        t0, x0, y0 = t1, x1, y1
        for t0, x0, y0, _ in reversed(self.hist):
            if t1 - t0 >= window:
                break
        dt = t1 - t0
        return 0.0 if dt <= 0 else math.hypot(x1 - x0, y1 - y0) / s1 / dt

    def pinch_range(self, window: float) -> float:
        """How much thumb-index distance changed over the last `window` seconds."""
        if not self.pinch_hist:
            return 0.0
        t1 = self.pinch_hist[-1][0]
        vals = [v for (tt, v) in self.pinch_hist if t1 - tt <= window]
        return max(vals) - min(vals) if vals else 0.0


@dataclass
class GestureFrame:
    events: list[Event]
    hands: dict[str, HandTrack]
    cursor_on: bool
    cursor_hand: str | None
    pointer_xy: np.ndarray | None      # normalized image coords of the cursor hand anchor
    freeze: bool                       # pointer must hold still (tap, scroll, zoom, fist)
    rewind: bool                       # a tap just started on the cursor hand: undo the last ~80 ms
    clutch: bool                       # cursor hand is holding a fist
    scrolling: bool
    zooming: bool
    zoom_pts: tuple[np.ndarray, np.ndarray] | None
    toggle_progress: float             # 0..1 toward a mouse-mode toggle (fist hold)
    arm_progress: float                # 0..1 toward an arm/disarm toggle
    pointer_scale: float | None        # cursor hand size / frame width (distance-adaptive mapping)
    pen_mode: bool = False


class GestureEngine:
    def __init__(self, tune: Tuning | None = None):
        self.tune = tune or Tuning()
        self.tracks: dict[str, HandTrack] = {}
        self.cursor_on = False
        self.cursor_hand: str | None = None
        self._cursor_lost_at: float | None = None
        self._scrolling = False
        self._scroll_since = 0.0
        self._last_t: float | None = None
        self._two_zoom_prev: float | None = None
        self._two_zoom_frames = 0
        self._arm_since: float | None = None
        self._arm_latched = False
        self.pen_mode = False
        self._pen_lost_at: float | None = None
        self._pending_release: list[Event] = []

    # ------------------------------------------------------------------ helpers
    def _release_taps(self, tr: HandTrack, events: list[Event]) -> None:
        if tr.left.force_off():
            events.append(Event("release", tr.label, button="left"))
        if tr.right.force_off():
            events.append(Event("release", tr.label, button="right"))

    def _set_cursor(self, on: bool, hand: str | None, events: list[Event]) -> None:
        if on == self.cursor_on and hand == self.cursor_hand:
            return
        if not on:
            for tr in self.tracks.values():
                self._release_taps(tr, events)
        if not on:
            self.pen_mode = False
        self.cursor_on, self.cursor_hand = on, hand if on else None
        self._cursor_lost_at = None
        events.append(Event("cursor_on" if on else "cursor_off", hand or ""))

    def _end_scroll(self, events: list[Event]) -> None:
        if self._scrolling:
            self._scrolling = False
            events.append(Event("scroll_end"))

    # ------------------------------------------------------------------ main
    def update(self, hands: list, t: float) -> GestureFrame:
        """hands: objects with .label, .pts (21,3), .norm (21,2) and optionally .world (21,3)."""
        tune = self.tune
        events: list[Event] = self._pending_release   # pen clicks: release one frame after press
        self._pending_release = []

        seen = set()
        for h in hands:
            m = compute_metrics(h.pts, h.norm, tune, getattr(h, "world", None))
            if m.scale_norm > tune.max_hand_size or 0 < m.distance_cm < tune.min_distance_cm:
                continue    # too close to the camera to trust: treat as not there
            tr = self.tracks.get(h.label)
            if tr is None:
                tr = self.tracks[h.label] = HandTrack(h.label, tune)
            if tr.m is not None and t - tr.last_seen <= tune.lost_timeout:
                a = tune.smooth  # light EMA on distances: kills single-frame landmark jitter
                for name in SMOOTHED:
                    setattr(m, name, a * getattr(m, name) + (1 - a) * getattr(tr.m, name))
            tr.last_seen = t
            tr.m = m
            # pose voting: a new pose only takes over once it wins most recent frames
            tr.raw_poses.append(classify_pose(m, tune))
            votes = list(tr.raw_poses)[-tune.pose_votes:]
            best = max(set(votes), key=votes.count)
            pose = best if votes.count(best) * 2 > len(votes) else tr.pose
            if tr.pose == "none":
                pose = votes[-1]
            if pose != tr.pose:
                tr.pose_since = t
                tr.pose_frames = 0
            tr.pose_frames += 1
            tr.pose = pose
            tr.hist.append((t, float(m.palm[0]), float(m.palm[1]), m.scale))
            tip = getattr(h, "pen_tip_norm", None)
            tr.tip_now = None if tip is None else np.asarray(tip, dtype=float)
            tr.pen_seen.append(tr.tip_now is not None)
            if tr.tip_now is not None and m.pen_anchor is not None:
                off = tr.tip_now - m.pen_anchor
                tr.tip_offset = off if tr.tip_offset is None else 0.8 * tr.tip_offset + 0.2 * off
            tr.pinch_hist.append((t, m.tap_index))
            seen.add(h.label)

        # forget hands that have been gone a while (release anything they hold)
        for label in list(self.tracks):
            tr = self.tracks[label]
            if label not in seen and t - tr.last_seen > tune.lost_timeout:
                self._release_taps(tr, events)
                del self.tracks[label]

        if self.cursor_on:
            if self.cursor_hand in seen:
                self._cursor_lost_at = None
            else:
                self._cursor_lost_at = self._cursor_lost_at or t
                if t - self._cursor_lost_at > tune.cursor_timeout:
                    self._set_cursor(False, None, events)

        present = [self.tracks[label] for label in seen]

        # ---- fist held: toggle mouse mode
        toggle_progress = 0.0
        for tr in present:
            if tr.pose == "fist":
                held = t - tr.pose_since
                toggle_progress = max(toggle_progress, min(1.0, held / tune.fist_hold))
                self._release_taps(tr, events)
                if held >= tune.fist_hold and not tr.fist_latched and not self.pen_mode:
                    tr.fist_latched = True
                    if self.cursor_on and self.cursor_hand == tr.label:
                        self._set_cursor(False, None, events)
                    else:
                        self._set_cursor(True, tr.label, events)
            else:
                tr.fist_latched = False

        # ---- pen mode: gripping a pen turns the cursor on; letting go turns it off
        def pen_grip(tr):
            if sum(tr.pen_seen) >= 4:          # the pen itself is visible in this hand
                return True
            m = tr.m
            # index bent around the pen (a straight index is a POINT, a fully curled one a FIST)
            return (m.thumb_side < tune.pen_thumb_near and not m.extended[2] and not m.extended[3]
                    and tune.point_bend <= m.bends[0] < tune.fist_bend and tr.pose not in ("fist", "open"))
        for tr in present:
            tr.ring_hist.append((t, tr.m.ring_reach))
            if pen_grip(tr):
                tr.pen_since = tr.pen_since if tr.pen_since is not None else t
                if not self.cursor_on and t - tr.pen_since >= tune.pen_hold:
                    self._set_cursor(True, tr.label, events)
                    self.pen_mode = True
                    events.append(Event("pen_on", tr.label))
            else:
                tr.pen_since = None
        if self.pen_mode:
            ctr0 = self.tracks.get(self.cursor_hand) if self.cursor_hand in seen else None
            if ctr0 is not None and ctr0.pen_since is not None:
                self._pen_lost_at = None
            else:
                self._pen_lost_at = self._pen_lost_at or t
                if t - self._pen_lost_at > tune.pen_release:
                    self._pen_lost_at = None
                    self._set_cursor(False, None, events)

        # ---- two-hand zoom: both hands pointing with the index finger
        zoom_pts = None
        pointers = [tr for tr in present if tr.m.extended[0] and not tr.m.extended[1]
                    and not tr.m.extended[2]]
        two_zoom = False
        if len(pointers) == 2:
            self._two_zoom_frames += 1
            a, b = pointers
            spread = float(np.linalg.norm(a.m.index_tip - b.m.index_tip)) / ((a.m.scale + b.m.scale) / 2)
            if self._two_zoom_frames >= tune.two_hand_settle:
                two_zoom = True
                zoom_pts = (a.m.index_tip, b.m.index_tip)
                if self._two_zoom_prev is None:
                    events.append(Event("zoom_start", "both"))
                    self._two_zoom_prev = spread
                else:
                    dlog = math.log(max(spread, 1e-3)) - math.log(max(self._two_zoom_prev, 1e-3))
                    if abs(dlog) > tune.zoom_deadband:
                        events.append(Event("zoom", amount=dlog * tune.zoom_gain))
                        self._two_zoom_prev = spread
        else:
            if self._two_zoom_prev is not None:
                events.append(Event("zoom_end"))
            self._two_zoom_prev = None
            self._two_zoom_frames = 0

        zooming = two_zoom

        # ---- taps (mouse mode). A real tap keeps the ring finger straight; resting hands,
        # fists and pointing hands have it curled, which is what keeps them from clicking.
        rewind = False
        pen_tapping = False
        if self.cursor_on and self.pen_mode:
            ctr0 = self.tracks.get(self.cursor_hand) if self.cursor_hand in seen else None
            if ctr0 is not None and not zooming:
                m = ctr0.m
                rings = [v for (tt, v) in ctr0.ring_hist if t - tt <= tune.tap_baseline]
                hand_still = len(rings) < 5 or abs(m.ring_reach - float(np.median(rings))) < tune.tap_ring_still
                was = ctr0.pen_left.active or ctr0.pen_right.active
                tl = ctr0.pen_left.update(m.tap_index, t, hand_still)
                tr_ = ctr0.pen_right.update(m.tap_middle, t, hand_still)
                if tl and tr_:   # both moved: the one that moved more wins
                    tl, tr_ = ctr0.pen_left.peak >= ctr0.pen_right.peak, ctr0.pen_right.peak > ctr0.pen_left.peak
                for fired, button in ((tl, "left"), (tr_, "right")):
                    if fired:
                        events.append(Event("press", ctr0.label, button=button))
                        self._pending_release.append(Event("release", ctr0.label, button=button))
                now = ctr0.pen_left.active or ctr0.pen_right.active
                pen_tapping = now
                if now and not was:
                    rewind = True        # snap the cursor back to where it was before the finger moved
        elif self.cursor_on:
            for tr in present:
                m = tr.m
                if zooming or tr.pose == "fist":
                    self._release_taps(tr, events)
                    continue
                ring_straight = m.extended[2]
                # pick the finger that is relatively closer to its own threshold
                li_rel = m.tap_index / tune.tap_on
                ri_rel = m.tap_middle / tune.right_on
                li = m.tap_index if (tr.left.on or (ring_straight and not tr.right.on and li_rel <= ri_rel)) else INF
                ri = m.tap_middle if (tr.right.on or (ring_straight and not tr.left.on and ri_rel < li_rel)) else INF
                for deb, ratio, button in ((tr.left, li, "left"), (tr.right, ri, "right")):
                    change = deb.update(ratio)
                    if change is True:
                        events.append(Event("press", tr.label, button=button))
                        if tr.label == self.cursor_hand:
                            rewind = True
                    elif change is False:
                        events.append(Event("release", tr.label, button=button))

        # ---- scroll: point the index up / down and hold it still, any mode
        scroller = None
        if not zooming:
            for tr in present:
                if (tr.pose == "point" and t - tr.pose_since >= tune.scroll_still
                        and tr.pen_since is None and not (self.pen_mode and tr.label == self.cursor_hand)
                        and tr.pinch_range(tune.scroll_still) < tune.zoom_engage
                        and abs(tr.m.index_dir[1]) > tune.scroll_tilt
                        and not (tr.left.on or tr.right.on)):
                    scroller = tr
                    break
        if scroller is not None:
            if not self._scrolling:
                self._scrolling = True
                self._scroll_since = t
                events.append(Event("scroll_start", scroller.label))
            dt = min(max(t - self._last_t, 0.0), 0.1) if self._last_t is not None else 0.0
            held = t - self._scroll_since
            rate = tune.scroll_rate * (1 + (tune.scroll_accel - 1) * min(held / tune.scroll_ramp, 1.0))
            sign = 1.0 if scroller.m.index_dir[1] < 0 else -1.0   # pointing up = scroll up
            if dt > 0:
                events.append(Event("scroll", scroller.label, dy=sign * rate * dt))
        else:
            self._end_scroll(events)
        self._last_t = t

        # ---- swipes: open palm, fast, mostly along one axis
        for tr in present:
            if (self.cursor_on and tr.label == self.cursor_hand) or zooming or tr.pose != "open" \
                    or tr.pose_frames < tune.swipe_open_frames:
                continue
            if t < tr.swipe_ready_at or len(tr.hist) < 3:
                continue
            t1, x1, y1, s1 = tr.hist[-1]
            old = [hh for hh in tr.hist if t1 - hh[0] <= tune.swipe_window]
            t0, x0, y0, _ = old[0]
            dt = t1 - t0
            if dt < 0.06:
                continue
            dx, dy = (x1 - x0) / s1, (y1 - y0) / s1
            dist = math.hypot(dx, dy)
            if dist / dt < tune.swipe_speed or dist < tune.swipe_dist:
                continue
            if abs(dx) > abs(dy) * tune.swipe_axis:
                direction = "right" if dx > 0 else "left"
            else:
                continue    # only clean horizontal swipes count
            opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}[direction]
            if tr.last_swipe and tr.last_swipe[0] == opposite and t - tr.last_swipe[1] < tune.swipe_return_block:
                tr.hist.clear()     # this is the hand travelling back, not a new swipe
                continue
            events.append(Event("swipe", tr.label, dir=direction))
            tr.last_swipe = (direction, t)
            tr.swipe_ready_at = t + tune.swipe_cooldown
            tr.hist.clear()

        # ---- arm / disarm: both palms open and still
        arm_progress = 0.0
        both_open = len(present) == 2 and all(
            tr.pose == "open" and tr.speed() < tune.arm_still for tr in present
        )
        if both_open:
            self._arm_since = self._arm_since if self._arm_since is not None else t
            arm_progress = min(1.0, (t - self._arm_since) / tune.arm_hold)
            if arm_progress >= 1.0 and not self._arm_latched:
                self._arm_latched = True
                events.append(Event("arm"))
        else:
            self._arm_since = None
            self._arm_latched = False

        # ---- pointer signals
        ctr = self.tracks.get(self.cursor_hand) if self.cursor_on and self.cursor_hand in seen else None
        if ctr is None:
            pointer_xy = None
        elif self.pen_mode and ctr.m.pen_anchor is not None:
            # follow the pen TIP; while the pen is hidden, follow the grip plus the learned
            # tip offset, so the cursor doesn't jump when detection flickers
            if ctr.tip_offset is not None:
                pointer_xy = (ctr.m.pen_anchor + ctr.tip_offset).copy()
            else:
                pointer_xy = ctr.m.pen_anchor.copy()
        else:
            pointer_xy = ctr.m.anchor.copy()
        clutch = ctr is not None and ctr.pose == "fist"
        freeze = bool(ctr is not None and (ctr.right.on or self._scrolling or zooming or clutch or pen_tapping))
        return GestureFrame(
            events=events,
            hands=self.tracks,
            cursor_on=self.cursor_on,
            cursor_hand=self.cursor_hand,
            pointer_xy=pointer_xy,
            freeze=freeze,
            rewind=rewind,
            clutch=clutch,
            scrolling=self._scrolling,
            zooming=zooming,
            zoom_pts=zoom_pts,
            toggle_progress=toggle_progress,
            arm_progress=arm_progress,
            pointer_scale=ctr.m.scale_norm if ctr is not None else None,
            pen_mode=self.pen_mode,
        )
