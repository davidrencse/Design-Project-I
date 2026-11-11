"""The seven AirPen actions, recognized on the user's dominant hand only.

  1 index_tap          index fingertip taps the thumb tip
  2 middle_tap         middle fingertip taps the thumb tip
  3 fist               a fist is made (held ~0.3 s); latches until the hand clearly opens
  4 thumb_swipe_right  in a fist, the thumb swipes to the user's right
  5 thumb_swipe_left   in a fist, the thumb swipes to the user's left
  6 peace              index + middle raised in a V, ring + pinky folded
  7 thumb_tap          in a fist, the thumb lifts and comes back (a press)

The other hand is ignored. Everything is measured relative to the hand and divided by the hand's
own size, so it holds at any distance. Finger shape (bends, touch distances) reuses
airpen.gestures.compute_metrics and the calibrated thresholds in `Tuning`; the extra thresholds
live in `ActionTuning` and can be overridden by name in ~/.config/airpen/tuning.json.

Swipe history stays in a hand-relative 3D basis (world landmarks when available). Only the
emitted event is mapped to screen left/right, so turning the palm cannot flip the history.
Return to the starting thumb position between swipes; opening the hand resets the swipe state.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from airpen.gestures import (
    INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, SMOOTHED, THUMB_TIP, WRIST,
    Debounced, Metrics, TapDetector, Tuning, classify_pose, compute_metrics,
)
from airpen.settings import norm_hand

ACTIONS: dict[int, str] = {1: "index_tap", 2: "middle_tap", 3: "fist", 4: "thumb_swipe_right",
                           5: "thumb_swipe_left", 6: "peace", 7: "thumb_tap"}
LABELS: dict[int, str] = {1: "index tap", 2: "middle tap", 3: "fist", 4: "thumb swipe right",
                          5: "thumb swipe left", 6: "peace", 7: "thumb tap"}
INF = float("inf")
DIPS = (7, 11, 15, 19)      # finger DIP joints: the "surface" of a fist the thumb rests on


@dataclass
class ActionTuning:
    """Extra thresholds. Lengths are in hand-scales (wrist -> middle knuckle), times in seconds."""
    min_hand_score: float = 0.5       # reject uncertain handedness before driving PC controls
    # fist (3)
    fist_hold_s: float = 0.30          # curled fingers held this long = fist
    fist_grace_s: float = 0.15         # landmark flicker shorter than this never cancels/un-latches
    fist_exit_bend: float = 170.0      # mean finger bend below this = hand opened (fist un-latches)
    # taps (1, 2)
    tap_quiet_s: float = 0.35          # no taps this soon after the hand was curled (fist forming/opening)
    # peace (6)
    peace_hold_s: float = 0.30
    peace_curl_bend: float = 160.0     # ring + pinky bend above this = folded
    peace_angle: float = 14.0          # degrees between index and middle direction = a V, not two fingers together
    peace_grace_s: float = 0.15
    # thumb actions (4, 5, 7), only inside a latched fist
    thumb_arm_s: float = 0.30          # thumb must settle this long after the fist latches
    thumb_swipe_dist: float = 0.15     # lateral thumb travel within the window = swipe
    thumb_swipe_window_s: float = 0.60
    thumb_swipe_lateral: float = 1.5   # lateral travel must beat travel across the fist by this factor
    thumb_settle_s: float = 0.12       # after a swipe the thumb must be quiet this long...
    thumb_settle_range: float = 0.06   # ...(range of lateral position under this) to re-arm
    thumb_swipe_cooldown_s: float = 0.30
    thumb_return_range: float = 0.10   # return within this distance of the stroke's start to re-arm
    thumb_swipe_min_s: float = 0.05    # require a sustained stroke, not a landmark jump
    thumb_palm_still: float = 1.5      # palm speed limit, hand-scales per second (whole-hand moves don't count)
    thumb_tap_dev: float = 0.12        # thumb must leave its resting height by this much for a tap
    thumb_tap_lat_max: float = 0.12    # ...while moving sideways less than this (a swipe is never a tap)
    thumb_tap_cooldown_s: float = 0.25  # no tap this soon after a swipe


@dataclass
class ActionEvent:
    id: int
    name: str
    hand: str
    t: float


@dataclass
class ActionFrame:
    events: list[ActionEvent]
    hand_seen: bool
    pose: str
    in_fist: bool
    metrics: Metrics | None
    debug: dict = field(default_factory=dict)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / (na * nb)))))


def is_fist(m: Metrics, tune: Tuning) -> bool:
    """All four fingers curled, thumb not pinching the index."""
    return all(b > tune.fist_bend for b in m.bends) and m.tap_index > tune.fist_gap


def is_peace_shape(m: Metrics, tune: Tuning, act: ActionTuning, v_angle: float | None = None) -> bool:
    """Index + middle straight, ring + pinky folded, thumb away from both. `v_angle` is the angle
    between index and middle (degrees); None skips the spread check (metrics alone cannot tell)."""
    return (m.extended[0] and m.extended[1]
            and m.bends[2] > act.peace_curl_bend and m.bends[3] > act.peace_curl_bend
            and (v_angle is None or v_angle >= act.peace_angle)
            and m.tap_index > tune.tap_off and m.tap_middle > tune.right_off)


def tap_candidate(m: Metrics, tune: Tuning) -> str | None:
    """Which finger the thumb is touching right now ('index' / 'middle' / None), picking the
    finger relatively closer to its own threshold. Needs a straight ring finger (not a fist)."""
    if not m.extended[2]:
        return None
    li, ri = m.tap_index / tune.tap_on, m.tap_middle / tune.right_on
    if li <= ri:
        return "index" if li < 1.0 else None
    return "middle" if ri < 1.0 else None


def make_engine(hand: str) -> "ActionEngine":
    """The engine exactly as the live service builds it: personal thresholds from tuning.json.
    main.py and replay.py both use this so they cannot drift apart."""
    from airpen.tuning_io import load_tuning
    return ActionEngine(hand, load_tuning(Tuning()), load_tuning(ActionTuning()))


def effective_tuning() -> dict:
    """Every threshold the live engine would use right now (defaults + personal overrides)."""
    from dataclasses import asdict
    from airpen.tuning_io import load_tuning
    return {**asdict(load_tuning(Tuning())), **asdict(load_tuning(ActionTuning()))}


class ActionEngine:
    def __init__(self, hand: str = "Right", tune: Tuning | None = None, act: ActionTuning | None = None):
        self.hand = norm_hand(hand)
        self.tune = tune or Tuning()
        self.act = act or ActionTuning()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        tune, act = self.tune, self.act
        self.in_fist = False
        self._last_update: float | None = None
        self._last_seen: float | None = None
        self._m: Metrics | None = None
        self._fist_since: float | None = None
        self._fist_miss: float | None = None
        self._open_since: float | None = None
        self._latch_t = 0.0
        self._curl_t = -1e9
        self._peace_since: float | None = None
        self._peace_miss: float | None = None
        self._peace_fired = False
        self._left = Debounced(tune.tap_on, tune.tap_off, tune.confirm_frames, tune.release_frames)
        self._right = Debounced(tune.right_on, tune.right_off, tune.confirm_frames, tune.release_frames)
        self._palm: deque = deque(maxlen=30)           # (t, x, y, scale)
        self._reset_thumb()

    def _reset_thumb(self) -> None:
        self._thumb: deque = deque(maxlen=60)          # (t, lateral, across, height)
        self._tapdet = TapDetector(replace(self.tune, tap_dev=self.act.thumb_tap_dev))
        self._raw_thumb: deque = deque(maxlen=3)
        self._armed = False                            # first learn a settled neutral position
        self._neutral: tuple[float, float] | None = None
        self._return_origin: tuple[float, float] | None = None
        self._stroke_since: float | None = None
        self._lock_until = 0.0
        self._tap_block_until = 0.0
        self._axis_sign: float | None = None

    # ------------------------------------------------------------------ helpers
    def _pick(self, hands: list):
        best = None
        for h in hands:
            if h.label != self.hand:
                continue
            score = getattr(h, "score", 1.0)
            if not math.isfinite(score) or score < self.act.min_hand_score:
                continue
            pts, norm, world = h.pts, h.norm, getattr(h, "world", None)
            if (pts.shape != (21, 3) or norm.shape != (21, 2)
                    or not np.isfinite(pts).all() or not np.isfinite(norm).all()
                    or (world is not None and (world.shape != (21, 3) or not np.isfinite(world).all()))):
                continue
            if np.linalg.norm(pts[WRIST, :2] - pts[MIDDLE_MCP, :2]) < 2:
                continue
            if world is not None and np.linalg.norm(world[WRIST] - world[MIDDLE_MCP]) < 1e-5:
                continue
            m = compute_metrics(h.pts, h.norm, self.tune, getattr(h, "world", None))
            # Estimated world scale shrinks when the palm turns edge-on. It is not an
            # independent range sensor: require image-size evidence before rejecting it.
            if (m.scale_norm > self.tune.max_hand_size
                    or (m.scale_norm > 0.75 * self.tune.max_hand_size
                        and 0 < m.distance_cm < self.tune.min_distance_cm)):
                continue
            if best is None or m.scale_norm > best[1].scale_norm:
                best = (h, m)
        return best

    def _shape(self, h) -> np.ndarray:
        world = getattr(h, "world", None)
        return world if world is not None else h.pts * np.array([1.0, 1.0, self.tune.z_weight])

    def _palm_speed(self, t: float, window: float = 0.25) -> float:
        if len(self._palm) < 2:
            return 0.0
        t1, x1, y1, s1 = self._palm[-1]
        t0, x0, y0 = t1, x1, y1
        for t0, x0, y0, _ in reversed(self._palm):
            if t1 - t0 >= window:
                break
        return 0.0 if t1 <= t0 else math.hypot(x1 - x0, y1 - y0) / s1 / (t1 - t0)

    def _emit(self, events: list, aid: int, t: float) -> None:
        events.append(ActionEvent(aid, ACTIONS[aid], self.hand, t))

    # ------------------------------------------------------------------ main
    def update(self, hands: list, t: float) -> ActionFrame:
        tune, act = self.tune, self.act
        events: list[ActionEvent] = []
        if not math.isfinite(t):
            self.reset()
            return ActionFrame([], False, "none", False, None, {})
        if self._last_update is not None and t <= self._last_update:
            return ActionFrame([], False, "none", self.in_fist, None, {})
        self._last_update = t
        picked = self._pick(hands)
        if picked is None:
            # Keep the fist latch over short dropouts, but never bridge missing
            # landmarks with a thumb stroke from before the dropout.
            self._reset_thumb()
            self._palm.clear()
            if self._last_seen is not None and t - self._last_seen > tune.lost_timeout:
                self.reset()
            return ActionFrame(events, False, "none", self.in_fist, None, {})
        h, m = picked
        if self._last_seen is not None and t - self._last_seen > tune.lost_timeout:
            self.reset()                        # reacquired after a gap: a new acquisition, no stale state
        if self._m is not None and self._last_seen is not None and t - self._last_seen <= tune.lost_timeout:
            a = tune.smooth                     # same light EMA GestureEngine uses on distance features
            for name in SMOOTHED:
                setattr(m, name, a * getattr(m, name) + (1 - a) * getattr(self._m, name))
        self._last_update = t
        self._m, self._last_seen = m, t
        self._palm.append((t, float(m.palm[0]), float(m.palm[1]), m.scale))
        shape = self._shape(h)
        bends = m.bends
        ext = m.extended
        n_curled = sum(b > tune.fist_bend for b in bends)

        # ---- fist (3): all four fingers curled, held; latched until the hand clearly opens
        raw_fist = is_fist(m, tune)
        if not self.in_fist:
            if raw_fist:
                self._fist_since = t if self._fist_since is None else self._fist_since
                self._fist_miss = None
                if t - self._fist_since >= act.fist_hold_s:
                    self.in_fist, self._latch_t = True, t
                    self._open_since = None
                    self._reset_thumb()
                    self._emit(events, 3, t)
            elif self._fist_since is not None:
                self._fist_miss = t if self._fist_miss is None else self._fist_miss
                if t - self._fist_miss > act.fist_grace_s:
                    self._fist_since = self._fist_miss = None
        else:
            opened = any(ext) or float(np.mean(bends)) < act.fist_exit_bend
            if opened:
                self._open_since = t if self._open_since is None else self._open_since
                if t - self._open_since >= act.fist_grace_s:
                    self.in_fist, self._open_since = False, None
                    self._fist_since = self._fist_miss = None
                    self._reset_thumb()
            else:
                self._open_since = None
        if raw_fist or self.in_fist or n_curled >= 2:
            self._curl_t = t

        # ---- peace (6): index + middle in a V, ring + pinky folded, thumb away
        v_idx = shape[INDEX_TIP] - shape[INDEX_MCP]
        v_mid = shape[MIDDLE_TIP] - shape[MIDDLE_MCP]
        v_angle = _angle_deg(v_idx, v_mid)
        raw_peace = not self.in_fist and is_peace_shape(m, tune, act, v_angle)
        if raw_peace:
            self._peace_miss = None
            self._peace_since = t if self._peace_since is None else self._peace_since
            if not self._peace_fired and t - self._peace_since >= act.peace_hold_s:
                self._peace_fired = True
                self._emit(events, 6, t)
        elif self._peace_since is not None:
            self._peace_miss = t if self._peace_miss is None else self._peace_miss
            if t - self._peace_miss > act.peace_grace_s:
                self._peace_since = self._peace_miss = None
                self._peace_fired = False

        # ---- taps (1, 2): fire on press. A real tap keeps the ring finger straight; fists,
        # peace signs, pointing and a hand that was just curled never tap.
        quiet = t - self._curl_t >= act.tap_quiet_s
        tap_ok = (not self.in_fist and not raw_peace and ext[2] and n_curled < 2 and quiet
                  and self._peace_since is None)
        if tap_ok:
            li_rel = m.tap_index / tune.tap_on                  # pick the finger relatively closer to its threshold
            ri_rel = m.tap_middle / tune.right_on
            li = m.tap_index if (self._left.on or (not self._right.on and li_rel <= ri_rel)) else INF
            ri = m.tap_middle if (self._right.on or (not self._left.on and ri_rel < li_rel)) else INF
            if self._left.update(li) is True:
                self._emit(events, 1, t)
            if self._right.update(ri) is True:
                self._emit(events, 2, t)
        else:
            self._left.force_off()
            self._right.force_off()

        # ---- thumb actions (4, 5, 7): only inside a latched fist, measured relative to the hand
        lat = across = height = 0.0
        thumb_active = self.in_fist and self._open_since is None and t - self._latch_t >= act.thumb_arm_s
        pstill = self._palm_speed(t)
        if self.in_fist:
            # Keep the motion history in anatomical coordinates. Map only the final
            # event's direction to the camera, so a palm-axis sign change cannot
            # manufacture a large thumb displacement.
            u = _unit(shape[PINKY_MCP] - shape[INDEX_MCP])
            v = shape[MIDDLE_MCP] - shape[WRIST]
            v = _unit(v - u * float(v @ u))
            image_axis = h.pts[PINKY_MCP, :2] - h.pts[INDEX_MCP, :2]
            if abs(image_axis[0]) > max(0.15 * m.scale, 0.3 * float(np.linalg.norm(image_axis))):
                self._axis_sign = 1.0 if image_axis[0] > 0 else -1.0
            s3 = max(float(np.linalg.norm(shape[WRIST] - shape[MIDDLE_MCP])), 1e-9)
            rel = (shape[THUMB_TIP] - (shape[INDEX_MCP] + shape[PINKY_MCP]) / 2) / s3
            lat = float(rel @ u)
            across = float(rel @ v)
            height = float(np.linalg.norm(shape[THUMB_TIP] - shape[list(DIPS)].mean(axis=0))) / s3
        if thumb_active and self._axis_sign is not None:
            self._raw_thumb.append((lat, across))
            lat, across = np.median(self._raw_thumb, axis=0)
            self._thumb.append((t, float(lat), float(across), height))
            still = pstill < act.thumb_palm_still
            self._thumb_swipe(events, t, still)
            xs = [q[1] for q in self._thumb]
            ys = [q[2] for q in self._thumb]
            xspan = max(xs) - min(xs) if xs else 0.0
            yspan = max(ys) - min(ys) if ys else 0.0
            lateral_motion = (self._return_origin is not None
                              or (xspan > act.thumb_tap_lat_max and xspan > act.thumb_swipe_lateral * yspan))
            if t >= self._tap_block_until and not lateral_motion:
                self._thumb_tap(events, t, height, still)
            else:
                self._tapdet.reset()
        else:
            # Never bridge a pose transition with a motion history from the old pose.
            self._thumb.clear()
            self._raw_thumb.clear()
            self._armed = False
            self._neutral = None
            self._stroke_since = None
            self._tapdet.reset()

        if self.in_fist or raw_fist:
            pose = "fist"
        elif raw_peace:
            pose = "peace"
        else:
            pose = classify_pose(m, tune)
            pose = {"peace": "other", "fist": "curled"}.get(pose, pose)
        debug = dict(
            fist_progress=0.0 if self._fist_since is None else min(1.0, (t - self._fist_since) / act.fist_hold_s),
            peace_progress=0.0 if self._peace_since is None else min(1.0, (t - self._peace_since) / act.peace_hold_s),
            thumb_lat=lat, thumb_across=across, thumb_height=height, palm_speed=pstill,
            v_angle=v_angle, n_curled=n_curled, tap_ok=tap_ok, swipe_armed=self._armed,
            tap_index=m.tap_index, tap_middle=m.tap_middle,
        )
        return ActionFrame(events, True, pose, self.in_fist, m, debug)

    # ------------------------------------------------------------------ thumb pieces
    def _thumb_swipe(self, events: list, t: float, still: bool) -> None:
        act, hist = self.act, self._thumb
        if not hist:
            return
        _, lat, across, _ = hist[-1]
        if not still:
            hist.clear()
            self._raw_thumb.clear()
            self._armed = False
            self._neutral = None
            self._stroke_since = None
            return
        if not self._armed:
            # A return stroke is a reset, even when it takes longer than a cooldown.
            if self._return_origin is not None:
                x, _ = self._return_origin
                if abs(lat - x) > act.thumb_return_range:
                    hist.clear()
                    return
                # Returning through the origin completes the reset; the next resting
                # position may drift slightly as the grip relaxes.
                self._return_origin = None
                hist.clear()
                hist.append((t, lat, across, 0.0))
            if t < self._lock_until:
                hist.clear()
                return
            quiet = list(hist)
            # Include a complete time interval, independent of camera frame rate.
            while len(quiet) > 2 and t - quiet[1][0] >= act.thumb_settle_s:
                quiet.pop(0)
            if len(quiet) < 2 or t - quiet[0][0] < act.thumb_settle_s:
                return
            xs, ys = [q[1] for q in quiet], [q[2] for q in quiet]
            if max(xs) - min(xs) > act.thumb_settle_range or max(ys) - min(ys) > act.thumb_settle_range:
                return
            self._neutral = (float(np.median(xs)), float(np.median(ys)))
            self._return_origin = None
            self._stroke_since = None
            self._armed = True
            hist.clear()
            hist.append((t, lat, across, 0.0))
            return
        assert self._neutral is not None
        dx, dy = lat - self._neutral[0], across - self._neutral[1]
        if abs(dx) <= act.thumb_settle_range:
            self._stroke_since = None
            while len(hist) > 2 and t - hist[1][0] >= act.thumb_settle_s:
                hist.popleft()
            return
        if self._stroke_since is None:
            self._stroke_since = hist[-2][0] if len(hist) >= 2 else t
        duration = t - self._stroke_since
        if duration > act.thumb_swipe_window_s:
            # A slow pose change is not a swipe. Learn a new rest after it settles.
            self._armed, self._neutral, self._stroke_since = False, None, None
            hist.clear()
            return
        if duration < act.thumb_swipe_min_s or abs(dx) < act.thumb_swipe_dist:
            return
        if abs(dx) < act.thumb_swipe_lateral * abs(dy):
            return
        path = [q[1] for q in hist if q[0] >= self._stroke_since]
        if len(path) < 3:
            return
        travel = sum(abs(b - a) for a, b in zip(path, path[1:]))
        if travel and abs(path[-1] - path[0]) / travel < 0.7:
            return    # jitter has large accumulated travel but little coherent displacement
        aid = 4 if dx * self._axis_sign > 0 else 5
        self._emit(events, aid, t)
        self._return_origin = self._neutral
        self._armed, self._neutral, self._stroke_since = False, None, None
        self._lock_until = t + act.thumb_swipe_cooldown_s
        self._tap_block_until = t + act.thumb_tap_cooldown_s
        hist.clear()
        self._tapdet.reset()

    def _thumb_tap(self, events: list, t: float, height: float, still: bool) -> None:
        act = self.act
        det = self._tapdet
        start = det.start
        if det.update(height, t, still):
            lats = [l for (tt, l, _, _) in self._thumb if tt >= start - 0.15]
            if lats and max(lats) - min(lats) < act.thumb_tap_lat_max:
                self._emit(events, 7, t)
