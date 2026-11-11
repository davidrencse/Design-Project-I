"""Preview window renderer: dark monochrome plate, hairline hand linework, small editorial labels.

The camera frame becomes a darkened, grainy grey plate; the dominant hand gets a bright bone-white
skeleton with square joint markers and corner-bracket reticles on the fingertips, the ignored hand a
dashed grey one. Chrome (frame, crop marks, wordmark, live status), a seven-row action rail and a
quiet "last action" numeral complete it. Shared helpers (palette, text, grain) live in airpen.ui.
Everything is drawn at 2x on a single grey channel and shown at 2x, so 1 px lines stay crisp.

`render(frame, hands, af=None, hand=None)` also works without an ActionFrame (calibrate / record):
then only the plate, the linework of every hand and the chrome are drawn.
"""
from __future__ import annotations

import math
import sys
import time

import cv2
import numpy as np

from airpen import ui
from airpen.actions import LABELS, ActionFrame
from airpen.ui import ASH, BONE, GRAPHITE, INK, SILVER, SMOKE, put_text

LAST_S = 1.2              # seconds the last-action numeral and the rail inversion take to fade out
THUMB_ONLY = (4, 5, 7)    # actions that exist only inside a latched fist

TIPS = (4, 8, 12, 16, 20)
CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
JOINTS = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
BONES = ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
GHOST_V = 138


def _drawable(hand) -> bool:
    """False for a hand whose image coordinates are not finite or absurdly large (bad tracking)."""
    xy = np.asarray(hand.pts)[:, :2]
    return bool(np.isfinite(xy).all() and np.abs(xy).max() < 1e6)


def _dbg(af, key: str, default: float = 0.0) -> float:
    """A float from the ActionFrame debug dict; missing or non-finite counts as absent."""
    try:
        v = float(af.debug.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fade(age: float) -> float:
    """1 right after the event, hold briefly, then a linear fade to 0 at LAST_S."""
    if not 0 <= age < LAST_S:
        return 0.0
    return 1.0 if age < 0.12 else 1.0 - (age - 0.12) / (LAST_S - 0.12)


class HandFX:
    def __init__(self, scale: int = 2):
        self.S = scale
        self.back = ui.Backdrop()

    def render(self, frame: np.ndarray, hands: list, af: ActionFrame | None = None, hand: str | None = None,
               last_action=None, now: float | None = None, fps: float | None = None,
               caption: str | None = None, overlay=None) -> np.ndarray:
        """hand: dominant hand label (None = draw every hand as dominant). last_action: (action id,
        time it fired) or None. now: the clock last_action's time uses (default time.monotonic()).
        fps: shown in the status block. caption: replaces the wordmark's sub-label. overlay: callable
        drawing on the grey canvas just before it becomes BGR (calibrate / record instructions)."""
        S = self.S
        H0, W0 = frame.shape[:2]
        g = self.back.plate(frame, (W0 * S, H0 * S))
        now = time.monotonic() if now is None else now
        flash = last_action[0] if last_action and 0 <= now - last_action[1] < 0.35 else 0
        k = ui.k_of(g)
        hands = [h for h in hands if _drawable(h)]
        for h in hands:
            if hand is not None and h.label != hand:
                self._ghost(g, h, k)
            else:
                self._hand(g, h, af, flash, k)
        ui.chrome(g, caption or "GESTURE INTERFACE / 07 ACTIONS")
        self._status(g, hands, af, hand, fps)
        if af is not None:
            self._rail(g, af, last_action, now, k)
            self._last(g, last_action, now, k)
        if overlay is not None:
            overlay(g)
        return ui.to_bgr(g)

    def render_safe(self, frame: np.ndarray, *args, **kw) -> np.ndarray:
        """render(), but a drawing failure (logged once) falls back to the plain camera frame."""
        try:
            return self.render(frame, *args, **kw)
        except Exception as exc:
            if not getattr(self, "_failed", False):
                self._failed = True
                print(f"[airpen] preview render failed ({type(exc).__name__}: {exc}); showing the plain frame",
                      file=sys.stderr)
            return frame

    # ------------------------------------------------------------------ chrome pieces
    def _status(self, g, hands, af, hand, fps):
        live = af.hand_seen if af is not None else bool(hands)
        rows = [("DOMINANT HAND", hand.upper() if hand else "ANY"),
                ("HANDS", f"{len(hands):02d}"),
                ("FPS", f"{fps:02.0f}" if fps else "--")]
        ui.status_block(g, rows, "TRACKING" if live else "NO HAND", live)

    def _rail(self, g, af, last_action, now, k):
        H, W = g.shape[:2]
        m = int(26 * k)
        x0 = W - m - int(214 * k)
        row = 24 * k
        top = int(H * 0.5 - 3.5 * row)
        ui.darken(g, x0 - 18 * k, top - 34 * k, W - m - 1, top + 7 * row + 40 * k, 0.5, 14 * k)
        cv2.line(g, (int(x0 - 10 * k), int(top - 4 * k)), (int(x0 - 10 * k), int(top + 7 * row + 2 * k)), ASH, 1, cv2.LINE_8)
        put_text(g, "ACTIONS", x0, top - 26 * k, 9.5 * k, ASH, "regular", track=3 * k)
        put_text(g, "07", W - m - 12 * k, top - 26 * k, 9.5 * k, ASH, "regular", track=3 * k, anchor="r")
        a_last = _fade(now - last_action[1]) if last_action else 0.0
        fired = last_action[0] if last_action else 0
        for aid in range(1, 8):
            y = top + (aid - 1) * row
            dim = aid in THUMB_ONLY and not af.in_fist
            base = ASH if dim else SILVER
            a = a_last ** 1.6 if aid == fired else 0.0          # fill alpha: lingers, then thins out
            if a > 0.01:                                   # inverted row: bone fill, black text, fading back
                reg = g[int(y): int(y + row - 2 * k), int(x0 - 6 * k): int(W - m - 4 * k)]
                cv2.convertScaleAbs(reg, dst=reg, alpha=1 - a, beta=BONE * a)
            inv = a > 0.4                                       # text flips to black once the fill is solid
            v = INK if inv else base
            put_text(g, f"{aid:02d}", x0, y + 4 * k, 10.5 * k, INK if inv else (ASH if dim else SMOKE),
                     "regular", track=2.4 * k)
            put_text(g, LABELS[aid].upper(), x0 + 34 * k, y + 4 * k, 10.5 * k, v, "regular", track=2.4 * k)
            prog = {3: _dbg(af, "fist_progress"), 6: _dbg(af, "peace_progress")}.get(aid, 0.0)
            if 0 < prog < 1:                               # hold progress: hairline under the row
                ui.hbar(g, x0 - 6 * k, y + row - 4 * k, 204 * k, prog, BONE, GRAPHITE)
        y = top + 7 * row + 14 * k
        ui.square(g, (x0 + 3 * k, y + 6 * k), max(2, int(3 * k)), BONE if af.in_fist else SMOKE, fill=af.in_fist)
        put_text(g, "FIST LATCH", x0 + 14 * k, y, 9.5 * k, BONE if af.in_fist else SMOKE, "regular", track=2.6 * k)
        put_text(g, "ON" if af.in_fist else "OFF", W - m - 12 * k, y, 9.5 * k, BONE if af.in_fist else SMOKE,
                 "regular", track=2.6 * k, anchor="r")

    def _last(self, g, last_action, now, k):
        if last_action is None:
            return
        aid, t0 = last_action
        a = _fade(now - t0)
        if a <= 0.01:
            return
        H, W = g.shape[:2]
        m = int(26 * k)
        x = m + 18 * k
        put_text(g, "LAST ACTION", x, H - m - 168 * k, 9.5 * k, SMOKE, "regular", track=3 * k, alpha=a)
        put_text(g, f"{aid:02d}", x - 4 * k, H - m - 152 * k, 84 * k, BONE, "thin", track=6 * k, alpha=a)
        y = H - m - 36 * k
        label = "THUMB SWIPE" if aid in (4, 5) else LABELS[aid].upper()
        w = put_text(g, label, x, y, 12 * k, BONE, "regular", track=3 * k, alpha=a)
        if aid in (4, 5):                                   # hairline arrow: right = 4, left = 5
            d = 1 if aid == 4 else -1
            ax = x + w + 16 * k
            v = int(BONE * a)
            cv2.line(g, (int(ax), int(y + 9 * k)), (int(ax + d * 26 * k), int(y + 9 * k)), v, 1, cv2.LINE_AA)
            tip = ax + d * 26 * k
            cv2.line(g, (int(tip), int(y + 9 * k)), (int(tip - d * 6 * k), int(y + 5 * k)), v, 1, cv2.LINE_AA)
            cv2.line(g, (int(tip), int(y + 9 * k)), (int(tip - d * 6 * k), int(y + 13 * k)), v, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ hands
    def _ghost(self, g, hand, k):
        P = hand.pts[:, :2] * self.S
        for a, b in BONES:
            ui.dashed(g, P[a], P[b], GHOST_V, 5 * k, 5 * k)
        for chain in CHAINS:
            base = [0] if chain[0] in (1, 5, 17) else []
            pts = P[base + list(chain)]
            for i in range(len(pts) - 1):
                ui.dashed(g, pts[i], pts[i + 1], GHOST_V, 5 * k, 5 * k)
        x, y = P[0] + np.array([14 * k, 8 * k])
        put_text(g, "IGNORED", x, y, 9.5 * k, SMOKE, "regular", track=3 * k)

    def _hand(self, g, hand, af, flash, k):
        S = self.S
        P = hand.pts[:, :2] * S
        hs = float(np.linalg.norm(P[0] - P[9]))
        if hs < 8:
            return
        H, W = g.shape[:2]
        m = int(26 * k)
        v = P[9] - P[0]
        roll = int((math.degrees(math.atan2(v[0], -v[1])) + 360) % 360)

        # --- skeleton: palm bones and finger chains
        for a, b in BONES:
            ui.line(g, P[a], P[b], SILVER)
        for chain in CHAINS:
            base = [0] if chain[0] in (1, 5, 17) else []
            pts = P[base + list(chain)]
            cv2.polylines(g, [np.round(pts).astype(np.int32)], False, BONE, 1, cv2.LINE_AA)
        r = max(1, int(1.6 * k))
        for j in JOINTS:
            ui.square(g, P[j], r, BONE)
        ui.cross(g, P[0], max(3, int(4 * k)), BONE)

        # --- thumb links to the two tap fingertips (dotted), brighter while a tap is firing
        for tip, ids in ((8, (1, 6)), (12, (2, 6))):
            ui.dashed(g, P[4], P[tip], BONE if flash in ids and flash != 6 else GRAPHITE + 14, 3 * k, 4 * k)

        # --- fingertip bracket reticles
        # a fist hides its fingertips: use the latch when we have it, else the tips' reach from the wrist
        fist = af.in_fist if af is not None else float(np.mean(np.linalg.norm(P[[8, 12, 16, 20]] - P[0], axis=1))) < hs * 1.15
        for i, tip in enumerate(TIPS):
            pressed = (tip == 8 and flash in (1, 6)) or (tip == 12 and flash in (2, 6)) or (tip == 4 and flash in (1, 2, 7))
            if fist and tip != 4:                          # a fist hides its fingertips: just a tick
                ui.square(g, P[tip], max(1, int(1.6 * k)), SILVER)
                continue
            rad = max(4, min(hs * (0.085 if pressed else 0.12), 17 * k))
            ui.bracket(g, P[tip], rad, BONE if pressed else SILVER)
            if pressed:
                ui.square(g, P[tip], max(1, int(2 * k)), BONE, fill=True)

        # --- palm mark: crosshair, a faint ring and four ticks turned with the hand
        c = P[[0, 5, 9, 13, 17]].mean(axis=0)
        R = hs * 0.26
        cv2.circle(g, ui.ip(c), int(R), GRAPHITE, 1, cv2.LINE_AA)
        for q in range(4):
            ang = math.radians(roll + 90 * q - 90)
            d = np.array([math.cos(ang), math.sin(ang)])
            ui.line(g, c + d * R, c + d * (R + 5 * k), SILVER)
        ui.cross(g, c, max(2, int(3 * k)), SILVER)

        # --- readouts beside the wrist: pose / distance / rotation
        side = 1 if P[17, 0] >= P[5, 0] else -1
        bx = P[0, 0] + side * (hs * 0.52 + 16 * k)
        by = P[0, 1] - 40 * k
        bx = min(max(bx, m + 14 * k), W - m - 250 * k) if side > 0 else min(max(bx - 118 * k, m + 14 * k), W - m - 250 * k)
        by = min(max(by, m + 14 * k), H - m - 70 * k)
        ui.line(g, P[0] + np.array([side * 8 * k, -4 * k]), (bx + (0 if side > 0 else 112 * k) - side * 5 * k, by + 6 * k), GRAPHITE + 20)
        rows = []
        if af is not None:
            rows.append(("POSE", af.pose.upper() if af.pose != "none" else "--"))
            mt = af.metrics
            rows.append(("DIST", f"{mt.distance_cm:.0f} CM" if mt is not None and mt.distance_cm > 0 else "--"))
        rows.append(("ROT", f"{roll:03d}°"))
        for i, (lab, val) in enumerate(rows):
            put_text(g, lab, bx, by + i * 15 * k, 9.5 * k, SMOKE, "regular", track=2.4 * k)
            put_text(g, val, bx + 40 * k, by + i * 15 * k, 9.5 * k, BONE, "regular", track=2.4 * k)

        # --- fist: thumb track (lateral thumb position) under the hand, to aim swipes
        if af is not None and af.in_fist and af.hand_seen:
            self._thumb_track(g, P, hs, af, k, m, H)

    def _thumb_track(self, g, P, hs, af, k, m, H):
        w = float(np.clip(hs * 1.3, 130 * k, 260 * k))
        cx = float(P[:, 0].mean())
        y = min(float(P[:, 1].max()) + 40 * k, H - m - 52 * k)
        x0, x1 = cx - w / 2, cx + w / 2
        cv2.line(g, (int(x0), int(y)), (int(x1), int(y)), ASH, 1, cv2.LINE_8)
        for x in (x0, x1):
            cv2.line(g, (int(x), int(y - 5 * k)), (int(x), int(y + 5 * k)), SILVER, 1, cv2.LINE_8)
        cv2.line(g, (int(cx), int(y - 3 * k)), (int(cx), int(y + 3 * k)), ASH, 1, cv2.LINE_8)
        put_text(g, "L", x0 - 6 * k, y - 5 * k, 9.5 * k, SMOKE, "regular", anchor="r")
        put_text(g, "R", x1 + 6 * k, y - 5 * k, 9.5 * k, SMOKE, "regular")
        lat = _dbg(af, "thumb_lat")
        px = cx + float(np.clip(lat / 0.5, -1, 1)) * w / 2
        armed = bool(af.debug.get("swipe_armed", False))
        cv2.line(g, (int(px), int(y - 7 * k)), (int(px), int(y + 7 * k)), BONE, 1, cv2.LINE_8)
        ui.square(g, (px, y), max(2, int(3 * k)), BONE, fill=armed)
        put_text(g, "THUMB", x0, y + 12 * k, 9 * k, SMOKE, "regular", track=2.6 * k)
        put_text(g, "ARMED" if armed else "SETTLING", x1, y + 12 * k, 9 * k, BONE if armed else SMOKE, "regular",
                 track=2.6 * k, anchor="r")


class CameraOverlay:
    """Hand landmarks and action feedback over the original color camera image."""

    FINGER_COLORS = ((60, 190, 255), (90, 225, 100), (240, 215, 70),
                     (255, 150, 100), (210, 110, 235))  # BGR
    ACTION_SECONDS = 2.0

    @staticmethod
    def _label(image, text, x, y, color=(255, 255, 255), scale=0.5):
        height, width = image.shape[:2]
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw > width - 24:
            scale *= max(1, width - 24) / tw
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = int(np.clip(x, 6, max(6, width - tw - 6)))
        y = int(np.clip(y, th + 6, max(th + 6, height - baseline - 6)))
        cv2.rectangle(image, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 3), (20, 20, 20), -1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def render(self, frame, hands, af=None, hand=None, last_action=None, now=None):
        image = frame.copy()
        height, width = image.shape[:2]
        now = time.monotonic() if now is None else now
        age = now - last_action[1] if last_action else -1
        flash = last_action[0] if last_action and 0 <= age < 0.4 else None
        highlights = {1: (4, 8), 2: (4, 12), 4: (4,), 5: (4,), 6: (8, 12), 7: (4,)}.get(flash, ())
        for tracked in hands:
            points = np.asarray(getattr(tracked, "pts", []))
            if (points.shape != (21, 3) or not np.isfinite(points).all()
                    or np.abs(points[:, :2]).max() > 1e6):
                continue
            xy = points[:, :2]
            if (xy[:, 0].max() < 0 or xy[:, 0].min() >= width
                    or xy[:, 1].max() < 0 or xy[:, 1].min() >= height):
                continue
            p = np.rint(xy).astype(np.int32)
            active = hand is None or tracked.label == hand
            palm_color = (225, 225, 225) if active else (135, 135, 135)
            for a, b in BONES:
                cv2.line(image, tuple(p[a]), tuple(p[b]), (20, 20, 20), 4, cv2.LINE_AA)
                cv2.line(image, tuple(p[a]), tuple(p[b]), palm_color, 2, cv2.LINE_AA)
            for chain, color in zip(CHAINS, self.FINGER_COLORS):
                color = color if active else (135, 135, 135)
                joints = (0, *chain)
                for a, b in zip(joints, joints[1:]):
                    cv2.line(image, tuple(p[a]), tuple(p[b]), (20, 20, 20), 4, cv2.LINE_AA)
                    cv2.line(image, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
                for joint in chain:
                    radius = 6 if active and joint in highlights else (4 if joint in TIPS else 3)
                    cv2.circle(image, tuple(p[joint]), radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
                    cv2.circle(image, tuple(p[joint]), radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, tuple(p[0]), 4, palm_color, -1, cv2.LINE_AA)
            label = f"{tracked.label} hand" + (" (ignored)" if not active else "")
            if active and af is not None and af.hand_seen:
                label += f" | {af.pose}"
            self._label(image, label, p[0, 0] + 10, p[0, 1] + 22, palm_color)
        if last_action and last_action[0] in LABELS and 0 <= age < self.ACTION_SECONDS:
            self._label(image, f"Recognized: {LABELS[last_action[0]]}", 12, height - 18,
                        color=(160, 240, 170), scale=0.6)
        return image

    def render_safe(self, frame, *args, **kwargs):
        try:
            return self.render(frame, *args, **kwargs)
        except Exception as exc:
            if not getattr(self, "_failed", False):
                self._failed = True
                print(f"[airpen] camera overlay failed ({exc}); showing the plain camera", file=sys.stderr)
            return frame
