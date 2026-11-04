"""Holographic hand linework for the preview window.

Thin white traces follow every finger, segmented reticles spin on the fingertips and the
palm, fingertips are webbed together, and a small wireframe prism over a grid floats by the
wrist with live readouts (rotation, distance, grip). The camera image itself is untouched.
Everything is drawn at 2x and shown at 2x, so 1 px lines stay crisp.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from airpen.gestures import GestureFrame

WHITE = (255, 255, 255)
SOFT = (200, 200, 200)
ORANGE = (40, 110, 255)   # BGR
BLUE = (255, 90, 50)
LIME = (120, 255, 120)
AA = cv2.LINE_AA

TIPS = (4, 8, 12, 16, 20)
CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
JOINTS = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
BONES = ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17))

_rng = np.random.default_rng(7)
# segment pattern per palm ring: list of (start_deg, length_deg); fixed so it looks designed
PALM_RINGS = []
for k in range(4):
    segs, a = [], 0.0
    while a < 360:
        length = float(_rng.uniform(18, 70))
        segs.append((a, min(length, 360 - a - 4)))
        a += length + float(_rng.uniform(6, 22))
    PALM_RINGS.append([s for s in segs if s[1] > 4])


def _ip(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _arc(img, c, r, a0, span, color=WHITE, th=1):
    if r >= 2:
        cv2.ellipse(img, _ip(c), (int(r), int(r)), 0, a0, a0 + span, color, th, AA)


def _chaikin(pts: np.ndarray, iters: int = 2) -> np.ndarray:
    for _ in range(iters):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((len(q) * 2, 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


class HandFX:
    def __init__(self, scale: int = 2):
        self.S = scale
        self.t0 = time.monotonic()

    def render(self, frame: np.ndarray, hands: list, gf: GestureFrame) -> np.ndarray:
        S = self.S
        img = cv2.resize(frame, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
        web = np.zeros_like(img)
        t = time.monotonic() - self.t0
        for hand in hands:
            tr = gf.hands.get(hand.label)
            self._hand(img, web, hand, tr, gf, t)
        return cv2.add(img, web)

    # ------------------------------------------------------------------ one hand
    def _hand(self, img, web, hand, tr, gf, t):
        S = self.S
        P = hand.pts[:, :2] * S
        hs = float(np.linalg.norm(P[0] - P[9]))          # hand size in display pixels
        if hs < 8:
            return
        m = tr.m if tr is not None else None
        is_cursor = gf.cursor_on and gf.cursor_hand == hand.label
        accent = LIME if is_cursor else WHITE

        # roll angle of the hand (0 = fingers up), 0..360 like a compass
        v = P[9] - P[0]
        roll = (math.degrees(math.atan2(v[0], -v[1])) + 360) % 360

        # --- skeleton: palm bones + finger chains
        for a, b in BONES:
            cv2.line(img, _ip(P[a]), _ip(P[b]), SOFT, 1, AA)
        for chain in CHAINS:
            base = 0 if chain[0] in (1, 5, 17) else chain[0]
            pts = P[[base, *chain]] if base == 0 else P[list(chain)]
            cv2.polylines(img, [np.round(pts).astype(np.int32)], False, WHITE, 1, AA)
            # circuit trace: a smoothed line riding alongside the finger
            seg = pts[1:] - pts[:-1]
            nrm = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
            off = hs * 0.07
            side = np.array([1.0, -0.6, 1.0, -0.6, 1.0])[: len(seg)]
            mids = (pts[1:] + pts[:-1]) / 2 + nrm * off * side[:, None]
            trace = _chaikin(np.vstack([pts[:1], mids, pts[-1:]]), 3)
            cv2.polylines(img, [np.round(trace).astype(np.int32)], False, SOFT, 1, AA)
        for j in JOINTS:
            cv2.circle(img, _ip(P[j]), max(2, int(hs * 0.03)), WHITE, 1, AA)

        # --- fingertip web
        tips = P[list(TIPS)]
        for i in range(1, 5):
            cv2.line(web, _ip(tips[0]), _ip(tips[i]), (110, 110, 110), 1, AA)
        cv2.polylines(web, [np.round(tips[1:]).astype(np.int32)], False, (150, 150, 150), 1, AA)

        # --- fingertip reticles
        for i, tip in enumerate(TIPS):
            c = P[tip]
            ph = t * (70 + 13 * i) * (1 if i % 2 else -1) + i * 47
            r1, r2 = hs * 0.15, hs * 0.22
            pressed = tr is not None and ((tip == 8 and tr.left.on) or (tip == 12 and tr.right.on))
            col = accent if not pressed else LIME
            _arc(img, c, r1, ph, 250, col, 1)
            for k in range(3):
                _arc(img, c, r2, -ph * 0.7 + k * 120, 46, col, 2 if pressed else 1)
            cv2.circle(img, _ip(c), max(2, int(hs * 0.025)), col, -1, AA)
            if pressed:
                cv2.circle(img, _ip(c), int(r2 * 1.35), col, 1, AA)

        # --- palm reticle: segmented rings spinning against each other
        c = P[[0, 5, 9, 13, 17]].mean(axis=0)
        R = hs * 0.40
        for k, segs in enumerate(PALM_RINGS):
            r = R * (0.38 + 0.21 * k)
            rot = roll + t * (25 + 12 * k) * (1 if k % 2 else -1)
            for a0, span in segs:
                _arc(img, c, r, a0 + rot, span, WHITE, 1)
                if k < 3 and span > 40:      # radial connectors give the maze look
                    ang = math.radians(a0 + rot)
                    p0 = c + r * np.array([math.cos(ang), math.sin(ang)])
                    p1 = c + (r + R * 0.21) * np.array([math.cos(ang), math.sin(ang)])
                    cv2.line(img, _ip(p0), _ip(p1), SOFT, 1, AA)
        _arc(img, c, R * 1.18, roll + 110, 120, accent, 2)
        _arc(img, c, R * 1.30, roll + 130, 70, SOFT, 1)

        # --- readouts under the palm
        lab = c + np.array([-hs * 0.45, hs * 0.62])
        tri = np.array([[0, -5], [-4, 3], [4, 3]]) * (S / 2) + lab + np.array([0, -4 * S / 2])
        cv2.fillPoly(img, [np.round(tri).astype(np.int32)], WHITE, AA)
        fs = 0.55 * S
        cv2.putText(img, f"rotation {int(roll):03d}", _ip(lab + np.array([9 * S, 0])),
                    cv2.FONT_HERSHEY_PLAIN, fs, WHITE, 1, AA)
        if m is not None and m.distance_cm > 0:
            cv2.putText(img, f"distance {m.distance_cm:4.0f} cm", _ip(lab + np.array([9 * S, 14 * S])),
                        cv2.FONT_HERSHEY_PLAIN, fs, SOFT, 1, AA)
        if tr is not None and tr.pose not in ("none", "other"):
            cv2.putText(img, tr.pose, _ip(lab + np.array([9 * S, 28 * S])),
                        cv2.FONT_HERSHEY_PLAIN, fs, accent, 1, AA)

        # --- detected stylus: shaft + tip reticle
        pen = getattr(hand, "pen", None)
        if pen is not None:
            tip, back = pen.tip * S, pen.back * S
            cv2.line(img, _ip(back), _ip(tip), (255, 230, 0), 2, AA)
            cv2.circle(img, _ip(tip), int(hs * 0.10), (255, 230, 0), 2, AA)
            _arc(img, tip, hs * 0.18, -t * 120, 270, (255, 230, 0), 1)
            cv2.putText(img, "PEN", _ip(tip + np.array([hs * 0.2, -hs * 0.1])),
                        cv2.FONT_HERSHEY_PLAIN, 0.7 * S, (255, 230, 0), 1, AA)

        # --- wireframe prism + grid floor beside the wrist
        side = -1 if hand.label == "Right" else 1
        anchor = P[0] + np.array([side * hs * 1.15, -hs * 0.05])
        grip = 0.0
        if m is not None:
            grip = float(np.clip((np.mean(m.bends) - 20) / 220, 0, 1)) * 100
        self._prism(img, anchor, hs, math.radians(roll * 2 + t * 40), grip)

    def _prism(self, img, anchor, hs, yaw, grip):
        S = self.S
        a, h = hs * 0.20, hs * 0.34
        tilt = math.radians(22)
        verts = []
        for y in (-h / 2, h / 2):
            for k in range(6):
                ang = yaw + k * math.pi / 3
                x, z = a * math.cos(ang), a * math.sin(ang)
                yy = y * math.cos(tilt) - z * math.sin(tilt)
                zz = y * math.sin(tilt) + z * math.cos(tilt)
                verts.append(anchor + np.array([x, yy + zz * 0.15]))
        top, bot = verts[:6], verts[6:]
        for i in range(6):
            j = (i + 1) % 6
            cv2.line(img, _ip(top[i]), _ip(top[j]), ORANGE, 1, AA)
            cv2.line(img, _ip(bot[i]), _ip(bot[j]), ORANGE, 1, AA)
            cv2.line(img, _ip(top[i]), _ip(bot[i]), ORANGE, 1, AA)

        # grid floor (perspective parallelogram)
        base = anchor + np.array([0, h * 0.95])
        u = np.array([hs * 0.16, hs * 0.035])
        w = np.array([-hs * 0.07, hs * 0.07])
        origin = base - 2 * u - 1.5 * w
        for i in range(5):
            cv2.line(img, _ip(origin + i * u), _ip(origin + i * u + 3 * w), BLUE, 1, AA)
        for j in range(4):
            cv2.line(img, _ip(origin + j * w), _ip(origin + j * w + 4 * u), BLUE, 1, AA)

        txt = anchor + np.array([-hs * 0.55, -h * 0.75])
        cv2.putText(img, "grip load", _ip(txt), cv2.FONT_HERSHEY_PLAIN, 0.55 * S, ORANGE, 1, AA)
        cv2.putText(img, f"{grip:4.1f}%", _ip(txt + np.array([0, 15 * S])),
                    cv2.FONT_HERSHEY_PLAIN, 0.75 * S, ORANGE, 1, AA)
