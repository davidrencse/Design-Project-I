"""Finds a white stylus (e.g. Apple Pencil) in the hand with classic OpenCV.

Colour alone can't do it: a white pen against a white wall or a washed-out camera image is
the same colour as the background. So, inside a box around the grip:
  1. keep bright, unsaturated pixels (white)
  2. remove the palm (the pen tip sticks out past it; skin can wash out to white)
  3. top-hat style: subtract a morphological opening with a kernel wider than a pen, which
     deletes every white area wider than a pen (walls, sheets, paper) and keeps thin strokes
  4. keep the elongated component that touches the thumb/index grip, fit a line to it
  5. the writing tip is the end farther from the index knuckle
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PALM = (0, 1, 2, 5, 9, 13, 17)


@dataclass
class PenSighting:
    tip: np.ndarray        # (2,) pixels, full frame
    back: np.ndarray       # (2,) pixels, the other end
    length: float          # pixels
    score: float           # 0..1


class PenDetector:
    def __init__(self, sat_max: int = 70, val_min: int = 185, roi_scales: float = 2.4,
                 min_len: float = 0.35, max_width: float = 0.20, min_aspect: float = 3.0,
                 finger_width: float = 0.24, ridge_min: float = 0.6):
        self.sat_max = sat_max          # white = low saturation...
        self.val_min = val_min          # ...and bright
        self.roi_scales = roi_scales    # search box half-size, in hand sizes
        self.min_len = min_len          # pen must be at least this long (hand sizes)
        self.max_width = max_width      # allow antialiasing/closing around a narrow stylus
        self.min_aspect = min_aspect
        self.finger_width = finger_width  # fingers are masked out with capsules this wide
        self.ridge_min = ridge_min        # share of the stroke that must be darker on BOTH sides
        self.debug_mask: np.ndarray | None = None

    def detect(self, frame_bgr: np.ndarray, pts: np.ndarray) -> PenSighting | None:
        """pts: (21,2+) hand landmarks in pixels (same frame)."""
        self.debug_mask = None
        if pts.shape[0] != 21 or not np.isfinite(pts).all():
            return None
        P = pts[:, :2]
        scale = float(np.linalg.norm(P[0] - P[9]))
        if scale < 12:
            return None
        h, w = frame_bgr.shape[:2]
        grip = (P[4] + P[7] + P[8]) / 3.0          # thumb tip + index DIP/tip: where a pen is held
        r = self.roi_scales * scale
        x0, y0 = int(max(0, grip[0] - r)), int(max(0, grip[1] - r))
        x1, y1 = int(min(w, grip[0] + r)), int(min(h, grip[1] + r))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        # work at ~40 px per hand-size: plenty for a pen, several times fewer pixels
        f = min(1.0, 40.0 / scale)
        roi = frame_bgr[y0:y1, x0:x1]
        if f < 1.0:
            roi = cv2.resize(roi, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        P = (P - [x0, y0]) * f + [x0, y0]          # landmarks in working coordinates (offset kept)
        grip = (grip - [x0, y0]) * f + [x0, y0]
        scale = scale * f
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # "white" = unsaturated AND clearly brighter than this neighbourhood (adapts to exposure)
        thr = min(245, max(self.val_min, float(np.median(hsv[::3, ::3, 2])) + 25))
        white = ((hsv[..., 1] < self.sat_max) & (hsv[..., 2] > thr)).astype(np.uint8) * 255

        # 2. drop the palm and the fingers (washed-out skin reads as white; the pen tip sticks
        # out past the fingertips, so masking the fingers never hides all of it)
        palm = np.round(P[list(PALM)] - [x0, y0]).astype(np.int32)
        hull = cv2.convexHull(palm)
        cv2.fillConvexPoly(white, hull, 0)
        fw = max(3, int(self.finger_width * scale))
        for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
            q = np.round(P[list(chain)] - [x0, y0]).astype(np.int32)
            cv2.polylines(white, [q], False, 0, fw)
            cv2.circle(white, tuple(int(v) for v in q[-1]), fw // 2, 0, -1)
        white_full = white.copy()

        # 3. remove anything wider than a pen
        k = max(3, int(self.max_width * scale) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wide = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        thin = cv2.subtract(white, wide)
        # Keep enough working resolution for a diagonal stylus to survive this cleanup.
        thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.debug_mask = thin

        # 4. elongated component touching the grip
        n, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
        g = grip - [x0, y0]
        best, best_len = None, 0.0
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area < max(6, (0.05 * scale) ** 2) or np.hypot(bw, bh) < self.min_len * scale:
                continue                                  # specks / too short: cheap reject
            dxb = max(bx - g[0], 0, g[0] - (bx + bw))
            dyb = max(by - g[1], 0, g[1] - (by + bh))
            if np.hypot(dxb, dyb) > 0.6 * scale:
                continue                                  # bounding box nowhere near the grip
            ys, xs = np.nonzero(labels[by:by + bh, bx:bx + bw] == i)
            pix = np.stack([xs + bx, ys + by], axis=1).astype(np.float32)
            if len(pix) > 600:
                pix = np.ascontiguousarray(pix[:: len(pix) // 600 + 1])   # OpenCV needs contiguous memory
            if np.min(np.hypot(pix[:, 0] - g[0], pix[:, 1] - g[1])) > 0.6 * scale:
                continue                                  # not in the hand
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(pix)
            long_, short = max(rw, rh), max(1.0, min(rw, rh))
            if long_ < self.min_len * scale or short > self.max_width * scale or long_ / short < self.min_aspect:
                continue
            if long_ > best_len and self._sticks_out(pix, P - [x0, y0], scale) \
                    and self._is_ridge(pix, white_full, scale, roi):
                best, best_len = pix, long_
        if best is None:
            return None

        # 5. line fit, endpoints, tip = end farther from the index knuckle
        vx, vy, cx, cy = cv2.fitLine(best, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy])
        proj = (best - [cx, cy]) @ d
        # back to full-frame pixels
        a = (np.array([cx, cy]) + d * proj.min()) / f + [x0, y0]
        b = (np.array([cx, cy]) + d * proj.max()) / f + [x0, y0]
        knuckle = (P[5] - [x0, y0]) / f + [x0, y0]
        best_len /= f
        scale /= f
        tip, back = (a, b) if np.linalg.norm(a - knuckle) > np.linalg.norm(b - knuckle) else (b, a)
        score = float(min(1.0, best_len / (2.0 * scale)))
        return PenSighting(tip=tip, back=back, length=float(best_len), score=score)

    def _is_ridge(self, pix: np.ndarray, white: np.ndarray, scale: float, roi: np.ndarray) -> bool:
        """A pen in front of something is a bright stroke with darker pixels on BOTH sides.
        A sliver of white wall next to a finger has white on one side, so it fails."""
        vx, vy, cx, cy = cv2.fitLine(pix, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy])
        n = np.array([-vy, vx])
        proj = (pix - [cx, cy]) @ d
        off = max(3.0, 0.14 * scale)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.int16)
        h, w = gray.shape
        good = total = 0
        for s_ in np.linspace(proj.min(), proj.max(), 12):
            c = np.array([cx, cy]) + d * s_
            ci = np.round(c).astype(int)
            if not (0 <= ci[0] < w and 0 <= ci[1] < h):
                continue
            centre = gray[ci[1], ci[0]]
            sides = []
            for sgn in (1, -1):
                q = np.round(c + sgn * n * off).astype(int)
                if not (0 <= q[0] < w and 0 <= q[1] < h):
                    break
                sides.append(gray[q[1], q[0]])
            if len(sides) < 2:
                continue
            total += 1
            if all(centre - v > 25 for v in sides):
                good += 1
        return total >= 4 and good / total >= self.ridge_min

    @staticmethod
    def _sticks_out(pix: np.ndarray, P: np.ndarray, scale: float) -> bool:
        """A held pen pokes out of the hand outline; gaps between fingers never do."""
        hull = cv2.convexHull(np.round(P).astype(np.float32))
        far = [cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in pix[:: max(1, len(pix) // 60)]]
        outside = [-d for d in far if d < 0]
        return bool(outside) and max(outside) > 0.2 * scale
