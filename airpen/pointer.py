"""Maps the cursor hand's anchor (camera space) to a cursor position (screen space, 0..1).

Behaves like a mouse rather than a touchscreen: when the pointer is frozen (tap, scroll,
zoom, fist clutch) or the hand is reacquired, the cursor stays put and the hand-to-screen
offset absorbs the difference. Hitting a screen edge re-syncs the offset, the same way
a mouse "catches up" at the edge of the screen.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from airpen.filters import OneEuro2D

# filter in pseudo-pixels so the classic One Euro parameters (tuned for pixels) apply
UNITS = 1920.0


class Pointer:
    def __init__(self, box: tuple[float, float, float, float] = (0.25, 0.20, 0.75, 0.70),
                 min_cutoff: float = 0.8, beta: float = 0.010,
                 rewind_s: float = 0.08, settle_s: float = 0.15,
                 ref_scale: float = 0.11, adapt_tau: float = 1.0):
        self.base_box = box                 # active region at the reference hand distance
        self.box = box                      # current active region in camera space: x0, y0, x1, y1
        self.ref_scale = ref_scale          # hand size (fraction of frame width) the box is tuned for
        self.adapt_tau = adapt_tau          # seconds; slow, so the zone doesn't breathe with jitter
        self._scale: float | None = None
        self._t_prev: float | None = None
        self.filter = OneEuro2D(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)
        self.rewind_s = rewind_s            # how far back to jump when a tap starts
        self.settle_s = settle_s            # how long to hold still after a tap starts
        self.offset = np.zeros(2)
        self.out: np.ndarray | None = None  # current cursor, screen-normalized
        self._hist: deque[tuple[float, np.ndarray]] = deque(maxlen=60)
        self._hold_until = 0.0
        self._tracking = False

    def recenter(self) -> None:
        self.offset[:] = 0.0

    def map(self, xy: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        return np.array([(xy[0] - x0) / (x1 - x0), (xy[1] - y0) / (y1 - y0)])

    def _adapt(self, hand_scale: float | None, t: float) -> None:
        """Distance-adaptive zone: a far hand (small in frame) moves fewer pixels for the same
        physical motion, so shrink the zone around its centre; a near hand gets a bigger zone."""
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        if hand_scale is None or hand_scale <= 0:
            return
        if self._scale is None:
            self._scale = hand_scale
        else:
            a = 1 - np.exp(-dt / self.adapt_tau)
            self._scale += a * (hand_scale - self._scale)
        k = float(np.clip(self._scale / self.ref_scale, 0.5, 2.0))
        x0, y0, x1, y1 = self.base_box
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * k, (y1 - y0) / 2 * k
        self.box = (cx - hw, cy - hh, cx + hw, cy + hh)

    def update(self, xy: np.ndarray | None, t: float, freeze: bool = False,
               rewind: bool = False, hand_scale: float | None = None) -> np.ndarray | None:
        self._adapt(hand_scale, t)
        if xy is None:
            self._tracking = False
            self.filter.reset()
            return self.out

        mx, my = self.filter(*(self.map(xy) * UNITS), t)
        m = np.array([mx, my]) / UNITS

        if self.out is None:
            self.out = np.clip(m, 0.0, 1.0)
            self.offset = self.out - m
        elif not self._tracking:
            self.offset = self.out - m      # hand reacquired: keep the cursor where it was

        if rewind:
            target = t - self.rewind_s
            past = [p for (tt, p) in self._hist if tt <= target]
            if past:
                self.out = past[-1].copy()
            self._hold_until = t + self.settle_s

        self._tracking = True
        if freeze or t < self._hold_until:
            self.offset = self.out - m
        else:
            p = m + self.offset
            clipped = np.clip(p, 0.0, 1.0)
            self.offset += clipped - p      # edge re-sync
            self.out = clipped
        self._hist.append((t, self.out.copy()))
        return self.out
