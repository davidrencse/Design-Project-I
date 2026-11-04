"""Renders AirPen state into an 8x13 grayscale frame for the Arduino UNO Q LED matrix.

The PC computes the frame and the board only displays it, so the on-screen mirror in the
HUD and the physical matrix always show the same thing.
"""
from __future__ import annotations

import math

import numpy as np

ROWS, COLS = 8, 13


def _splat(buf: np.ndarray, x: float, y: float, value: float, radius: float = 1.1) -> None:
    """Draw a soft dot at fractional (col=x, row=y)."""
    ys, xs = np.mgrid[0:ROWS, 0:COLS]
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    np.maximum(buf, value * np.exp(-d2 / (2 * (radius / 1.6) ** 2)), out=buf)


def _chevron(direction: str, shift: int) -> np.ndarray:
    g = np.zeros((ROWS, COLS))
    if direction in ("left", "right"):
        for r in range(ROWS):
            c = abs(r - 3.5) - 0.5          # '<' shape
            c = int(round(c)) + shift
            for thick in (0, 1):
                cc = c + thick
                if direction == "right":
                    cc = COLS - 1 - cc
                if 0 <= cc < COLS:
                    g[r, cc] = 255
    else:
        for c in range(COLS):
            r = int(round(abs(c - 6) * 0.6)) + shift
            for thick in (0, 1):
                rr = r + thick
                if direction == "down":
                    rr = ROWS - 1 - rr
                if 0 <= rr < ROWS:
                    g[rr, c] = 255
    return g


class LedRenderer:
    def __init__(self):
        self.buf = np.zeros((ROWS, COLS))
        self._t = None
        self._anims: list[tuple[str, float, dict]] = []

    def trigger(self, kind: str, t: float, **data) -> None:
        """kind: click, rclick, swipe(dir=...), cursor_on, cursor_off, arm, disarm"""
        self._anims.append((kind, t, data))

    def render(self, t: float, *, armed: bool, cursor_on: bool, cursor: np.ndarray | None,
               hands: int, left_down: bool, right_down: bool, scrolling: bool,
               zooming: bool, zoom_level: float, clutch: bool) -> np.ndarray:
        dt = 0.033 if self._t is None else max(1e-3, t - self._t)
        self._t = t
        self.buf *= math.exp(-dt / 0.10)   # trail fade
        layer = np.zeros((ROWS, COLS))

        if not armed:
            # slow red-alert style blink of the border
            if int(t * 2) % 2 == 0:
                layer[0, :] = layer[-1, :] = layer[:, 0] = layer[:, -1] = 90
        elif hands == 0:
            b = 0.5 + 0.5 * math.sin(t * 2.4)
            _splat(layer, 6, 3.5, 30 + 110 * b, radius=1.6)
        elif zooming:
            r = 1.0 + min(max(zoom_level, 0.0), 5.0)
            for c in range(COLS):
                for rr in range(ROWS):
                    if abs(max(abs(c - 6) / 1.4, abs(rr - 3.5)) - r) < 0.55:
                        layer[rr, c] = 220
        elif cursor_on and clutch:
            layer[2:6, 4] = layer[2:6, 5] = 200
            layer[2:6, 7] = layer[2:6, 8] = 200
        elif cursor_on and cursor is not None:
            _splat(self.buf, cursor[0] * (COLS - 1), cursor[1] * (ROWS - 1), 255)
        elif not cursor_on:
            # idle "radar" sweep while waiting for a gesture
            col = int((t * 8) % COLS)
            layer[:, col] = 40

        if scrolling:
            phase = int(t * 12) % 3
            layer[phase::3, 6] = 160
        if left_down:
            layer[:, 0] = 255
        if right_down:
            layer[:, -1] = 255
        if armed and cursor_on:
            layer[0, 6] = max(layer[0, 6], 60)  # "cursor live" pip

        keep = []
        for kind, t0, data in self._anims:
            age = t - t0
            if kind in ("click", "rclick") and age < 0.30:
                cx = cursor[0] * (COLS - 1) if cursor is not None else 6
                cy = cursor[1] * (ROWS - 1) if cursor is not None else 3.5
                rad = 0.5 + age * 18
                ys, xs = np.mgrid[0:ROWS, 0:COLS]
                ring = np.abs(np.hypot(xs - cx, ys - cy) - rad) < 0.7
                layer[ring] = np.maximum(layer[ring], 255 * (1 - age / 0.30))
                keep.append((kind, t0, data))
            elif kind == "swipe" and age < 0.45:
                shift = int(age / 0.45 * 6)
                np.maximum(layer, _chevron(data.get("dir", "right"), shift) * (1 - age / 0.45), out=layer)
                keep.append((kind, t0, data))
            elif kind in ("cursor_on", "cursor_off", "arm", "disarm") and age < 0.35:
                level = 255 * (1 - age / 0.35) if kind in ("cursor_on", "arm") else 120 * (1 - age / 0.35)
                layer = np.maximum(layer, level)
                keep.append((kind, t0, data))
        self._anims = keep

        return np.clip(np.maximum(self.buf, layer), 0, 255).astype(np.uint8)
