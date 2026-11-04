"""One Euro filter (Casiez, Roussel, Vogel, CHI 2012).

Low cutoff when the hand is still (kills jitter), higher cutoff when it moves fast (kills lag).
"""
from __future__ import annotations

import math


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t is None or self._x is None:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = max(t - self._t, 1e-4)
        self._t = t
        dx = (x - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        return self._x


class OneEuro2D:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.fx = OneEuro(min_cutoff, beta, d_cutoff)
        self.fy = OneEuro(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)
