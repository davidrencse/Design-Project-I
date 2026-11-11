"""Camera instructions plus drawing helpers for the legacy effects renderer.

`camera_guide` keeps the original BGR camera pixels and resolution, adding only
recording/calibration instructions. The older grayscale helpers remain for HandFX.
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

# ---- palette (grey levels, 0..255)
INK = 8
CHARCOAL = 34
GRAPHITE = 70
ASH = 116
SMOKE = 150
SILVER = 194
BONE = 234
AA = cv2.LINE_AA

# ---- fonts: first that exists wins, per weight
_FONT_DIRS = ("/usr/share/fonts/TTF", "/usr/share/fonts/noto", "/usr/share/fonts/truetype/jetbrains-mono",
              "/usr/share/fonts/truetype/dejavu", os.path.expanduser("~/.local/share/fonts"))
_FONT_FILES = {
    "thin": ("JetBrainsMonoNerdFont-Thin.ttf", "JetBrainsMono-Thin.ttf", "NotoSansMono-Thin.ttf",
             "NotoSansMono-ExtraLight.ttf"),
    "light": ("JetBrainsMonoNerdFont-Light.ttf", "JetBrainsMono-Light.ttf", "NotoSansMono-Light.ttf"),
    "regular": ("JetBrainsMonoNerdFont-Regular.ttf", "JetBrainsMono-Regular.ttf", "NotoSansMono-Regular.ttf",
                "DejaVuSansMono.ttf"),
    "bold": ("JetBrainsMonoNerdFont-SemiBold.ttf", "JetBrainsMono-SemiBold.ttf", "NotoSansMono-SemiBold.ttf",
             "DejaVuSansMono-Bold.ttf"),
}

try:                                   # Pillow is a declared dependency, but never crash without it
    from PIL import Image, ImageDraw, ImageFont
except ImportError:                    # pragma: no cover
    Image = ImageDraw = ImageFont = None


def _find(weight: str) -> str | None:
    for name in _FONT_FILES.get(weight, ()) + _FONT_FILES["regular"]:
        for d in _FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


@lru_cache(maxsize=32)
def _font(weight: str, px: int):
    if ImageFont is None:
        return None
    path = _find(weight)
    if path is None:
        return None
    try:
        return ImageFont.truetype(path, px)
    except OSError:                    # pragma: no cover
        return None


_SPRITES: dict[tuple, np.ndarray] = {}


def _sprite(s: str, px: int, weight: str, track: float) -> np.ndarray:
    """Alpha mask (uint8, h x w) of `s` with `track` px of extra spacing between glyphs."""
    key = (s, px, weight, track)
    m = _SPRITES.get(key)
    if m is not None:
        return m
    if len(_SPRITES) > 600:
        _SPRITES.clear()
    font = _font(weight, px)
    if font is not None:
        adv = font.getlength("M")
        w = int(np.ceil(len(s) * (adv + track))) + 2
        h = int(px * 1.45) + 2
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        for i, ch in enumerate(s):
            if ch != " ":
                d.text((i * (adv + track) + 1, 1), ch, fill=255, font=font)
        m = np.asarray(im)
    else:                              # Hershey fallback: still monochrome, still spaced
        scale = px / 22.0
        cw = int(px * 0.62 + track)
        w, h = len(s) * cw + 4, int(px * 1.4) + 2
        m = np.zeros((h, w), np.uint8)
        for i, ch in enumerate(s):
            cv2.putText(m, ch, (i * cw + 1, int(px * 1.05)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, AA)
    # trim empty rows/cols so anchors are exact
    ys, xs = np.nonzero(m)
    if len(ys):
        m = m[:, : xs.max() + 2]
    m = np.ascontiguousarray(m)
    _SPRITES[key] = m
    return m


def text_size(s: str, px: int = 12, weight: str = "regular", track: float = 2.0) -> tuple[int, int]:
    m = _sprite(s, px, weight, track)
    return m.shape[1], m.shape[0]


def put_text(img: np.ndarray, s: str, x: float, y: float, px: float = 12, v: int = BONE,
             weight: str = "regular", track: float = 2.0, anchor: str = "l", alpha: float = 1.0) -> int:
    """Blit `s` (x, y = top-left, or top-right/top-centre via anchor 'r'/'c') onto grey `img`.
    Returns the text width. `alpha` fades it, `v` is its grey level (can be darker than the ground)."""
    if alpha <= 0.01 or not s:
        return 0
    m = _sprite(s, max(6, int(round(px))), weight, round(track, 1))
    h, w = m.shape
    x = x - w if anchor == "r" else x - w // 2 if anchor == "c" else x
    x0, y0 = int(round(x)), int(round(y))
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if cx0 >= cx1 or cy0 >= cy1:
        return w
    reg = img[cy0:cy1, cx0:cx1]
    a = m[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0].astype(np.int32) * int(round(alpha * 256)) >> 8
    r16 = reg.astype(np.int32)
    reg[...] = np.clip(r16 + ((int(v) - r16) * a) // 255, 0, 255).astype(np.uint8)
    return w


# ---- line helpers (all take a grey level, never a colour)
def ip(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def line(img, a, b, v, th=1):
    cv2.line(img, ip(a), ip(b), int(v), th, AA)


def dashed(img, a, b, v, dash=6.0, gap=5.0, max_dashes=400):
    """Dashed line, clipped to the image first so off-screen length costs nothing."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or max(np.abs(a).max(), np.abs(b).max()) > 1e6:
        return
    d = b - a
    n = float(np.hypot(*d))
    if n < 1:
        return
    H, W = img.shape[:2]
    ok, p, q = cv2.clipLine((0, 0, W, H), ip(a), ip(b))
    if not ok:
        return
    u = d / n
    t0 = float(np.hypot(p[0] - a[0], p[1] - a[1]))         # visible part is [t0, t1] along a -> b
    t1 = min(n, float(np.hypot(q[0] - a[0], q[1] - a[1])))
    period = dash + gap
    t = (t0 // period) * period                             # keep the dash phase of the full line
    for _ in range(max_dashes):
        if t >= t1:
            break
        s0, s1 = max(t, t0), min(t + dash, t1)
        if s1 > s0:
            line(img, a + u * s0, a + u * s1, v)
        t += period


def cross(img, c, r, v):
    x, y = ip(c)
    cv2.line(img, (x - r, y), (x + r, y), int(v), 1, cv2.LINE_8)
    cv2.line(img, (x, y - r), (x, y + r), int(v), 1, cv2.LINE_8)


def square(img, c, r, v, fill=False):
    x, y = ip(c)
    cv2.rectangle(img, (x - r, y - r), (x + r, y + r), int(v), -1 if fill else 1, cv2.LINE_8)


def bracket(img, c, r, v, arm=None):
    """Four corner brackets around `c` (a reticle that never hides what it marks)."""
    x, y = ip(c)
    r = max(3, int(r))
    arm = max(2, int(arm if arm is not None else r * 0.55))
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = x + sx * r, y + sy * r
            cv2.line(img, (cx, cy), (cx - sx * arm, cy), int(v), 1, cv2.LINE_8)
            cv2.line(img, (cx, cy), (cx, cy - sy * arm), int(v), 1, cv2.LINE_8)


def hbar(img, x, y, w, frac, v, track=GRAPHITE):
    """Hairline progress bar: dim full-width track, bright filled part."""
    cv2.line(img, (int(x), int(y)), (int(x + w), int(y)), int(track), 1, cv2.LINE_8)
    if frac > 0:
        cv2.line(img, (int(x), int(y)), (int(x + w * min(1.0, frac)), int(y)), int(v), 1, cv2.LINE_8)


_SCRIMS: dict[tuple, np.ndarray] = {}


def darken(img, x0, y0, x1, y1, f=0.55, feather: float = 14.0):
    """Feathered scrim: multiply a soft-edged rectangle by f so labels read over a bright feed."""
    H, W = img.shape[:2]
    p = int(feather * 2)
    X0, Y0, X1, Y1 = int(x0) - p, int(y0) - p, int(x1) + p, int(y1) + p
    key = (X1 - X0, Y1 - Y0, round(f, 2), feather)
    mask = _SCRIMS.get(key)
    if mask is None:
        if len(_SCRIMS) > 40:
            _SCRIMS.clear()
        m = np.full((Y1 - Y0, X1 - X0), 255, np.uint8)
        m[p:-p, p:-p] = int(255 * f)
        mask = cv2.GaussianBlur(m, (0, 0), feather / 2)
        _SCRIMS[key] = mask
    cx0, cy0, cx1, cy1 = max(X0, 0), max(Y0, 0), min(X1, W), min(Y1, H)
    if cx1 > cx0 and cy1 > cy0:
        reg = img[cy0:cy1, cx0:cx1]
        cv2.multiply(reg, mask[cy0 - Y0: cy1 - Y0, cx0 - X0: cx1 - X0], dst=reg, scale=1 / 255.0)


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---- backdrop
class Backdrop:
    """Camera feed -> dark, contrasty grey plate with vignette, faint scanlines and film grain."""
    N_TILES = 4

    def __init__(self, level: float = 0.52):
        x = np.arange(256) / 255.0
        s = x ** 1.18                                          # deepen mids
        s = s + 0.18 * np.sin((s - 0.5) * np.pi) * (1 - np.abs(2 * s - 1))   # gentle S-curve
        self.lut = np.clip(s * 255 * level + 3, 0, 255).astype(np.uint8)
        self._shape = None
        self._tick = 0

    def _prepare(self, h: int, w: int) -> None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
        vig = np.clip(1.0 - 0.62 * np.clip(r - 0.45, 0, None) ** 1.6, 0.22, 1.0)
        vig *= 1.0 - 0.07 * ((yy.astype(np.int32) // 2) % 2)  # faint scanlines
        self.vig = np.clip(vig * 255, 0, 255).astype(np.uint8)
        rng = np.random.default_rng(11)
        self.grain = [np.clip(rng.normal(8, 4.2, (h, w)), 0, 20).astype(np.uint8) for _ in range(self.N_TILES)]
        self._shape = (h, w)

    def plate(self, frame: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
        """size = (w, h) of the output. frame None -> a black plate."""
        w, h = size
        if self._shape != (h, w):
            self._prepare(h, w)
        if frame is None:
            g = np.zeros((h, w), np.uint8)
        else:
            g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.LUT(g, self.lut)
            if g.shape[:2] != (h, w):
                g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        g = cv2.multiply(g, self.vig, scale=1 / 255.0)
        self._tick = (self._tick + 1) % self.N_TILES
        return cv2.addWeighted(g, 1.0, self.grain[self._tick], 1.0, -8.0)


# ---- chrome
def k_of(img) -> float:
    """Layout scale: 1.0 at a 1280-wide canvas."""
    return img.shape[1] / 1280.0


def chrome(img: np.ndarray, sub: str = "GESTURE INTERFACE / 07 ACTIONS") -> dict:
    """Inset hairline frame, corner crop marks, edge registration ticks, wordmark and sub-label.
    Returns the layout box: {'m': inset, 'k': scale}."""
    H, W = img.shape[:2]
    k = k_of(img)
    m = int(round(26 * k))
    c = int(round(22 * k))
    cv2.rectangle(img, (m, m), (W - m, H - m), GRAPHITE, 1, cv2.LINE_8)
    for sx, x in ((1, m), (-1, W - m)):
        for sy, y in ((1, m), (-1, H - m)):
            cv2.line(img, (x - sx * c // 2, y), (x + sx * c, y), SILVER, 1, cv2.LINE_8)   # crop marks
            cv2.line(img, (x, y - sy * c // 2), (x, y + sy * c), SILVER, 1, cv2.LINE_8)
    t = max(3, int(5 * k))
    for x, y in ((W // 2, m), (W // 2, H - m), (m, H // 2), (W - m, H // 2)):
        cross(img, (x, y), t, ASH)
    # sparse ruler along the bottom edge
    for i in range(1, 16):
        x = m + i * (W - 2 * m) // 16
        cv2.line(img, (x, H - m), (x, H - m - (int(5 * k) if i % 4 == 0 else int(2 * k))), ASH, 1, cv2.LINE_8)
    x0, y0 = m + int(16 * k), m + int(14 * k)
    put_text(img, "AIRPEN", x0, y0, 21 * k, BONE, "light", track=10 * k)
    put_text(img, sub, x0 + 1, y0 + 30 * k, 10.5 * k, SMOKE, "regular", track=2.6 * k)
    return {"m": m, "k": k}


def status_block(img: np.ndarray, rows: list[tuple[str, str]], state: str, live: bool) -> None:
    """Top-right live readouts: label left, value right, then state with a marker square."""
    H, W = img.shape[:2]
    k = k_of(img)
    m = int(round(26 * k))
    right = W - m - int(16 * k)
    left = right - int(176 * k)
    y = m + int(14 * k)
    darken(img, left - 14 * k, m + 1, W - m - 1, y + (len(rows) + 1) * 19 * k + 8 * k, 0.5, 12 * k)
    for lab, val in rows:
        put_text(img, lab, left, y, 10 * k, SMOKE, "regular", track=2.4 * k)
        put_text(img, val, right, y, 10.5 * k, BONE, "regular", track=2.4 * k, anchor="r")
        y += int(19 * k)
    y += int(3 * k)
    sq = max(2, int(3 * k))
    cv2.line(img, (int(left), int(y - 4 * k)), (int(right), int(y - 4 * k)), GRAPHITE, 1, cv2.LINE_8)
    square(img, (left + sq, y + 6 * k), sq, BONE if live else SMOKE, fill=live)
    put_text(img, state, right, y, 10.5 * k, BONE if live else SMOKE, "regular", track=2.4 * k, anchor="r")


def wrap(s: str, width: int) -> list[str]:
    words, lines, cur = s.upper().split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def guide_overlay(img: np.ndarray, step: str, total: str, text: str, status: str, ok: bool,
                  frac: float | None, hint: str = "") -> None:
    """Bottom-left instruction block shared by calibrate and record: step counter, wrapped
    instruction, status line and a hairline progress bar."""
    H, W = img.shape[:2]
    k = k_of(img)
    m = int(round(26 * k))
    x = m + int(16 * k)
    lines = wrap(text, 52)
    block_h = int((46 + 20 * len(lines) + 40) * k)
    y = H - m - int(12 * k) - block_h
    darken(img, m + 1, y - 12 * k, m + 600 * k, H - m - 1, 0.5, 16 * k)
    w = put_text(img, step, x, y, 11 * k, SMOKE, "regular", track=3 * k)
    put_text(img, total, x + w + 16 * k, y, 11 * k, ASH, "regular", track=3 * k)
    y += int(24 * k)
    for ln in lines:
        put_text(img, ln, x, y, 14 * k, BONE, "light", track=1.8 * k)
        y += int(21 * k)
    y += int(10 * k)
    sq = max(2, int(3 * k))
    square(img, (x + sq, y + 7 * k), sq, BONE if ok else SMOKE, fill=ok)
    put_text(img, status, x + 14 * k, y, 10.5 * k, BONE if ok else SMOKE, "regular", track=2.4 * k)
    if hint:
        put_text(img, hint, x + 14 * k, y + 18 * k, 9.5 * k, ASH, "regular", track=2.2 * k)
    if frac is not None:
        hbar(img, x, H - m - int(14 * k), 540 * k, frac, BONE)


def window_closed(name: str) -> bool:
    """True once the user has closed the window. With OpenCV's Qt backend, closing the last window
    makes getWindowProperty raise cv2.error instead of returning < 1, so treat that as closed too."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def camera_guide(frame: np.ndarray, lines: list[str], frac: float | None = None) -> np.ndarray:
    """Minimal recording instructions over the unfiltered color camera image."""
    import textwrap

    image = frame.copy()
    height, width = image.shape[:2]
    font_scale = min(0.55, max(0.3, width / 1280))
    char_width = cv2.getTextSize("M", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
    columns = max(8, (width - 24) // max(1, char_width))
    y = 24
    for line in lines:
        for row in textwrap.wrap(line, width=columns):
            (text_width, text_height), _ = cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(image, (10, y - text_height - 3),
                          (min(width - 1, text_width + 14), y + 5), (0, 0, 0), -1)
            cv2.putText(image, row, (12, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
    if frac is not None:
        end = round((width - 1) * float(np.clip(frac, 0, 1)))
        cv2.line(image, (0, height - 3), (end, height - 3), (80, 220, 100), 3)
    return image
