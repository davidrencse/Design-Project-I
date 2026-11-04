"""AirPen Matrix, Linux side of the UNO Q.

AirPen on the PC POSTs {"frame": [104 ints 0..255], "state": {...}} to http://<board>:7000/airpen.
We forward the frame to the STM32 sketch over the Router Bridge. If the PC goes quiet we
draw an idle animation so you can tell the board is alive but unlinked.
"""
import math
import threading
import time

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

ROWS, COLS = 8, 13
IDLE_AFTER_S = 2.0

ui = WebUI()
_lock = threading.Lock()   # Bridge calls come from the HTTP thread and the idle loop
_last_frame_at = 0.0
_frames = 0


def _draw(values) -> None:
    with _lock:
        Bridge.call("draw", bytes(values))


def on_frame(payload: dict):
    global _last_frame_at, _frames
    frame = payload.get("frame") if isinstance(payload, dict) else None
    if not isinstance(frame, list) or len(frame) != ROWS * COLS:
        return {"ok": False, "error": f"frame must be a list of {ROWS * COLS} values"}
    try:
        values = [max(0, min(255, int(v))) for v in frame]
    except (TypeError, ValueError):
        return {"ok": False, "error": "frame values must be numbers"}
    _draw(values)
    _last_frame_at = time.monotonic()
    _frames += 1
    return {"ok": True, "frames": _frames}


def status():
    return {"ok": True, "frames": _frames, "linked": time.monotonic() - _last_frame_at < IDLE_AFTER_S}


ui.expose_api("POST", "/airpen", on_frame)
ui.expose_api("GET", "/airpen/status", status)


def loop():
    """Idle animation: a soft pulse travelling around the border while no PC is linked."""
    if time.monotonic() - _last_frame_at < IDLE_AFTER_S:
        time.sleep(0.2)
        return
    t = time.monotonic()
    border = [(0, c) for c in range(COLS)] + [(r, COLS - 1) for r in range(1, ROWS)] \
        + [(ROWS - 1, c) for c in range(COLS - 2, -1, -1)] + [(r, 0) for r in range(ROWS - 2, 0, -1)]
    head = (t * 14) % len(border)
    values = [0] * (ROWS * COLS)
    for i, (r, c) in enumerate(border):
        d = (head - i) % len(border)
        values[r * COLS + c] = int(220 * math.exp(-d / 3.0)) if d < 10 else 0
    _draw(values)
    time.sleep(0.05)


App.run(user_loop=loop)
