"""Validated gesture bindings and PC control; recognition never launches a shell."""
from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import threading
from pathlib import Path

from airpen.actions import ACTIONS
from airpen.pointer import Pointer
from airpen.gestures import Tuning
from airpen.tuning_io import CONFIG

BINDINGS = CONFIG.with_name("bindings.json")
DEFAULT_BINDINGS = {
    "index_tap": {"type": "click", "button": "left"},
    "middle_tap": {"type": "click", "button": "right"},
    "fist": {"type": "toggle_pointer"},
    "thumb_swipe_right": {"type": "workspace", "direction": "next"},
    "thumb_swipe_left": {"type": "workspace", "direction": "prev"},
    "peace": {"type": "toggle_enabled"},
    "thumb_tap": {"type": "hotkey", "keys": ["play_pause"]},
}
NAMED_KEYS = {"ctrl", "shift", "alt", "super", "left", "right", "up", "down", "space",
              "enter", "esc", "tab", "backspace", "play_pause", "volume_up", "volume_down", "mute"}


def load_bindings(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else BINDINGS
    bindings = {k: dict(v) for k, v in DEFAULT_BINDINGS.items()}
    if path.exists():
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) - set(ACTIONS.values()):
            raise ValueError("bindings must be an object keyed by the seven gesture names")
        bindings.update(data)
    for name, spec in bindings.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{name}: binding must be an object")
        kind = spec.get("type")
        valid = kind in {"none", "toggle_pointer", "toggle_enabled"}
        if kind == "click":
            valid = spec.get("button") in {"left", "right"}
        elif kind == "workspace":
            valid = spec.get("direction") in {"next", "prev"}
        elif kind == "scroll":
            v = spec.get("amount")
            valid = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) <= 1200
        elif kind == "hotkey":
            keys = spec.get("keys")
            valid = (isinstance(keys, list) and 0 < len(keys) <= 6
                     and all(isinstance(k, str) and (k in NAMED_KEYS or (len(k) == 1 and k.isascii() and k.isalnum())) for k in keys))
        elif kind == "command":
            argv = spec.get("argv")
            valid = (isinstance(argv, list) and bool(argv)
                     and all(isinstance(a, str) and a and "\0" not in a for a in argv))
        if not valid:
            raise ValueError(f"{name}: invalid {kind!r} binding")
    return bindings


class Controls:
    """Discrete controls with bounded command execution and distance-adaptive cursor motion."""
    def __init__(self, output, bindings: dict, box=None, invert_swipe=False, invert_scroll=False, tune=None):
        self.output, self.bindings = output, bindings
        self.tune = tune or Tuning()
        self.pointer = Pointer(ref_scale=self.tune.pointer_ref_scale, **({"box": tuple(box)} if box else {}))
        self.enabled, self.pointer_on = True, False
        self.invert_swipe, self.invert_scroll = invert_swipe, invert_scroll
        self._last: dict[str, float] = {}
        self._epoch = 0
        self._commands = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._worker = None
        self._process = None
        self._process_lock = threading.Lock()

    def _run_commands(self):
        while not self._stop.is_set():
            try:
                epoch, argv = self._commands.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._process_lock:
                    if self._stop.is_set():
                        return
                    if not self.enabled or epoch != self._epoch:
                        continue
                    self._process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    rc = self._process.wait(timeout=3)
                    if rc:
                        print(f"[airpen] command {argv[0]!r} exited {rc}", file=sys.stderr)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                    print(f"[airpen] command {argv[0]!r} timed out", file=sys.stderr)
            except OSError as exc:
                print(f"[airpen] command failed: {exc}", file=sys.stderr)
            finally:
                with self._process_lock:
                    self._process = None

    def dispatch(self, event) -> None:
        spec = self.bindings[event.name]
        kind = spec["type"]
        # Ignore repeated events from noisy transitions, while allowing deliberate double clicks.
        if event.t - self._last.get(event.name, -math.inf) < (0.15 if kind == "click" else 0.4):
            return
        self._last[event.name] = event.t
        if self.output.name == "none":
            return
        if kind == "toggle_enabled":
            with self._process_lock:
                self.enabled = not self.enabled
                self._epoch += 1
            self.output.release_all()
            self.pointer.update(None, event.t)
            print(f"[airpen] controls {'enabled' if self.enabled else 'paused'}", flush=True)
            return
        if not self.enabled:
            return
        if kind == "toggle_pointer":
            self.pointer_on = not self.pointer_on
            self.pointer.update(None, event.t)
            print(f"[airpen] pointer {'on' if self.pointer_on else 'off'}", flush=True)
        elif kind == "click":
            try:
                self.output.press(spec["button"])
            finally:
                self.output.release(spec["button"])
        elif kind == "workspace":
            direction = spec["direction"]
            if self.invert_swipe:
                direction = "prev" if direction == "next" else "next"
            self.output.workspace(direction)
        elif kind == "scroll":
            self.output.scroll(spec["amount"] * (-1 if self.invert_scroll else 1))
        elif kind == "hotkey":
            self.output.hotkey(spec["keys"])
        elif kind == "command":
            if self._worker is None:
                self._worker = threading.Thread(target=self._run_commands, name="actions", daemon=True)
                self._worker.start()
            try:
                self._commands.put_nowait((self._epoch, spec["argv"]))
            except queue.Full:
                print("[airpen] command queue full; dropping action", file=sys.stderr)

    def update(self, frame, t: float) -> None:
        m = frame.metrics
        active = self.enabled and self.pointer_on and frame.hand_seen and m is not None
        # Keep the cursor steady during click formation and while using fist/thumb controls.
        tapping = frame.debug.get("tap_ok", False) and m is not None and (
            m.tap_index < self.tune.tap_off * 1.1 or m.tap_middle < self.tune.right_off * 1.1)
        xy = self.pointer.update(m.anchor if active else None, t,
                                 freeze=frame.in_fist or tapping,
                                 rewind=any(e.id in (1, 2) for e in frame.events),
                                 hand_scale=m.scale_norm if active else None)
        if active and xy is not None:
            self.output.move(*xy)
        for event in frame.events:
            self.dispatch(event)

    def close(self):
        self._stop.set()
        try:
            with self._process_lock:
                if self._process is not None:
                    try:
                        self._process.kill()
                    except ProcessLookupError:
                        pass
            if self._worker is not None:
                self._worker.join(timeout=1)
        finally:
            self.output.close()
