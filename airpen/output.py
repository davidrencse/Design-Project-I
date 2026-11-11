"""OS output: cursor, buttons, wheel, zoom and workspace switching.

Backends
  uinput  Linux (Wayland or X11). Creates a virtual absolute pointer + keyboard via /dev/uinput.
          Needs write access to /dev/uinput (be in the `input` group or add a udev rule).
  pynput  X11 / Windows / macOS fallback.
  none    Preview only; nothing leaves the app.

Workspace switching on Hyprland uses `hyprctl dispatch workspace e±1`. Elsewhere it sends a
key combo (Ctrl+Super+Left/Right by default), overridable with --ws-next / --ws-prev.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys

BUTTONS = ("left", "right")


def _spawn(cmd: str) -> None:
    import threading

    def run():
        try:
            r = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=3)
            out = (r.stdout + r.stderr).strip()
            if r.returncode != 0 or (out and out != "ok"):
                print(f"[airpen] {cmd!r} -> rc={r.returncode} {out[:200]}", flush=True)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[airpen] workspace command failed: {cmd!r}: {exc}", flush=True)

    threading.Thread(target=run, daemon=True).start()


def _hyprland_lua() -> bool:
    """Hyprland 0.56+ takes Lua in `hyprctl dispatch` (hl.dsp.*); older versions the classic syntax."""
    try:
        out = subprocess.run(["hyprctl", "version"], capture_output=True, text=True, timeout=2).stdout
        import re
        m = re.search(r"Hyprland (\d+)\.(\d+)", out)
        return bool(m) and (int(m.group(1)), int(m.group(2))) >= (0, 56)
    except (OSError, subprocess.SubprocessError):
        return False


def default_workspace_cmds() -> tuple[str | None, str | None]:
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("hyprctl"):
        if _hyprland_lua():
            return ("""hyprctl dispatch 'hl.dsp.focus({ workspace = "e+1" })'""",
                    """hyprctl dispatch 'hl.dsp.focus({ workspace = "e-1" })'""")
        return "hyprctl dispatch workspace e+1", "hyprctl dispatch workspace e-1"
    return None, None  # fall back to the key combo


class Output:
    name = "none"

    def __init__(self, ws_next: str | None = None, ws_prev: str | None = None):
        auto_next, auto_prev = default_workspace_cmds()
        self.ws_next = ws_next or auto_next
        self.ws_prev = ws_prev or auto_prev
        self.held: set[str] = set()
        self._ctrl = False

    # subclasses implement these primitives
    def _move(self, x: float, y: float) -> None: ...
    def _button(self, button: str, down: bool) -> None: ...
    def _wheel(self, v_hi: int, h_hi: int) -> None: ...
    def _ctrl_key(self, down: bool) -> None: ...
    def _combo(self, direction: str) -> None: ...
    def hotkey(self, keys: list[str]) -> None: ...

    # public API (safe to call in any state)
    def move(self, x: float, y: float) -> None:
        self._move(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))

    def press(self, button: str) -> None:
        if button not in self.held:
            self.held.add(button)
            self._button(button, True)

    def release(self, button: str) -> None:
        if button in self.held:
            self.held.discard(button)
            self._button(button, False)

    def scroll(self, v_hi: float, h_hi: float = 0.0) -> None:
        """Hi-res wheel units: 120 = one notch. +v scrolls up, +h scrolls right."""
        self._acc_v = getattr(self, "_acc_v", 0.0) + v_hi
        self._acc_h = getattr(self, "_acc_h", 0.0) + h_hi
        v, h = int(self._acc_v), int(self._acc_h)
        self._acc_v -= v
        self._acc_h -= h
        if v or h:
            self._wheel(v, h)

    def zoom_begin(self) -> None:
        if not self._ctrl:
            self._ctrl = True
            self._ctrl_key(True)

    def zoom(self, amount: float) -> None:
        """Ctrl + whole wheel notches. Browsers zoom one full step per wheel event, so
        sending tiny hi-res deltas would zoom erratically; accumulate to 120 first."""
        import time

        self._zacc = getattr(self, "_zacc", 0.0) + amount
        now = time.monotonic()
        while abs(self._zacc) >= 120 and now - getattr(self, "_zlast", 0.0) >= 0.06:
            step = 120 if self._zacc > 0 else -120
            self.zoom_begin()
            self._wheel(step, 0)
            self._zacc -= step
            self._zlast = now

    def zoom_end(self) -> None:
        self._zacc = 0.0
        if self._ctrl:
            self._ctrl = False
            self._ctrl_key(False)

    def workspace(self, direction: str) -> None:
        cmd = self.ws_next if direction == "next" else self.ws_prev
        if cmd:
            _spawn(cmd)
        else:
            self._combo(direction)

    def cursor_big(self, big: bool) -> None:
        """Enlarge the system cursor while AirPen drives it, so you can always see it."""
        if not (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("hyprctl")):
            return
        if getattr(self, "_big", False) == big:
            return
        self._big = big
        theme = os.environ.get("XCURSOR_THEME") or "breeze_cursors"
        normal = os.environ.get("XCURSOR_SIZE") or "24"
        _spawn(f"hyprctl setcursor {theme} {64 if big else normal}")

    def release_all(self) -> None:
        error = None
        for b in list(self.held):
            try:
                self.release(b)
            except Exception as exc:
                error = exc
        try:
            self.zoom_end()
        finally:
            if error is not None:
                raise error

    def close(self) -> None:
        self.release_all()
        if getattr(self, "_big", False):
            theme = os.environ.get("XCURSOR_THEME") or "breeze_cursors"
            subprocess.run(["hyprctl", "setcursor", theme, os.environ.get("XCURSOR_SIZE") or "24"],
                           capture_output=True, timeout=2)
            self._big = False


class NullOutput(Output):
    name = "none"

    def workspace(self, direction: str) -> None:
        pass


class UInputOutput(Output):
    name = "uinput"
    RANGE = 65535

    def __init__(self, **kw):
        super().__init__(**kw)
        from evdev import AbsInfo, UInput
        from evdev import ecodes as e

        self.e = e
        # Shaped like QEMU's USB tablet: libinput treats it as an absolute pointer, so the
        # cursor lands exactly where the hand points (no acceleration curve).
        self.ptr = UInput(
            {
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
                e.EV_ABS: [
                    (e.ABS_X, AbsInfo(0, 0, self.RANGE, 0, 0, 0)),
                    (e.ABS_Y, AbsInfo(0, 0, self.RANGE, 0, 0, 0)),
                ],
                e.EV_REL: [e.REL_WHEEL, e.REL_HWHEEL, e.REL_WHEEL_HI_RES, e.REL_HWHEEL_HI_RES],
            },
            name="AirPen virtual pointer", vendor=0x1209, product=0xA1E0, version=1,
        )
        try:
            self.kbd = UInput(
                {e.EV_KEY: list(set(self._key_codes().values()))},
                name="AirPen virtual keyboard", vendor=0x1209, product=0xA1E1, version=1,
            )
        except BaseException:
            self.ptr.close()
            raise
        self._lo_v = 0.0
        self._lo_h = 0.0

    def _key_codes(self) -> dict[str, int]:
        e = self.e
        names = {"ctrl": "LEFTCTRL", "shift": "LEFTSHIFT", "alt": "LEFTALT", "super": "LEFTMETA",
                 "left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN", "space": "SPACE",
                 "enter": "ENTER", "esc": "ESC", "tab": "TAB", "backspace": "BACKSPACE",
                 "play_pause": "PLAYPAUSE", "volume_up": "VOLUMEUP", "volume_down": "VOLUMEDOWN", "mute": "MUTE"}
        names.update({c: c.upper() for c in "abcdefghijklmnopqrstuvwxyz0123456789"})
        return {k: getattr(e, "KEY_" + v) for k, v in names.items()}

    def hotkey(self, keys: list[str]) -> None:
        codes = self._key_codes()
        pressed = []
        try:
            for key in keys:
                code = codes[key.lower()]
                pressed.append(code)
                self._key(code, True)
        finally:
            error = None
            for code in reversed(pressed):
                try:
                    self._key(code, False)
                except Exception as exc:
                    error = exc
            if error is not None:
                raise error

    def _move(self, x: float, y: float) -> None:
        e = self.e
        self.ptr.write(e.EV_ABS, e.ABS_X, round(x * self.RANGE))
        self.ptr.write(e.EV_ABS, e.ABS_Y, round(y * self.RANGE))
        self.ptr.syn()

    def _button(self, button: str, down: bool) -> None:
        code = self.e.BTN_LEFT if button == "left" else self.e.BTN_RIGHT
        self.ptr.write(self.e.EV_KEY, code, 1 if down else 0)
        self.ptr.syn()

    def _wheel(self, v_hi: int, h_hi: int) -> None:
        e = self.e
        # send hi-res for smooth-scrolling apps plus legacy notches for everything else
        if v_hi:
            self.ptr.write(e.EV_REL, e.REL_WHEEL_HI_RES, v_hi)
            self._lo_v += v_hi / 120.0
            n = int(self._lo_v)
            if n:
                self.ptr.write(e.EV_REL, e.REL_WHEEL, n)
                self._lo_v -= n
        if h_hi:
            self.ptr.write(e.EV_REL, e.REL_HWHEEL_HI_RES, h_hi)
            self._lo_h += h_hi / 120.0
            n = int(self._lo_h)
            if n:
                self.ptr.write(e.EV_REL, e.REL_HWHEEL, n)
                self._lo_h -= n
        self.ptr.syn()

    def _key(self, code: int, down: bool) -> None:
        self.kbd.write(self.e.EV_KEY, code, 1 if down else 0)
        self.kbd.syn()

    def _ctrl_key(self, down: bool) -> None:
        self._key(self.e.KEY_LEFTCTRL, down)

    def _combo(self, direction: str) -> None:
        e = self.e
        arrow = e.KEY_RIGHT if direction == "next" else e.KEY_LEFT
        for k in (e.KEY_LEFTCTRL, e.KEY_LEFTMETA, arrow):
            self._key(k, True)
        for k in (arrow, e.KEY_LEFTMETA, e.KEY_LEFTCTRL):
            self._key(k, False)

    def close(self) -> None:
        try:
            super().close()
        finally:
            try:
                self.ptr.close()
            finally:
                self.kbd.close()


class PynputOutput(Output):
    name = "pynput"

    def __init__(self, **kw):
        super().__init__(**kw)
        from pynput import keyboard, mouse

        self._m = mouse.Controller()
        self._k = keyboard.Controller()
        self._Button, self._Key = mouse.Button, keyboard.Key
        self.size = self._screen_size()

    @staticmethod
    def _screen_size() -> tuple[int, int]:
        try:
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            size = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return size
        except Exception:
            return 1920, 1080

    def hotkey(self, keys: list[str]) -> None:
        aliases = {"super": "cmd", "play_pause": "media_play_pause", "volume_up": "media_volume_up",
                   "volume_down": "media_volume_down", "mute": "media_volume_mute"}
        pressed = []
        try:
            for name in keys:
                key = name if len(name) == 1 else getattr(self._Key, aliases.get(name, name))
                pressed.append(key)
                self._k.press(key)
        finally:
            error = None
            for key in reversed(pressed):
                try:
                    self._k.release(key)
                except Exception as exc:
                    error = exc
            if error is not None:
                raise error

    def _move(self, x: float, y: float) -> None:
        self._m.position = (round(x * (self.size[0] - 1)), round(y * (self.size[1] - 1)))

    def _button(self, button: str, down: bool) -> None:
        b = self._Button.left if button == "left" else self._Button.right
        (self._m.press if down else self._m.release)(b)

    def _wheel(self, v_hi: int, h_hi: int) -> None:
        self._m.scroll(h_hi / 120.0, v_hi / 120.0)

    def _ctrl_key(self, down: bool) -> None:
        (self._k.press if down else self._k.release)(self._Key.ctrl)

    def _combo(self, direction: str) -> None:
        arrow = self._Key.right if direction == "next" else self._Key.left
        with self._k.pressed(self._Key.ctrl, self._Key.cmd):
            self._k.tap(arrow)


def make_output(backend: str = "auto", **kw) -> Output:
    if backend not in {"auto", "none", "uinput", "pynput"}:
        raise ValueError(f"unknown output backend: {backend}")
    if backend == "none":
        return NullOutput(**kw)
    if backend in ("auto", "uinput") and sys.platform.startswith("linux"):
        try:
            return UInputOutput(**kw)
        except Exception as exc:
            if backend == "uinput":
                raise
            print(f"[airpen] uinput unavailable ({exc})", file=sys.stderr)
            if os.environ.get("WAYLAND_DISPLAY"):
                print("[airpen] note: pynput cannot move the cursor on Wayland. "
                      "Give yourself /dev/uinput access: sudo usermod -aG input $USER, then log in again.",
                      file=sys.stderr)
    if backend == "auto" and os.environ.get("WAYLAND_DISPLAY"):
        print("[airpen] native Wayland input unavailable; recognition only", file=sys.stderr)
        return NullOutput(**kw)
    if backend == "uinput":
        raise RuntimeError("uinput requires Linux")
    try:
        return PynputOutput(**kw)
    except Exception as exc:
        if backend == "pynput":
            raise
        print(f"[airpen] pynput unavailable ({exc}); preview only", file=sys.stderr)
        return NullOutput(**kw)
