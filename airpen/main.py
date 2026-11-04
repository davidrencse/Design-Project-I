"""AirPen: hand gestures from the webcam control the cursor system-wide.

Runs as a background process. The preview window is optional: closing it only hides the
preview, gesture control keeps running. Stop it with `./run.sh stop` (or SIGTERM / Ctrl+C).
Toggle the preview from anywhere with `./run.sh show` (SIGUSR1).
"""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from airpen.gestures import GestureEngine, Tuning
from airpen.fx import HandFX
from airpen.output import make_output
from airpen.pen import PenDetector
from airpen.pointer import Pointer
from airpen.tracker import Camera, HandTracker
from airpen.tuning_io import CONFIG, load_overrides, load_tuning

ROOT = Path(__file__).resolve().parent.parent
WINDOW = "AirPen"
RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
PIDFILE = RUNTIME / "airpen.pid"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="airpen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show", action="store_true", help="open the preview window at start")
    p.add_argument("--camera", default="0", help="camera index or device path (default 0)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dynamic-fps", action="store_true",
                   help="let the webcam lower its frame rate in dim light (default: lock it)")
    p.add_argument("--model", default=str(ROOT / "models" / "hand_landmarker.task"))
    p.add_argument("--backend", choices=("auto", "uinput", "pynput", "none"), default="auto")
    p.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   default=(0.25, 0.20, 0.75, 0.70), help="active zone in camera space (0..1)")
    p.add_argument("--invert-scroll", action="store_true")
    p.add_argument("--invert-swipe", action="store_true",
                   help="swipe right = next workspace (default: swipe left = next, like a touchpad)")
    p.add_argument("--ws-next", help="command for next workspace (default: hyprctl on Hyprland)")
    p.add_argument("--ws-prev", help="command for previous workspace")
    p.add_argument("--no-notify", action="store_true", help="no desktop notifications")
    p.add_argument("--unoq", metavar="HOST[:PORT]", help="optional: mirror state to an Arduino UNO Q")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (testing)")
    return p.parse_args(argv)


def single_instance():
    """Hold an exclusive lock on the pidfile for the life of the process."""
    fh = open(PIDFILE, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        print(f"[airpen] already running (pid {fh.read().strip() or '?'}). Use ./run.sh stop first.",
              file=sys.stderr)
        sys.exit(1)
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


class Notifier:
    def __init__(self, enabled: bool):
        self.cmd = shutil.which("notify-send") if enabled else None

    def __call__(self, title: str, body: str = "") -> None:
        if not self.cmd:
            return
        try:
            subprocess.Popen(
                [self.cmd, "-a", "AirPen", "-t", "1500",
                 "-h", "string:x-canonical-private-synchronous:airpen", title, body],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass


def place_window_on_screen() -> None:
    """Hyprland opens the window on whatever workspace AirPen was started from, tiled and
    tiny. Pull it to the workspace you're looking at, float it, size it, centre it, focus it."""
    if not (os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and shutil.which("hyprctl")):
        return
    try:
        import json as _json
        ws = _json.loads(subprocess.run(["hyprctl", "activeworkspace", "-j"], capture_output=True,
                                        text=True, timeout=2).stdout)["id"]
        mon = _json.loads(subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True,
                                         text=True, timeout=2).stdout)
        m = next((x for x in mon if x.get("focused")), mon[0])
        w = int(min(1280, m["width"] / m.get("scale", 1) * 0.6))
        h = w * 3 // 4
        sel = "title:^(AirPen)"
        cmds = [f"dispatch movetoworkspacesilent {ws},{sel}", f"dispatch setfloating {sel}",
                f"dispatch resizewindowpixel exact {w} {h},{sel}", f"dispatch focuswindow {sel}",
                "dispatch centerwindow"]
        subprocess.run(["hyprctl", "--batch", " ; ".join(cmds)], capture_output=True, timeout=3)
    except Exception as exc:  # cosmetic only; never let window placement kill gesture control
        print(f"[airpen] could not place preview window: {exc}", file=sys.stderr)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def open_camera(args, cam_id):
    while True:
        try:
            return Camera(cam_id, args.width, args.height, args.fps, lock_fps=not args.dynamic_fps)
        except RuntimeError as exc:
            print(f"[airpen] {exc}; retrying in 3 s", file=sys.stderr)
            time.sleep(3)


def main(argv=None) -> int:
    args = parse_args(argv)
    lock = single_instance()
    cam_id = int(args.camera) if args.camera.isdigit() else args.camera

    stop = threading.Event()
    toggle_preview = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGUSR1, lambda *_: toggle_preview.set())
    force_preview = threading.Event()
    signal.signal(signal.SIGUSR2, lambda *_: force_preview.set())   # always show (./run.sh)

    notify = Notifier(not args.no_notify)
    cam = open_camera(args, cam_id)
    tracker = HandTracker(args.model)
    engine = GestureEngine(load_tuning(Tuning()))
    overrides = load_overrides()
    pointer = Pointer(tuple(args.box), ref_scale=float(overrides.get("pointer_ref_scale", 0.11)))
    if overrides:
        print(f"[airpen] using {len(overrides)} personal thresholds from {CONFIG}")
    out = make_output(args.backend, ws_next=args.ws_next, ws_prev=args.ws_prev)

    board = led = None
    if args.unoq:
        from airpen.ledframe import LedRenderer
        from airpen.unoq import BoardLink

        board, led = BoardLink(args.unoq), LedRenderer()

    armed = out.name != "none"
    preview = args.show
    show_fx = True
    pen_det = PenDetector()
    fx = HandFX()
    window_open = False
    placed = False
    swipe_next = "right" if args.invert_swipe else "left"
    scroll_sign = -1.0 if args.invert_scroll else 1.0
    fps, last, seq, frames = 0.0, time.monotonic(), 0, 0
    last_hand_t = 0.0
    last_status = 0.0

    print(f"[airpen] pid {os.getpid()}  camera {cam.size[0]}x{cam.size[1]}  output={out.name}")
    notify("AirPen running", "Gesture mode. Hold a fist for mouse mode.")

    try:
        while not stop.is_set():
            try:
                seq, frame = cam.read(seq)
            except RuntimeError:
                # camera vanished (suspend/resume, unplug): drop buttons, reopen, carry on
                out.release_all()
                cam.release()
                cam, seq = open_camera(args, cam_id), 0
                continue
            t = time.monotonic()
            frames += 1
            # idle: no hand for a while -> look at only every 3rd frame to save battery
            if t - last_hand_t > 2.0 and frames % 3 and not preview:
                continue
            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame, t)
            if hands:
                last_hand_t = t
            fh, fw = frame.shape[:2]
            for hh in hands:                       # look for a white stylus in each hand
                sight = pen_det.detect(frame, hh.pts)
                hh.pen = sight
                hh.pen_tip_norm = None if sight is None else sight.tip / np.array([fw, fh])
            gf = engine.update(hands, t)
            cursor = pointer.update(gf.pointer_xy, t, freeze=gf.freeze, rewind=gf.rewind,
                                    hand_scale=gf.pointer_scale)

            for ev in gf.events:
                k = ev.kind
                if k not in ("scroll", "zoom"):
                    log(f"EVENT {k} {ev.hand} {ev.button}{ev.dir}  armed={armed}")
                elif frames % 15 == 0:
                    log(f"EVENT {k} dy={ev.dy:.0f} amount={ev.amount:.0f}  armed={armed}")
                if k == "pen_on":
                    notify("Pen mode", "Your grip moves the cursor. Tap the pen with your index finger = click, middle finger = right click. Let go of the pen to stop.")
                elif k == "cursor_on":
                    out.cursor_big(True)
                    notify("Mouse mode", "Move hand = cursor. Tap thumb+index = click, thumb+middle = right click. Hold a fist again to exit.")
                elif k == "cursor_off":
                    out.release_all()
                    out.cursor_big(False)
                    notify("Gesture mode", "Swipe = switch workspace. Spread / pinch thumb+index = zoom. Point up/down = scroll.")
                elif k == "press" and armed:
                    out.press(ev.button)
                    log(f"ACTION {ev.button} button down")
                elif k == "release":
                    out.release(ev.button)
                elif k == "scroll" and armed:
                    out.scroll(scroll_sign * ev.dy, scroll_sign * ev.dx)
                elif k == "zoom" and armed:
                    out.zoom(ev.amount)
                elif k == "zoom_end":
                    out.zoom_end()
                elif k == "swipe" and armed and ev.dir in ("left", "right"):
                    direction = "next" if ev.dir == swipe_next else "prev"
                    out.workspace(direction)
                    log(f"ACTION workspace {direction}")
                # (the old "both palms = pause" gesture is gone: it fired by accident and
                # silently blocked everything. Stop AirPen with ./run.sh stop instead.)
                if led is not None and k in ("press", "swipe", "cursor_on", "cursor_off"):
                    led.trigger("click" if k == "press" else k, t, dir=ev.dir)

            if hands and t - last_status > 1.0:
                last_status = t
                parts = []
                for hh in hands:
                    tr = gf.hands.get(hh.label)
                    if tr is not None and tr.m is not None:
                        m = tr.m
                        parts.append(f"{hh.label}:{tr.pose} bends={[int(b) for b in m.bends]} "
                                     f"ti={m.tap_index:.2f} tm={m.tap_middle:.2f} ts={m.thumb_side:.2f} "
                                     f"{m.distance_cm:.0f}cm v={tr.speed():.1f} pen={'YES' if getattr(hh, 'pen', None) else 'no'}")
                log(f"see {' | '.join(parts)}  mode={('PEN' if gf.pen_mode else 'MOUSE') if gf.cursor_on else 'gesture'} armed={armed}")
            if armed and gf.cursor_on and cursor is not None and not gf.freeze:
                out.move(float(cursor[0]), float(cursor[1]))
            if not armed:
                out.release_all()

            if board is not None:
                ledf = led.render(t, armed=armed, cursor_on=gf.cursor_on, cursor=cursor, hands=len(hands),
                                  left_down=any(tr.left.on for tr in gf.hands.values()),
                                  right_down=any(tr.right.on for tr in gf.hands.values()),
                                  scrolling=gf.scrolling, zooming=gf.zooming, zoom_level=0.0,
                                  clutch=gf.clutch)
                board.send(ledf, {"armed": armed, "cursor": gf.cursor_on, "hands": len(hands)})

            dt, last = t - last, t
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

            # ---- optional preview window; closing it never stops gesture control
            if toggle_preview.is_set():
                toggle_preview.clear()
                preview = not preview
            if force_preview.is_set():
                force_preview.clear()
                preview = True
                placed = False          # re-centre on the current workspace
            if preview:
                try:
                    if not window_open:
                        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(WINDOW, 1280, 960)
                        window_open = True
                        placed = False
                    mode = "MOUSE MODE" if gf.cursor_on else "GESTURE MODE"
                    title = (f"AirPen  -  {mode}  -  {'running' if armed else 'PAUSED'}  -  "
                             f"{len(hands)} hand(s)  -  {fps:.0f} fps   [l = linework on/off, q = hide]")
                    cv2.setWindowTitle(WINDOW, title)
                    cv2.imshow(WINDOW, fx.render(frame, hands, gf) if show_fx else frame)
                    key = cv2.waitKey(1) & 0xFF
                    if not placed:
                        place_window_on_screen()
                        placed = True
                    if key == ord("l"):
                        show_fx = not show_fx
                    elif key == ord("r"):
                        pointer.recenter()
                    visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
                    if key in (ord("q"), 27) or not visible:
                        preview = False
                except cv2.error:
                    # the user closed the window (or the GUI went away): just stop previewing,
                    # gesture control must keep running no matter what the window does
                    preview = False
            if not preview and window_open:
                try:
                    cv2.destroyWindow(WINDOW)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
                window_open = False

            if args.frames and frames >= args.frames:
                break
    finally:
        out.close()  # releases any held button / Ctrl before the virtual devices disappear
        cam.release()
        tracker.close()
        if board is not None:
            board.close()
        cv2.destroyAllWindows()
        try:
            PIDFILE.unlink()
        except OSError:
            pass
        lock.close()
        print(f"[airpen] stopped after {frames} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
