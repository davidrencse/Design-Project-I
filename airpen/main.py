"""AirPen: seven hand actions from the webcam, recognized on your dominant hand.

  1 index tap   2 middle tap   3 fist   4 / 5 thumb swipe right / left (in a fist)
  6 peace       7 thumb tap (in a fist)

Each action is logged (`ACTION <id> <name> <hand>`), optionally shown as a desktop notification,
and mapped to configurable PC controls. The preview keeps the color camera with hand landmarks and action feedback.
Runs as a background process; the preview window is
optional: closing it only hides the preview. Stop it with `./run.sh stop` (or SIGTERM / Ctrl+C).
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

from airpen.actions import LABELS, make_engine
from airpen.controls import Controls, load_bindings
from airpen.fx import CameraOverlay
from airpen.output import make_output
from airpen.settings import load_hand, norm_hand
from airpen.tracker import Camera, HandTracker
from airpen.tuning_io import CONFIG, load_overrides

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
    p.add_argument("--hand", type=norm_hand, default=None, metavar="{left,right}",
                   help="your dominant hand; the other hand is ignored (default: saved setting, else right)")
    p.add_argument("--backend", choices=("auto", "uinput", "pynput", "none"), default="auto",
                   help="PC input backend; none recognizes without controlling the desktop")
    p.add_argument("--bindings", type=Path, help="gesture bindings JSON (default: ~/.config/airpen/bindings.json)")
    p.add_argument("--box", type=float, nargs=4, default=None, help=argparse.SUPPRESS)
    p.add_argument("--invert-scroll", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--invert-swipe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--ws-next", help=argparse.SUPPRESS)
    p.add_argument("--ws-prev", help=argparse.SUPPRESS)
    p.add_argument("--no-notify", action="store_true", help="no desktop notifications")
    p.add_argument("--unoq", metavar="HOST[:PORT]", help="optional: mirror state to an Arduino UNO Q")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (testing)")
    args = p.parse_args(argv)
    if min(args.width, args.height, args.fps) <= 0 or args.frames < 0:
        p.error("dimensions and fps must be positive; frames must be >= 0")
    if args.box is not None:
        x0, y0, x1, y1 = args.box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            p.error("--box must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    if args.hand is None:
        args.hand = load_hand()
    return args


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
        self._process = None

    def __call__(self, title: str, body: str = "") -> None:
        if not self.cmd:
            return
        if self._process is not None and self._process.poll() is None:
            return
        try:
            self._process = subprocess.Popen(
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


def open_camera(args, cam_id, stop: threading.Event):
    """Open the camera, retrying every 3 s. Returns None as soon as `stop` is set."""
    while not stop.is_set():
        try:
            return Camera(cam_id, args.width, args.height, args.fps, lock_fps=not args.dynamic_fps)
        except RuntimeError as exc:
            print(f"[airpen] {exc}; retrying in 3 s", file=sys.stderr)
            stop.wait(3)
    return None


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        bindings = load_bindings(args.bindings)
    except (OSError, ValueError) as exc:
        print(f"[airpen] invalid bindings: {exc}", file=sys.stderr)
        return 2
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
    cam = tracker = board = controls = None
    frames = 0
    try:
        cam = open_camera(args, cam_id, stop)
        if cam is None:
            return 0
        tracker = HandTracker(args.model)
        overrides = load_overrides()
        engine = make_engine(args.hand)
        output = make_output(args.backend, ws_next=args.ws_next, ws_prev=args.ws_prev)
        controls = Controls(output, bindings, args.box, args.invert_swipe, args.invert_scroll, engine.tune)
        print(f"[airpen] output={output.name}; bindings=" + ", ".join(f"{k}={v['type']}" for k, v in bindings.items()))
        if overrides:
            print(f"[airpen] using {len(overrides)} personal thresholds from {CONFIG}")

        board = led = None
        if args.unoq:
            from airpen.ledframe import LedRenderer
            from airpen.unoq import BoardLink

            board, led = BoardLink(args.unoq), LedRenderer()

        overlay = CameraOverlay()
        last_action = None
        preview = args.show
        window_open = False
        placed = False
        seq = 0
        last_hand_t = 0.0
        last_status = 0.0

        print(f"[airpen] pid {os.getpid()}  camera {cam.size[0]}x{cam.size[1]}  dominant hand={args.hand}")
        notify("AirPen running", f"Dominant hand: {args.hand}. Output: {output.name}.")

        while not stop.is_set():
            try:
                seq, frame = cam.read(seq)
            except RuntimeError:
                # camera vanished (suspend/resume, unplug): reopen, forget hand state, carry on
                cam.release()
                cam = None
                cam, seq = open_camera(args, cam_id, stop), 0
                if cam is None:
                    break
                engine.reset()
                controls.pointer.update(None, time.monotonic())
                controls.output.release_all()
                continue
            t = time.monotonic()
            frames += 1
            # idle: no hand for a while -> look at only every 3rd frame to save battery
            if (t - last_hand_t > 2.0 and frames % 3 and not preview
                    and not toggle_preview.is_set() and not force_preview.is_set()):
                if args.frames and frames >= args.frames:
                    break
                continue
            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame, t)
            if hands:
                last_hand_t = t
            af = engine.update(hands, t)
            try:
                controls.update(af, t)
            except Exception as exc:
                # Failed input must not leave a held key or button, or stop recognition.
                print(f"[airpen] PC control failed; disabling output: {exc}", file=sys.stderr)
                try:
                    controls.close()
                except Exception as cleanup_exc:
                    print(f"[airpen] input cleanup failed: {cleanup_exc}", file=sys.stderr)
                from airpen.output import NullOutput
                controls = Controls(NullOutput(), bindings, args.box)


            for ev in af.events:
                log(f"ACTION {ev.id} {ev.name} {ev.hand}")
                last_action = (ev.id, t)
                notify(f"AirPen: {LABELS[ev.id]}", f"Gesture {ev.id} | {ev.hand} hand")
                if led is not None:
                    if ev.id in (4, 5):
                        led.trigger("swipe", t, dir="right" if ev.id == 4 else "left")
                    else:
                        led.trigger("click", t)

            if hands and t - last_status > 1.0:
                last_status = t
                m = af.metrics
                extra = ""
                if m is not None:
                    extra = (f" bends={[int(b) for b in m.bends]} ti={m.tap_index:.2f} tm={m.tap_middle:.2f} "
                             f"lat={af.debug.get('thumb_lat', 0.0):+.2f} {m.distance_cm:.0f}cm")
                log(f"see hand={engine.hand} seen={af.hand_seen} pose={af.pose} in_fist={af.in_fist}{extra}")

            if board is not None:
                ledf = led.render(t, armed=True, cursor_on=False, cursor=None, hands=len(hands),
                                  left_down=False, right_down=False, scrolling=False, zooming=False,
                                  zoom_level=0.0, clutch=af.in_fist)
                board.send(ledf, {"armed": True, "cursor": False, "hands": len(hands)})

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
                    cv2.setWindowTitle(WINDOW, "AirPen")
                    image = overlay.render_safe(frame, hands, af, engine.hand, last_action, now=t)
                    cv2.imshow(WINDOW, image)
                    key = cv2.waitKey(1) & 0xFF
                    if not placed:
                        place_window_on_screen()
                        placed = True
                    visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
                    if key in (ord("q"), 27) or not visible:
                        preview = False
                except Exception as exc:
                    # the user closed the window (or the GUI went away): just stop previewing,
                    # gesture control must keep running no matter what the window does
                    if not isinstance(exc, cv2.error):
                        print(f"[airpen] preview window failed ({type(exc).__name__}: {exc})", file=sys.stderr)
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
        # Release every resource even when an input device vanished during shutdown.
        for resource, method in ((controls, "close"), (cam, "release"), (tracker, "close"), (board, "close")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as exc:
                    print(f"[airpen] cleanup {method} failed: {exc}", file=sys.stderr)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        try:
            PIDFILE.unlink()
        except OSError:
            pass
        lock.close()
        print(f"[airpen] stopped after {frames} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
