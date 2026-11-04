"""End-to-end: the real airpen.main.main() loop with a fake camera + scripted hands, and a
recorder in place of the OS. If these pass, gestures really come out as OS actions."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_gestures import FIST, TUCKED, make_hand, rotate, tip  # noqa: E402

import airpen.main as M  # noqa: E402
from airpen.output import Output  # noqa: E402

FW, FH = 640, 480


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.frame = 0

    def monotonic(self):
        return self.t

    def perf_counter(self):
        return self.t

    @staticmethod
    def strftime(fmt):
        return "00:00:00 "

    @staticmethod
    def sleep(s):
        pass


class Recorder(Output):
    name = "rec"

    def __init__(self, **kw):
        super().__init__(ws_next="NEXT", ws_prev="PREV")
        self.log = []

    def _move(self, x, y): self.log.append(("move", round(x, 3), round(y, 3)))
    def _button(self, b, down): self.log.append(("btn", b, down))
    def _wheel(self, v, h): self.log.append(("wheel", v, h))
    def _ctrl_key(self, down): self.log.append(("ctrl", down))
    def workspace(self, direction): self.log.append(("ws", direction))


def hand(label="Right", at=(320, 400), s=0.7, **kw):
    """Synthetic hand in 640x480 frame coordinates."""
    h = make_hand(label, at=(0, 0), **kw)
    h.pts[:, :2] = h.pts[:, :2] * s + np.array(at)
    h.norm = h.pts[:, :2] / np.array([FW, FH])
    h.world = None
    return h


def run_main(script, monkeypatch, tmp_path, extra_args=()):
    """script: list of per-frame hand lists. Returns the Recorder."""
    clock = Clock()
    rec = Recorder()
    frames = list(script)

    class FakeCam:
        size = (FW, FH)

        def __init__(self, *a, **k):
            self.i = 0

        def read(self, seq, timeout=1.0):
            clock.t += 1 / 30
            clock.frame = seq          # tracker follows the CAMERA frame, even when AirPen skips frames
            return seq + 1, np.zeros((FH, FW, 3), np.uint8)

        def release(self):
            pass

    class FakeTracker:
        def __init__(self, *a, **k):
            self.i = 0

        def detect(self, frame, t):
            i = clock.frame
            return frames[i] if i < len(frames) else []

        def close(self):
            pass

    monkeypatch.setattr(M, "Camera", FakeCam)
    monkeypatch.setattr(M, "HandTracker", FakeTracker)
    monkeypatch.setattr(M, "make_output", lambda *a, **k: rec)
    monkeypatch.setattr(M, "time", clock)
    monkeypatch.setattr(M, "PIDFILE", tmp_path / "airpen.pid")
    monkeypatch.setattr(M, "load_overrides", lambda: {})          # built-in defaults, not ~/.config
    monkeypatch.setattr(M, "load_tuning", lambda t: t)
    monkeypatch.setattr(M.signal, "signal", lambda *a: None)       # pytest owns the signals
    M.main(["--no-notify", "--frames", str(len(frames)), *extra_args])
    return rec


def idle(n):
    return [[]] * n


def swipe_left(label="Right"):
    out = [[hand(label, at=(500 - 30 * i, 330))] for i in range(8)]
    back = [[hand(label, at=(260 + 30 * i, 330))] for i in range(8)]
    return out + back + [[hand(label, at=(500, 330))]] * 10


def fist_hold(label="Right"):
    return [[hand(label, extended=(), thumb_tip=FIST)]] * 34


def taps(finger, n=3):
    seq = []
    for _ in range(n):
        seq += [[hand(thumb_tip=tip(finger))]] * 4 + [[hand()]] * 4
    return seq


def test_full_session_produces_os_actions(monkeypatch, tmp_path):
    script = (
        idle(20)
        + swipe_left()                                                  # -> next workspace
        + fist_hold() + [[hand()]] * 4                                  # -> mouse mode
        + [[hand(at=(260 + 6 * i, 380 - 3 * i))] for i in range(25)]    # -> cursor moves
        + taps("index")                                                 # -> 3 left clicks
        + taps("middle", 2)                                             # -> 2 right clicks
        + [[hand(extended=("index",), thumb_tip=TUCKED)]] * 30          # point up -> scroll up
        + fist_hold() + [[hand()]] * 4                                  # -> back to gesture mode
        + [[hand(extended=("index",), thumb_tip=(tip("index")[0] - 20 - 8 * i, tip("index")[1]))]
           for i in range(12)]                                          # one-hand pinch: must NOT zoom
        + [[hand("Left", at=(260 - 12 * i, 380), extended=("index",)),
            hand("Right", at=(380 + 12 * i, 380), extended=("index",))] for i in range(14)]  # 2 hands apart -> zoom in
        + idle(15)
        + [[hand("Left", at=(180, 380)), hand("Right", at=(460, 380))]] * 60   # two palms held: must NOT pause
        + swipe_left()                                                  # still works afterwards
        + idle(10)
    )
    rec = run_main(script, monkeypatch, tmp_path)
    log = rec.log
    ws = [e for e in log if e[0] == "ws"]
    assert ws.count(("ws", "next")) >= 2, ws
    assert ("ws", "prev") not in ws, "return stroke undid the swipe"
    assert sum(1 for e in log if e[0] == "move") >= 10
    assert log.count(("btn", "left", True)) == 3 and log.count(("btn", "left", False)) == 3
    assert log.count(("btn", "right", True)) == 2 and log.count(("btn", "right", False)) == 2
    assert any(e[0] == "wheel" and e[1] > 0 for e in log), "point up did not scroll up"
    assert ("ctrl", True) in log and any(e[0] == "wheel" for e in log[log.index(("ctrl", True)):])
    assert not rec.held, "a button was left held down"


def test_point_down_scrolls_down_end_to_end(monkeypatch, tmp_path):
    down = rotate(hand(at=(320, 150), extended=("index",), thumb_tip=TUCKED), 180)
    rec = run_main(idle(5) + [[down]] * 30, monkeypatch, tmp_path)
    wheels = [e[1] for e in rec.log if e[0] == "wheel"]
    assert wheels and all(v < 0 for v in wheels)


def test_100k_random_frames_never_crash_stick_or_block(monkeypatch, tmp_path):
    rng = np.random.default_rng(1234)
    poses = [
        dict(),
        dict(extended=()),
        dict(extended=(), thumb_tip=FIST),
        dict(extended=("index",)),
        dict(extended=("index",), thumb_tip=TUCKED),
        dict(extended=("index", "middle")),
        dict(thumb_tip=tip("index")),
        dict(thumb_tip=tip("middle")),
    ]
    script = []
    pos = {"Left": np.array([200.0, 380.0]), "Right": np.array([440.0, 380.0])}
    cur = {"Left": 0, "Right": 0}
    while len(script) < 100_000:
        n = int(rng.integers(3, 40))             # hold a "situation" for a few frames
        present = [lab for lab in ("Left", "Right") if rng.random() < 0.6]
        for lab in present:
            if rng.random() < 0.5:
                cur[lab] = int(rng.integers(len(poses)))
        vel = {lab: rng.normal(0, 12, 2) for lab in present}
        for _ in range(n):
            frame = []
            for lab in present:
                pos[lab] = np.clip(pos[lab] + vel[lab] + rng.normal(0, 2, 2), [60, 120], [580, 470])
                if rng.random() < 0.03:
                    continue                      # tracking dropout
                h = hand(lab, at=tuple(pos[lab]), s=float(rng.uniform(0.5, 0.9)), **poses[cur[lab]])
                h.pts[:, :2] += rng.normal(0, 1.5, (21, 2))   # landmark jitter
                h.norm = h.pts[:, :2] / np.array([FW, FH])
                if rng.random() < 0.1:
                    h = rotate(h, float(rng.uniform(-60, 60)))
                frame.append(h)
            script.append(frame)
    script = script[:100_000] + idle(30)          # end with no hands: everything must release

    rec = run_main(script, monkeypatch, tmp_path)
    log = rec.log

    # buttons always balanced: never pressed twice in a row, never released when not held
    held = set()
    for e in log:
        if e[0] == "btn":
            _, b, down = e
            assert (b in held) != down, f"button {b} {'double press' if down else 'release without press'}"
            (held.add if down else held.discard)(b)
    assert not held and not rec.held, "button stuck down at the end"
    assert ("ctrl", True) not in log or log[::-1].index(("ctrl", False)) < log[::-1].index(("ctrl", True)), "Ctrl left held"
    kinds = {e[0] for e in log}
    assert {"move", "btn", "wheel", "ws"} <= kinds, f"some outputs never happened: {kinds}"
    assert all(0.0 <= e[1] <= 1.0 and 0.0 <= e[2] <= 1.0 for e in log if e[0] == "move")


def pen(at=(320, 400), s=0.7, **kw):
    from test_gestures import pen_hand
    h = pen_hand(at=(0, 0), **kw)
    h.pts[:, :2] = h.pts[:, :2] * s + np.array(at)
    h.norm = h.pts[:, :2] / np.array([FW, FH])
    h.world = None
    return h


def test_pen_session_end_to_end(monkeypatch, tmp_path):
    script = (
        idle(5)
        + [[pen()]] * 20                                                # pick up the pen -> pen mode
        + [[pen(at=(260 + 5 * i, 380 - 2 * i))] for i in range(30)]     # write / move -> cursor moves
        + ([[pen(lift_index=45)]] * 4 + [[pen()]] * 8) * 2              # 2 index taps -> 2 left clicks
        + [[pen(lift_middle=60)]] * 4 + [[pen()]] * 8                   # middle tap -> right click
        + [[hand()]] * 40                                               # put the pen down
        + idle(5)
    )
    rec = run_main(script, monkeypatch, tmp_path)
    log = rec.log
    assert sum(1 for e in log if e[0] == "move") >= 10
    assert log.count(("btn", "left", True)) == 2 and log.count(("btn", "left", False)) == 2
    assert log.count(("btn", "right", True)) == 1 and log.count(("btn", "right", False)) == 1
    assert not rec.held
