"""End-to-end: the real airpen.main.main() loop with a fake camera and scripted synthetic hands.
Actions come out as `ACTION <id> <name> <hand>` log lines (nothing touches the OS)."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_actions import fist_settled, mk, swipe_seq  # noqa: E402

import airpen.main as M  # noqa: E402
from airpen.actions import ACTIONS, ActionEngine  # noqa: E402
from airpen.fx import HandFX  # noqa: E402

FW, FH = 640, 480
ACTION_RE = re.compile(r"^ACTION (\d) (\w+) (Left|Right)$")


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
        return ""

    @staticmethod
    def sleep(s):
        pass


def run_main(script, monkeypatch, tmp_path, capsys, extra_args=(), fail_read_at=None, frames=None):
    """script: list of per-frame hand lists. Returns the parsed ACTION log lines [(id, name, hand)]."""
    clock = Clock()
    script = [f if isinstance(f, list) else [f] for f in script]
    reads = {"n": 0, "opened": 0}

    class FakeCam:
        size = (FW, FH)

        def __init__(self, *a, **k):
            reads["opened"] += 1

        def read(self, seq, timeout=1.0):
            reads["n"] += 1
            if fail_read_at is not None and reads["n"] == fail_read_at:
                raise RuntimeError("camera stopped delivering frames")
            clock.t += 1 / 30
            clock.frame = seq          # tracker follows the CAMERA frame, even when AirPen skips frames
            return seq + 1, np.zeros((FH, FW, 3), np.uint8)

        def release(self):
            pass

    class FakeTracker:
        def __init__(self, *a, **k):
            pass

        def detect(self, frame, t):
            i = clock.frame
            return script[i] if i < len(script) else []

        def close(self):
            pass

    from airpen.output import NullOutput
    monkeypatch.setattr(M, "make_output", lambda *a, **k: NullOutput())
    monkeypatch.setattr(M, "Camera", FakeCam)
    monkeypatch.setattr(M, "HandTracker", FakeTracker)
    monkeypatch.setattr(M, "time", clock)
    monkeypatch.setattr(M, "PIDFILE", tmp_path / "airpen.pid")
    monkeypatch.setattr(M, "load_overrides", lambda: {})          # built-in defaults, not ~/.config
    monkeypatch.setattr("airpen.tuning_io.load_overrides", lambda: {})
    monkeypatch.setattr(M, "load_hand", lambda: "Right")
    monkeypatch.setattr(M.signal, "signal", lambda *a: None)       # pytest owns the signals
    capsys.readouterr()
    M.main(["--backend", "none", "--no-notify", "--frames", str(frames or len(script)), *extra_args])
    out = capsys.readouterr().out
    run_main.reopened = reads["opened"]
    run_main.reads = reads["n"]
    return [(int(m.group(1)), m.group(2), m.group(3))
            for m in (ACTION_RE.match(ln.strip()) for ln in out.splitlines()) if m]


def idle(n):
    return [[]] * n


def tap(pose, n=4):
    return [mk(pose=pose)] * n + [mk()] * 5


def back(dx):
    return [mk(pose="fist", thumb=(dx * (1 - (i + 1) / 5), 0)) for i in range(5)]


def session(label="Right"):
    """All seven actions in one go: 1, 2, 3, 4, 5, 7, 6."""
    up = [mk(label, pose="fist", lift=0.5)] * 4
    rest = [mk(label, pose="fist")]
    L = {"label": label}
    return (idle(20) + [mk(label)] * 10
            + [mk(label, pose="tap_index")] * 4 + [mk(label)] * 5                      # 1
            + [mk(label, pose="tap_middle")] * 4 + [mk(label)] * 10                    # 2
            + fist_settled(25, **L)                                                    # 3
            + swipe_seq(+40, **L) + back(+40) + fist_settled(30, **L)                  # 4 (+ return stroke)
            + swipe_seq(-40, **L) + back(-40) + fist_settled(30, **L)                  # 5
            + rest * 6 + up + rest * 10                                                # 7
            + [mk(label)] * 15 + [mk(label, pose="peace")] * 15 + [mk(label)] * 10    # 6
            + idle(10))


EXPECT = [(1, "index_tap"), (2, "middle_tap"), (3, "fist"), (4, "thumb_swipe_right"),
          (5, "thumb_swipe_left"), (7, "thumb_tap"), (6, "peace")]


@pytest.mark.parametrize("hand", ["Right", "Left"])
def test_all_seven_actions_end_to_end(monkeypatch, tmp_path, capsys, hand):
    got = run_main(session(hand), monkeypatch, tmp_path, capsys, ["--hand", hand.lower()])
    assert got == [(i, n, hand) for i, n in EXPECT], got


def test_default_hand_comes_from_settings(monkeypatch, tmp_path, capsys):
    script = session("Left")
    monkeypatch.setattr(M, "load_hand", lambda: "Left")
    # run_main re-patches load_hand to Right, so patch the parse step instead
    real = M.parse_args

    def parse(argv=None):
        monkeypatch.setattr(M, "load_hand", lambda: "Left")
        return real(argv)

    monkeypatch.setattr(M, "parse_args", parse)
    got = run_main(script, monkeypatch, tmp_path, capsys)
    assert [g[0] for g in got] == [i for i, _ in EXPECT] and {g[2] for g in got} == {"Left"}


def test_non_dominant_hand_is_ignored(monkeypatch, tmp_path, capsys):
    assert run_main(session("Left"), monkeypatch, tmp_path, capsys, ["--hand", "right"]) == []
    assert run_main(session("Right"), monkeypatch, tmp_path, capsys, ["--hand", "left"]) == []


def test_other_hand_in_view_does_not_disturb(monkeypatch, tmp_path, capsys):
    left = mk("Left", at=(300, 600), pose="peace")
    script = (idle(20) + [[mk("Right"), left]] * 10 + [[mk("Right", pose="tap_index"), left]] * 4
              + [[mk("Right"), left]] * 10)
    got = run_main(script, monkeypatch, tmp_path, capsys, ["--hand", "right"])
    assert got == [(1, "index_tap", "Right")]


def test_old_flags_are_still_accepted(monkeypatch, tmp_path, capsys):
    got = run_main(session(), monkeypatch, tmp_path, capsys,
                   ["--backend", "uinput", "--invert-scroll", "--invert-swipe", "--box", "0", "0", "1", "1",
                    "--ws-next", "x", "--ws-prev", "y"])
    assert [g[0] for g in got] == [i for i, _ in EXPECT]


def test_camera_reopen_keeps_working(monkeypatch, tmp_path, capsys):
    script = session()
    got = run_main(script, monkeypatch, tmp_path, capsys, fail_read_at=5, frames=len(script) + 4)
    assert run_main.reopened == 2
    assert [g[0] for g in got] == [i for i, _ in EXPECT]


def test_no_hand_no_actions(monkeypatch, tmp_path, capsys):
    assert run_main(idle(100), monkeypatch, tmp_path, capsys) == []


def test_random_frames_never_crash_and_only_valid_actions(monkeypatch, tmp_path, capsys):
    rng = np.random.default_rng(1234)
    poses = ["open", "fist", "peace", "tap_index", "tap_middle"]
    script, pos = [], {"Left": np.array([300.0, 500.0]), "Right": np.array([800.0, 500.0])}
    cur = {"Left": 0, "Right": 0}
    while len(script) < 6000:
        present = [lab for lab in ("Left", "Right") if rng.random() < 0.7]
        for lab in present:
            if rng.random() < 0.6:
                cur[lab] = int(rng.integers(len(poses)))
        vel = {lab: rng.normal(0, 6, 2) for lab in present}
        for _ in range(int(rng.integers(3, 40))):
            frame = []
            for lab in present:
                pos[lab] = np.clip(pos[lab] + vel[lab], [100, 300], [1100, 650])
                if rng.random() < 0.03:
                    continue
                th = tuple(rng.normal(0, 10 if rng.random() < 0.1 else 0.8, 2))
                frame.append(mk(lab, at=tuple(pos[lab]), pose=poses[cur[lab]], thumb=th,
                                lift=float(rng.choice([0, 0, 0.5])), world=bool(rng.random() < 0.8)))
            script.append(frame)
    got = run_main(script + idle(30), monkeypatch, tmp_path, capsys, ["--hand", "right"])
    assert got, "random session produced no action at all"
    assert all(g[0] in ACTIONS and ACTIONS[g[0]] == g[1] and g[2] == "Right" for g in got)
    ids = [g[0] for g in got]
    thumb = [i for i, a in enumerate(ids) if a in (4, 5, 7)]
    if thumb:
        assert 3 in ids[:thumb[0]], "thumb action without a fist before it"


def test_preview_renders_last_action_and_dims_the_other_hand():
    fx = HandFX()
    eng = ActionEngine("Right")
    right, left = mk("Right", at=(300, 300)), mk("Left", at=(900, 300))
    af = eng.update([right, left], 0.0)
    frame = np.zeros((720, 1280, 3), np.uint8)
    plain = fx.render(frame, [right, left], af, "Right", None, now=0.0)
    shown = fx.render(frame, [right, left], af, "Right", (4, 0.0), now=0.2)
    faded = fx.render(frame, [right, left], af, "Right", (4, 0.0), now=0.9)
    gone = fx.render(frame, [right, left], af, "Right", (4, 0.0), now=1.3)
    assert plain.shape == (1440, 2560, 3)
    assert (plain[..., 0] == plain[..., 1]).all() and (plain[..., 0] == plain[..., 2]).all()   # monochrome
    zone = (slice(1000, 1440), slice(0, 900))                      # last-action numeral, lower left
    m = lambda im: float(im[zone].mean())
    assert m(shown) > m(plain) + 2                                  # numeral drawn
    assert m(plain) + 0.3 < m(faded) < m(shown)                     # ...and fading
    assert abs(m(gone) - m(plain)) < 0.5                            # ...and gone after ~1.2 s
    def box(h):
        lo, hi = (h.pts[:, :2] * 2).min(axis=0).astype(int), (h.pts[:, :2] * 2).max(axis=0).astype(int)
        return plain[lo[1]:hi[1], lo[0]:hi[0]]
    assert 0 < box(left).max() <= 170        # the ignored hand is dim grey, dashed
    assert box(right).max() >= 200           # the dominant hand is bone white


# ---- shutdown and cleanup (audit D#1, D#7)
def _patch_runtime(monkeypatch, tmp_path):
    """Patch signals so the SIGTERM handler can be called by hand; returns {signum: handler}."""
    handlers = {}
    monkeypatch.setattr(M.signal, "signal", lambda n, h: handlers.__setitem__(n, h))
    monkeypatch.setattr(M, "PIDFILE", tmp_path / "airpen.pid")
    monkeypatch.setattr(M, "load_hand", lambda: "Right")
    from airpen.output import NullOutput
    monkeypatch.setattr(M, "make_output", lambda *a, **k: NullOutput())
    monkeypatch.setattr(M, "load_overrides", lambda: {})
    monkeypatch.setattr("airpen.tuning_io.load_overrides", lambda: {})
    return handlers


def test_startup_camera_retry_stops_on_sigterm(monkeypatch, tmp_path):
    handlers = _patch_runtime(monkeypatch, tmp_path)
    attempts = []

    class NoCam:
        def __init__(self, *a, **k):
            attempts.append(1)
            handlers[M.signal.SIGTERM]()          # ./run.sh stop arrives while the camera is missing
            raise RuntimeError("could not open camera 0")

    monkeypatch.setattr(M, "Camera", NoCam)
    assert M.main(["--no-notify"]) == 0
    assert len(attempts) == 1                      # no further open after the stop request
    assert not (tmp_path / "airpen.pid").exists()


def test_reconnect_retry_stops_on_sigterm(monkeypatch, tmp_path):
    handlers = _patch_runtime(monkeypatch, tmp_path)
    opened, released = [], []

    class Cam:
        size = (FW, FH)

        def __init__(self, *a, **k):
            opened.append(1)
            if len(opened) > 1:
                handlers[M.signal.SIGTERM]()
                raise RuntimeError("could not open camera 0")

        def read(self, seq, timeout=1.0):
            raise RuntimeError("camera stopped delivering frames")

        def release(self):
            released.append(1)

    class Tracker:
        def __init__(self, *a, **k): pass
        def close(self): pass

    monkeypatch.setattr(M, "Camera", Cam)
    monkeypatch.setattr(M, "HandTracker", Tracker)
    assert M.main(["--no-notify"]) == 0
    assert len(opened) == 2 and released == [1]
    assert not (tmp_path / "airpen.pid").exists()


def test_startup_failure_after_camera_still_cleans_up(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch, tmp_path)
    state = {"released": False}

    class Cam:
        size = (FW, FH)
        def __init__(self, *a, **k): pass
        def release(self): state["released"] = True

    class BadTracker:
        def __init__(self, *a, **k):
            raise FileNotFoundError("model missing")

    monkeypatch.setattr(M, "Camera", Cam)
    monkeypatch.setattr(M, "HandTracker", BadTracker)
    with pytest.raises(FileNotFoundError):
        M.main(["--no-notify"])
    assert state["released"]
    assert not (tmp_path / "airpen.pid").exists()          # pidfile removed, lock released
    fh = M.single_instance()                               # ...so a new instance can start
    fh.close()


def test_preview_keeps_color_camera_with_hand_overlays(monkeypatch, tmp_path, capsys):
    """Landmarks/actions are overlaid without running the monochrome renderer."""
    calls = {"render": 0, "show": 0, "overlay": 0}

    def boom(self, *a, **k):
        calls["render"] += 1
        raise ValueError("cannot convert float NaN to integer")

    color_camera = np.empty((FH, FW, 3), dtype=np.uint8)
    color_camera[:] = (20, 100, 180)
    monkeypatch.setattr(M.cv2, "flip", lambda frame, axis: color_camera.copy())
    def show_camera(window, image):
        assert image.shape == color_camera.shape
        np.testing.assert_array_equal(image[0, 0], color_camera[0, 0])
        assert np.mean(np.all(image == color_camera, axis=2)) > 0.7
        calls["overlay"] += int(np.any(image != color_camera))
        calls["show"] += 1
    monkeypatch.setattr(HandFX, "render", boom)
    monkeypatch.setattr(M.cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(M.cv2, "resizeWindow", lambda *a: None)
    monkeypatch.setattr(M.cv2, "setWindowTitle", lambda *a: None)
    monkeypatch.setattr(M.cv2, "imshow", show_camera)
    monkeypatch.setattr(M.cv2, "waitKey", lambda *a: -1)
    monkeypatch.setattr(M.cv2, "getWindowProperty", lambda *a: 1)
    monkeypatch.setattr(M.cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(M.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(M, "place_window_on_screen", lambda: None)
    got = run_main(session(), monkeypatch, tmp_path, capsys, ["--show"])
    assert [g[0] for g in got] == [i for i, _ in EXPECT]
    assert calls["render"] == 0 and calls["show"] > 10 and calls["overlay"] > 10


def test_frame_limit_is_honored_while_idle(monkeypatch, tmp_path, capsys):
    assert run_main(idle(10), monkeypatch, tmp_path, capsys, frames=1) == []
    assert run_main.reads == 1
