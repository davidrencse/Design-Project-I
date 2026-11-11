"""Robustness: bad config files, camera construction, hostile landmarks in the renderer, run.sh pid checks."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_actions import mk  # noqa: E402

from airpen import settings, tuning_io  # noqa: E402
from airpen.actions import ActionEngine, ActionFrame, ActionTuning  # noqa: E402
from airpen.fx import HandFX  # noqa: E402
from airpen.gestures import Tuning  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ---- audit D#4: syntactically valid but wrong config never crashes startup
@pytest.fixture
def cfg(tmp_path, monkeypatch):
    f = tmp_path / "tuning.json"
    monkeypatch.setattr(tuning_io, "CONFIG", f)
    monkeypatch.setattr(settings, "SETTINGS", tmp_path / "settings.json")
    return f


@pytest.mark.parametrize("text", ["[]", "null", "3", '"x"', "{bad json"])
def test_non_object_json_is_ignored(cfg, text, capsys):
    cfg.write_text(text)
    settings.SETTINGS.write_text(text)
    assert tuning_io.load_overrides() == {}
    assert tuning_io.load_tuning(Tuning()) == Tuning()
    assert settings.load_hand() == "Right"
    assert "ignoring" in capsys.readouterr().err


def test_bad_values_keep_defaults_and_warn(cfg, capsys):
    cfg.write_text(json.dumps({"tap_on": "oops", "right_on": 0, "fist_bend": float("nan"), "tap_off": -1,
                               "confirm_frames": 0, "smooth": [1], "extend_bend": 120.5, "tap_min": 1e999,
                               "unknown_key": 5}))
    t = tuning_io.load_tuning(Tuning())
    d = Tuning()
    assert (t.tap_on, t.right_on, t.fist_bend, t.tap_off, t.confirm_frames, t.smooth, t.tap_min) == \
           (d.tap_on, d.right_on, d.fist_bend, d.tap_off, d.confirm_frames, d.smooth, d.tap_min)
    assert t.extend_bend == 120.5                       # the one good value still applies
    err = capsys.readouterr().err
    assert err.count("ignoring") >= 6
    ActionEngine(tune=t).update([mk()], 0)              # and the engine runs (no ZeroDivisionError)


def test_bad_action_tuning_keeps_defaults(cfg):
    cfg.write_text(json.dumps({"fist_hold_s": 0, "thumb_swipe_dist": "x", "peace_hold_s": 0.5}))
    a = tuning_io.load_tuning(ActionTuning())
    d = ActionTuning()
    assert a.fist_hold_s == d.fist_hold_s and a.thumb_swipe_dist == d.thumb_swipe_dist and a.peace_hold_s == 0.5


# ---- audit D#8: a failed construction releases the capture
def test_camera_init_failure_releases_capture(monkeypatch):
    from airpen import tracker
    cams = []

    class Cap:
        def __init__(self, *a): cams.append(self); self.released = False
        def isOpened(self): return self.opened
        def set(self, *a): return True
        def read(self): return False, None
        def release(self): self.released = True

    for opened in (True, False):
        Cap.opened = opened
        monkeypatch.setattr(tracker.cv2, "VideoCapture", Cap)
        with pytest.raises(RuntimeError):
            tracker.Camera(0, 640, 480, 30, lock_fps=False)
        assert cams[-1].released


# ---- audit E#1 / E#2: hostile geometry never raises or stalls the renderer
def _hand_with_nan(j):
    h = mk(at=(330, 330))
    h.pts[j, 0] = np.nan
    return h


@pytest.mark.parametrize("j", [0, 4, 9, 8])
def test_nan_landmark_hand_is_skipped(j):
    f = np.zeros((480, 640, 3), np.uint8)
    out = HandFX().render(f, [_hand_with_nan(j)], hand="Right")
    assert out.shape == (960, 1280, 3)
    out = HandFX().render(f, [_hand_with_nan(j), mk("Left", at=(100, 100))], hand="Right")
    assert out.shape == (960, 1280, 3)


def test_nan_debug_scalars_are_absent():
    f = np.zeros((480, 640, 3), np.uint8)
    h = mk(at=(330, 330), pose="fist")
    af = ActionFrame([], True, "fist", True, None,
                     {"thumb_lat": float("nan"), "fist_progress": float("nan"), "peace_progress": float("inf")})
    assert HandFX().render(f, [h], af, "Right").shape == (960, 1280, 3)


@pytest.mark.parametrize("label", ["Left", "Right"])
@pytest.mark.parametrize("at", [(2e9, 300), (-1e9, -1e9), (1e5, 1e5)])
def test_huge_coordinates_do_not_raise(label, at):
    f = np.zeros((480, 640, 3), np.uint8)
    h = mk(label, at=(330, 330))
    h.pts[:, :2] += np.array(at) - np.array([330, 330])
    assert HandFX().render(f, [h], hand="Right").shape == (960, 1280, 3)


def test_dashed_offscreen_segment_is_cheap_and_clipped(monkeypatch):
    from airpen import ui
    g = np.zeros((200, 300), np.uint8)
    calls = []
    real = ui.line
    monkeypatch.setattr(ui, "line", lambda *a, **k: (calls.append(1), real(*a, **k)))
    ui.dashed(g, (-1e5, -50), (-6e5, -50), 200)                     # entirely outside
    assert calls == []
    ui.dashed(g, (-640000, 100), (640000, 100), 200)                # crosses the image: only visible dashes
    assert 0 < len(calls) <= 300 / 11 + 3 and g[100].max() > 150
    ui.dashed(g, (float("nan"), 0), (10, 10), 200)                  # non-finite: skipped
    ui.dashed(g, (10, 10), (2e9, 0), 200)


def test_render_safe_falls_back_to_plain_frame(capsys):
    fx = HandFX()
    fx.render = lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    f = np.zeros((10, 10, 3), np.uint8)
    assert fx.render_safe(f, []) is f
    assert fx.render_safe(f, []) is f
    assert capsys.readouterr().err.count("render failed") == 1      # logged once


# ---- audit E#3 / E#4
def test_calibration_drops_steps_that_fit_never_used():
    from airpen import calibrate
    names = [s for s, _ in calibrate.STEPS]
    assert names == ["open", "fist", "tap_index", "tap_middle", "point", "rest"]
    out, notes = calibrate.fit({}, Tuning())
    assert out == {}


def test_fx_has_no_dead_helpers():
    import airpen.fx as fx
    assert not hasattr(fx, "_chaikin") and not hasattr(HandFX(), "t0")


# ---- audit D#3 / D#10: run.sh does not trust the pidfile and survives relative invocation
def _run_sh(args, runtime, cwd, script=None):
    env = dict(os.environ, XDG_RUNTIME_DIR=str(runtime), XDG_STATE_HOME=str(runtime))
    return subprocess.run(["bash", script or str(ROOT / "run.sh"), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs /proc")
def test_run_sh_ignores_pids_that_are_not_airpen(tmp_path):
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        for content in ("abc", "-1", "0", "", "99999999", str(sleeper.pid)):
            (tmp_path / "airpen.pid").write_text(content)
            assert _run_sh(["status"], tmp_path, ROOT).stdout.splitlines()[0] == "not running", content
        fake = subprocess.Popen(["bash", "-c", "exec -a 'python -m airpen' sleep 30"])
        try:
            import time; time.sleep(0.3)
            (tmp_path / "airpen.pid").write_text(str(fake.pid))
            assert _run_sh(["status"], tmp_path, ROOT).stdout.startswith(f"running (pid {fake.pid})")
        finally:
            fake.kill(); fake.wait()
    finally:
        sleeper.kill(); sleeper.wait()


def test_run_sh_works_from_another_directory_by_relative_path(tmp_path):
    rel = os.path.relpath(ROOT / "run.sh", ROOT.parent)
    r = _run_sh(["status"], tmp_path, ROOT.parent, script=rel)
    assert r.returncode == 0 and "not running" in r.stdout
    assert 'SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"' in (ROOT / "run.sh").read_text()


# --- camera index after replug / closed preview window ---------------------------------------

def test_resolve_camera_falls_back_to_nth_capture_node(tmp_path, monkeypatch):
    from airpen import tracker
    sysfs = tmp_path / "sys"
    for n, idx in ((1, 0), (2, 1), (3, 0), (4, 1)):     # replugged cam: capture+metadata pairs
        (sysfs / f"video{n}").mkdir(parents=True)
        (sysfs / f"video{n}" / "index").write_text(f"{idx}\n")
    real_path = tracker.Path

    def fake_path(p, *a):
        p = str(p)
        if p == "/sys/class/video4linux":
            return real_path(sysfs)
        if p.startswith("/dev/video"):
            return real_path(tmp_path / "dev" / p.rsplit("/", 1)[1])   # none of these exist
        return real_path(p, *a)

    monkeypatch.setattr(tracker, "Path", fake_path)
    monkeypatch.setattr(tracker.sys, "platform", "linux")
    assert tracker.resolve_camera(0) == 1      # /dev/video0 gone -> first capture node
    assert tracker.resolve_camera(1) == 3      # second capture node (skips metadata video2)
    assert tracker.resolve_camera(7) == 7      # nothing to map to: leave it, the open error says why
    assert tracker.resolve_camera("/dev/video9") == "/dev/video9"


def test_window_closed_treats_qt_error_as_closed(monkeypatch):
    import cv2
    from airpen import ui

    def boom(*_):
        raise cv2.error("NULL guiReceiver (please create a window)")

    monkeypatch.setattr(ui.cv2, "getWindowProperty", boom)
    assert ui.window_closed("AirPen record") is True
    monkeypatch.setattr(ui.cv2, "getWindowProperty", lambda *_: 1.0)
    assert ui.window_closed("AirPen record") is False
    monkeypatch.setattr(ui.cv2, "getWindowProperty", lambda *_: 0.0)
    assert ui.window_closed("AirPen record") is True


@pytest.mark.parametrize("field,value", [("pts", float("nan")), ("norm", float("inf")), ("world", float("nan"))])
def test_recognizer_ignores_invalid_landmarks(field, value):
    hand = mk()
    getattr(hand, field)[4, 0] = value
    assert not ActionEngine().update([hand], 0).hand_seen


def test_recognizer_ignores_uncertain_and_degenerate_hands():
    h = mk()
    h.score = 0.1
    assert not ActionEngine().update([h], 0).hand_seen
    h.score = 1
    h.world[:] = 0
    assert not ActionEngine().update([h], 0).hand_seen


def test_duplicate_timestamps_do_not_confirm_a_gesture():
    eng = ActionEngine()
    h = mk(pose="tap_index")
    for _ in range(20):
        assert not eng.update([h], 1).events
    assert not eng.update([h], float("nan")).hand_seen


def test_invalid_hysteresis_and_smoothing_keep_defaults(cfg):
    cfg.write_text(json.dumps({"tap_on": 1, "tap_off": 0.5, "smooth": 4, "confirm_frames": 2.5}))
    t = tuning_io.load_tuning(Tuning())
    d = Tuning()
    assert (t.tap_on, t.tap_off, t.smooth, t.confirm_frames) == (d.tap_on, d.tap_off, d.smooth, d.confirm_frames)


def test_two_crossed_hands_preserve_recent_identity():
    from airpen.tracker import HandTracker
    tracker = object.__new__(HandTracker)
    left, right = mk("Left", at=(900, 500)), mk("Right", at=(300, 500))
    tracker._last_wrist = {"Left": (0, left.norm[0].copy()), "Right": (0, right.norm[0].copy())}
    left.label = right.label = "Right"
    hands = [left, right]
    tracker._stabilize_labels(hands, 0.03)
    assert left.label == "Left" and right.label == "Right" and len(hands) == 2
