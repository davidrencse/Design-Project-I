"""Color-camera landmarks and action feedback; no camera or real notifications needed."""
from types import SimpleNamespace

import numpy as np
import pytest

from airpen.fx import CameraOverlay
from test_actions import mk


def camera():
    image = np.empty((480, 640, 3), np.uint8)
    image[:] = (35, 105, 175)
    return image


def test_empty_overlay_leaves_every_camera_pixel_unchanged():
    frame = camera()
    np.testing.assert_array_equal(CameraOverlay().render(frame, []), frame)


def test_fingers_and_hands_are_drawn_without_recoloring_camera(monkeypatch):
    frame = camera()
    overlay = CameraOverlay()
    right, left = mk(at=(380, 340)), mk("Left", at=(140, 340))
    labels = []
    real = overlay._label
    def label(image, text, *args, **kw):
        labels.append(text)
        real(image, text, *args, **kw)
    monkeypatch.setattr(overlay, "_label", label)
    result = overlay.render(frame, [right, left], hand="Right")
    assert result.shape == frame.shape
    assert np.mean(np.all(result == frame, axis=2)) > .85
    for tip in (4, 8, 12, 16, 20):
        x, y = right.pts[tip, :2].astype(int)
        assert not np.array_equal(result[y, x], frame[y, x])
    assert "Right hand" in labels and "Left hand (ignored)" in labels
    np.testing.assert_array_equal(frame[0, 0], [35, 105, 175])


@pytest.mark.parametrize("aid,label", [(1, "index tap"), (4, "thumb swipe right"), (5, "thumb swipe left")])
def test_recognized_action_stays_visible_then_expires(monkeypatch, aid, label):
    frame = camera()
    overlay = CameraOverlay()
    labels = []
    monkeypatch.setattr(overlay, "_label", lambda image, text, *a, **k: labels.append(text))
    overlay.render(frame, [], last_action=(aid, 10), now=10.5)
    assert labels == [f"Recognized: {label}"]
    labels.clear()
    result = overlay.render(frame, [], last_action=(aid, 10), now=12.1)
    assert labels == []
    np.testing.assert_array_equal(result, frame)


@pytest.mark.parametrize("bad", [np.full((21, 3), np.nan), np.zeros((5, 3)), np.full((21, 3), 2e9)])
def test_bad_hand_data_cannot_break_the_overlay(bad):
    frame = camera()
    result = CameraOverlay().render(frame, [SimpleNamespace(pts=bad, label="Right")])
    np.testing.assert_array_equal(result, frame)


def test_overlay_failure_falls_back_to_camera_and_logs_once(monkeypatch, capsys):
    frame = camera()
    overlay = CameraOverlay()
    def broken(*a, **kw): raise ValueError("bad drawing")
    monkeypatch.setattr(overlay, "render", broken)
    assert overlay.render_safe(frame, []) is frame
    assert overlay.render_safe(frame, []) is frame
    assert capsys.readouterr().err.count("camera overlay failed") == 1


def test_desktop_notifications_identify_the_gesture(monkeypatch):
    import airpen.main as main
    calls = []
    monkeypatch.setattr(main.shutil, "which", lambda command: "/usr/bin/notify-send")
    def popen(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(main.subprocess, "Popen", popen)
    notify = main.Notifier(True)
    notify("AirPen: thumb swipe left", "Gesture 5 | Right hand")
    assert calls[0][-2:] == ["AirPen: thumb swipe left", "Gesture 5 | Right hand"]
    main.Notifier(False)("disabled")
    assert len(calls) == 1
