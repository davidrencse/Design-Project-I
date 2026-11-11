"""PC integration tested through a recording backend: never controls the real desktop."""
import json
from types import SimpleNamespace

import pytest

from airpen.actions import ActionEvent
from airpen.controls import Controls, load_bindings
from airpen.output import NullOutput, Output, UInputOutput


class Recorder(Output):
    name = "recording"

    def __init__(self):
        # Avoid desktop discovery in tests.
        self.held, self._ctrl, self.calls = set(), False, []
        self.ws_next = self.ws_prev = None

    def _button(self, button, down): self.calls.append((button, down))
    def _move(self, x, y): self.calls.append(("move", x, y))
    def _wheel(self, v, h): self.calls.append(("scroll", v, h))
    def _ctrl_key(self, down): self.calls.append(("ctrl", down))
    def _combo(self, direction): self.calls.append(("workspace", direction))
    def hotkey(self, keys): self.calls.append(("hotkey", keys))


def event(name, t=1):
    return ActionEvent(0, name, "Right", t)


def test_default_bindings_reach_real_output_primitives(tmp_path):
    out = Recorder()
    c = Controls(out, load_bindings(tmp_path / "missing"))
    for name in ["index_tap", "middle_tap", "thumb_swipe_right", "thumb_swipe_left", "thumb_tap"]:
        c.dispatch(event(name))
    assert out.calls == [("left", True), ("left", False), ("right", True), ("right", False),
                         ("workspace", "next"), ("workspace", "prev"), ("hotkey", ["play_pause"])]
    c.close()
    assert not out.held


def test_pause_resume_and_repeat_guard(tmp_path):
    out = Recorder()
    c = Controls(out, load_bindings(tmp_path / "missing"))
    c.dispatch(event("peace", 1))
    c.dispatch(event("index_tap", 2))
    assert not c.enabled and not out.calls
    c.dispatch(event("peace", 3))
    c.dispatch(event("index_tap", 4))
    c.dispatch(event("index_tap", 4.01))
    c.dispatch(event("index_tap", 4.2))
    assert c.enabled and len(out.calls) == 4
    c.close()


@pytest.mark.parametrize("spec", [None, {"type": "command", "argv": "firefox"},
    {"type": "hotkey", "keys": ["invalid"]}, {"type": "scroll", "amount": float("nan")},
    {"type": "click", "button": "invalid"}, {"type": "oops"}])
def test_invalid_bindings_fail_before_desktop_access(tmp_path, spec):
    p = tmp_path / "bindings.json"
    p.write_text(json.dumps({"index_tap": spec}))
    with pytest.raises(ValueError): load_bindings(p)


def test_none_backend_never_launches_command(tmp_path):
    p = tmp_path / "bindings.json"
    p.write_text(json.dumps({"index_tap": {"type": "command", "argv": ["does-not-exist"]}}))
    c = Controls(NullOutput(), load_bindings(p))
    c.dispatch(event("index_tap"))
    assert c._worker is None
    c.close()


def test_shortcut_releases_keys_after_failed_press():
    out = object.__new__(UInputOutput)
    out._key_codes = lambda: {"ctrl": 1, "a": 2}
    calls = []
    def key(code, down):
        calls.append((code, down))
        if code == 2 and down: raise OSError("disconnected")
    out._key = key
    with pytest.raises(OSError): out.hotkey(["ctrl", "a"])
    assert calls == [(1, True), (2, True), (2, False), (1, False)]


def test_live_loop_dispatches_clicks_without_touching_pc(monkeypatch, tmp_path, capsys):
    import airpen.main as main
    from test_e2e import run_main, session
    made = []
    real = main.Controls
    def factory(output, bindings, *args):
        out = Recorder()
        made.append(out)
        return real(out, bindings, *args)
    monkeypatch.setattr(main, "Controls", factory)
    run_main(session(), monkeypatch, tmp_path, capsys)
    assert ("left", True) in made[0].calls
    assert ("right", True) in made[0].calls
    assert ("workspace", "next") in made[0].calls
    assert ("hotkey", ["play_pause"]) in made[0].calls
    assert not made[0].held


def test_commands_use_argv_on_worker_and_close_releases_output(tmp_path, monkeypatch):
    import threading
    from airpen import controls
    done, calls = threading.Event(), []
    class Process:
        def __init__(self, argv, **kw): calls.append((argv, kw))
        def wait(self, timeout=None): done.set(); return 0
        def kill(self): pass
    monkeypatch.setattr(controls.subprocess, "Popen", Process)
    p = tmp_path / "bindings.json"
    p.write_text(json.dumps({"thumb_tap": {"type": "command", "argv": ["launcher", "literal;argument"]}}))
    out = Recorder()
    c = Controls(out, load_bindings(p))
    try:
        c.dispatch(event("thumb_tap"))
        assert done.wait(2)
        assert calls[0][0] == ["launcher", "literal;argument"]
        assert "shell" not in calls[0][1]
    finally:
        c.close()
    assert not c._worker.is_alive() and not out.held


def test_auto_wayland_does_not_silently_use_xwayland(monkeypatch):
    import airpen.output as output
    monkeypatch.setattr(output.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-test")
    def unavailable(**kw): raise OSError("no permission")
    monkeypatch.setattr(output, "UInputOutput", unavailable)
    monkeypatch.setattr(output, "PynputOutput", lambda **kw: pytest.fail("cannot control native Wayland"))
    assert isinstance(output.make_output("auto"), NullOutput)
    with pytest.raises(OSError): output.make_output("uinput")


def test_calibrated_pointer_scale_is_applied(tmp_path):
    from airpen.gestures import Tuning
    c = Controls(Recorder(), load_bindings(tmp_path / "missing"), tune=Tuning(pointer_ref_scale=0.25))
    assert c.pointer.ref_scale == 0.25
    c.close()
