"""Record format + replay logic, on a tiny synthetic capture and a fake engine (no mediapipe, no camera)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from airpen.record import build_meta, expected_counts, save_step, write_meta
from airpen.replay import judge, load_meta, newest_capture, replay_dir

N = 21


class FakeEngine:
    """Emits action id k on each rising edge of marker k in norm[0,0] (0 = nothing).
    Records every hand list it was fed so tests can check the rebuilt objects."""
    def __init__(self, hand: str):
        self.hand = hand
        self.prev = 0.0
        self.calls = []

    def update(self, hands, t):
        self.calls.append((t, hands))
        events = []
        if hands:
            m = float(hands[0].norm[0, 0])
            if m > 0 and m != self.prev:
                events.append(SimpleNamespace(id=int(m), name="x", hand=self.hand, t=t))
            self.prev = m
        else:
            self.prev = 0.0
        return SimpleNamespace(events=events)


def _factory_store():
    made = []

    def factory(hand):
        eng = FakeEngine(hand)
        made.append(eng)
        return eng
    return factory, made


def _arrays(n, markers, present=None, world_nan=False):
    """markers: {frame index: action id} fired as norm[0,0] marker on that frame."""
    present = np.ones(n, bool) if present is None else np.asarray(present, bool)
    pts = np.zeros((n, N, 3))
    norm = np.zeros((n, N, 2))
    world = np.ones((n, N, 3)) * 0.01
    for i, aid in markers.items():
        norm[i, 0, 0] = aid
    if world_nan:
        world[:] = np.nan
    pts[~present] = np.nan
    norm[~present] = np.nan
    world[~present] = np.nan
    return dict(t=np.arange(n) * 0.1, present=present, pts=pts, norm=norm, world=world)


def _write_capture(root, steps: dict, reps=2, hand="Right"):
    """steps: name -> (n, markers, present-or-None, world_nan)"""
    cap = root / "20261107-101500"
    for name, (n, markers, present, wnan) in steps.items():
        a = _arrays(n, markers, present, wnan)
        save_step(cap, name, t=a["t"], present=a["present"], pts=a["pts"],
                  norm=a["norm"], world=a["world"])
    write_meta(cap, build_meta(hand, (640, 480), 30.0, reps, list(steps)))
    return cap


def test_expected_counts_match_spec():
    assert expected_counts("neutral", 5) == {}
    assert expected_counts("open", 5) == {}
    assert expected_counts("index_tap", 5) == {1: 5}
    assert expected_counts("thumb_swipe_right", 5) == {4: 5, 3: 1}
    assert expected_counts("thumb_tap", 4) == {7: 4, 3: 1}
    assert expected_counts("peace", 4) == {6: 4}


def test_meta_expected_keys_are_strings(tmp_path):
    meta = build_meta("Left", (640, 480), 29.97, 3, ["index_tap"])
    assert meta["expected"]["index_tap"] == {"1": 3}
    write_meta(tmp_path / "c", meta)
    loaded = json.loads((tmp_path / "c" / "meta.json").read_text())
    assert loaded["hand"] == "Left" and loaded["frame_size"] == [640, 480]
    assert load_meta(tmp_path / "c")["steps"] == ["index_tap"]


def test_npz_format_roundtrip(tmp_path):
    a = _arrays(4, {}, present=[1, 0, 1, 1], world_nan=False)
    save_step(tmp_path, "open", **{k: a[k] for k in ("t", "present", "pts", "norm", "world")})
    with np.load(tmp_path / "open.npz") as z:
        assert set(z.files) == {"t", "present", "pts", "norm", "world"}
        assert z["t"].shape == (4,) and z["present"].dtype == bool
        assert z["pts"].shape == (4, N, 3) and z["norm"].shape == (4, N, 2) and z["world"].shape == (4, N, 3)
        assert np.isnan(z["pts"][1]).all()


def test_pass_and_fail_report(tmp_path):
    # marker value = action id, fired on its rising edge (frames are 0.1 s apart)
    cap = _write_capture(tmp_path, {
        "neutral": (10, {}, None, False),
        "index_tap": (20, {2: 1, 5: 1, 8: 3}, None, False),   # 2 index taps + 1 stray fist
        "thumb_swipe_right": (20, {2: 3}, None, False),       # fist only: 4 missed twice
    }, reps=2)
    factory, _ = _factory_store()
    rep = replay_dir(cap, engine_factory=factory)
    assert set(rep["steps"]) == {"neutral", "index_tap", "thumb_swipe_right"}
    s = rep["steps"]
    assert s["neutral"]["ok"] and s["neutral"]["expected"] == {} and s["neutral"]["got"] == {}
    assert s["neutral"]["frames"] == 10 and s["neutral"]["present"] == 10
    assert s["index_tap"]["expected"] == {1: 2} and s["index_tap"]["got"] == {1: 2, 3: 1}
    assert s["index_tap"]["ok"]                                  # extra fist is info only
    assert any("tolerated" in m for m in s["index_tap"]["info"])
    assert not s["thumb_swipe_right"]["ok"]                      # 4 expected 2, got 0: beyond tol 1
    assert s["thumb_swipe_right"]["problems"] == ["thumb_swipe_right (4): expected 2, got 0"]
    assert rep["ok"] is False


def test_unexpected_non_fist_fails_and_all_ok_passes(tmp_path):
    cap = _write_capture(tmp_path, {
        "open": (12, {2: 2}, None, False),        # middle_tap fired during open -> false positive
    }, reps=1)
    factory, _ = _factory_store()
    rep = replay_dir(cap, engine_factory=factory)
    assert rep["ok"] is False
    assert rep["steps"]["open"]["problems"] == ["unexpected middle_tap (2) x1"]

    cap2 = _write_capture(tmp_path / "good", {
        "peace": (30, {2: 6}, None, False),       # exactly one peace, as expected
    }, reps=1)
    rep2 = replay_dir(cap2, engine_factory=factory)
    assert rep2["ok"] is True and rep2["steps"]["peace"]["ok"]


def test_nan_world_becomes_none_and_absent_frames_feed_empty_list(tmp_path):
    cap = _write_capture(tmp_path, {
        "open": (4, {}, [True, False, True, True], True),  # world NaN everywhere
    }, reps=1)
    factory, made = _factory_store()
    rep = replay_dir(cap, engine_factory=factory)
    eng = made[0]
    assert rep["steps"]["open"]["present"] == 3 and rep["steps"]["open"]["frames"] == 4
    assert len(eng.calls) == 4
    t1, hands1 = eng.calls[1]
    assert hands1 == [] and abs(t1 - 0.1) < 1e-9       # absent frame: engine still ticks, with no hands
    _, hands0 = eng.calls[0]
    assert len(hands0) == 1 and hands0[0].label == "Right" and hands0[0].score == 1.0
    assert hands0[0].world is None and hands0[0].pts.shape == (N, 3) and hands0[0].norm.shape == (N, 2)

    cap2 = _write_capture(tmp_path / "w", {"open": (2, {}, None, False)}, reps=1)
    factory2, made2 = _factory_store()
    replay_dir(cap2, engine_factory=factory2)
    assert made2[0].calls[0][1][0].world.shape == (N, 3)  # finite world passed through


def test_judge_rules():
    from collections import Counter
    assert judge({1: 5}, Counter({1: 5}), 1) == ([], [])
    assert judge({1: 5}, Counter({1: 4}), 1)[0] == []           # within tol
    assert judge({1: 5}, Counter({1: 3}), 1)[0]                 # outside tol
    p, info = judge({}, Counter({3: 2}), 1)
    assert p == [] and info                                     # extra fist tolerated
    assert judge({}, Counter({6: 1}), 1)[0]                     # other unexpected fails


def test_newest_capture_and_missing_step(tmp_path):
    assert newest_capture(tmp_path / "nope") is None
    _write_capture(tmp_path / "a", {"open": (2, {}, None, False)})
    new = tmp_path / "a" / "20991231-235959"
    new.mkdir()
    write_meta(new, build_meta("Right", (640, 480), 30, 1, ["open"]))
    assert newest_capture(tmp_path / "a") == new
    # listed step whose file is missing -> reported as a failure, not a crash
    write_meta(new, build_meta("Right", (640, 480), 30, 1, ["open", "fist"]))
    save_step(new, "open", t=[0.0], present=[True], pts=np.zeros((1, N, 3)),
              norm=np.zeros((1, N, 2)), world=np.zeros((1, N, 3)))
    factory, _ = _factory_store()
    rep = replay_dir(new, engine_factory=factory)
    assert rep["steps"]["open"]["ok"] and not rep["steps"]["fist"]["ok"]
    assert rep["ok"] is False


# ---- audit D#5 / D#11: replay builds the live engine and rejects malformed captures cleanly
def test_default_engine_factory_uses_personal_thresholds(monkeypatch):
    from airpen import tuning_io
    from airpen.actions import make_engine
    from airpen.replay import default_engine_factory
    monkeypatch.setattr(tuning_io, "load_overrides", lambda: {"thumb_swipe_dist": 0.9, "tap_on": 0.3})
    for eng in (default_engine_factory("Left"), make_engine("Left")):
        assert eng.hand == "Left"
        assert eng.act.thumb_swipe_dist == 0.9 and eng.tune.tap_on == 0.3


def test_main_builds_the_same_engine_as_replay():
    import inspect
    import airpen.main as M
    assert "make_engine(" in inspect.getsource(M.main)
    assert "make_engine" in inspect.getsource(__import__("airpen.replay", fromlist=["x"]).default_engine_factory)


def test_record_stores_effective_tuning_and_replay_notes_changes(tmp_path, monkeypatch):
    from airpen import tuning_io
    from airpen.actions import effective_tuning
    from airpen.replay import format_report
    cap = _write_capture(tmp_path, {"open": (3, {}, None, False)})
    monkeypatch.setattr(tuning_io, "load_overrides", lambda: {})
    meta = build_meta("Right", (640, 480), 30, 1, ["open"], effective_tuning())
    assert meta["tuning"]["tap_on"] == effective_tuning()["tap_on"] and "thumb_swipe_dist" in meta["tuning"]
    write_meta(cap, meta)
    monkeypatch.setattr(tuning_io, "load_overrides", lambda: {"tap_on": 0.3})
    rep = replay_dir(cap)                       # default factory: current thresholds differ
    assert rep["tuning_changed"] == ["tap_on"]
    assert "CURRENT thresholds" in format_report(rep)


def test_malformed_step_arrays_fail_the_step_not_the_run(tmp_path):
    cap = tmp_path / "c"
    cap.mkdir()
    a = _arrays(2, {})
    np.savez(cap / "short.npz", t=a["t"], present=np.array([False]), pts=a["pts"], norm=a["norm"], world=a["world"])
    np.savez(cap / "nokey.npz", t=a["t"])
    np.savez(cap / "badpts.npz", t=a["t"], present=a["present"], pts=np.zeros((2, 5, 3)), norm=a["norm"], world=a["world"])
    (cap / "garbage.npz").write_bytes(b"not a zip")
    write_meta(cap, {"hand": "Right", "steps": ["short", "nokey", "badpts", "garbage", "gone", "badexp"],
                     "expected": {"badexp": {"1": "x"}}})
    save_step(cap, "badexp", **{k: a[k] for k in ("t", "present", "pts", "norm", "world")})
    factory, _ = _factory_store()
    rep = replay_dir(cap, engine_factory=factory)
    assert rep["ok"] is False
    for step in ("short", "nokey", "badpts", "garbage", "gone", "badexp"):
        e = rep["steps"][step]
        assert not e["ok"] and e["problems"] and e["problems"][0].startswith("could not replay")
    assert "shape" in rep["steps"]["short"]["problems"][0] or "expected" in rep["steps"]["short"]["problems"][0]


def test_missing_meta_keys_exit_2(tmp_path, capsys):
    from airpen.replay import main as replay_main
    for meta in ({"steps": ["open"]}, {"hand": "Right"}, [], {"hand": "Up", "steps": []},
                 {"hand": "Right", "steps": "open"}):
        cap = tmp_path / "c"
        cap.mkdir(exist_ok=True)
        (cap / "meta.json").write_text(json.dumps(meta))
        assert replay_main([str(cap)]) == 2
        err = capsys.readouterr().err
        assert "cannot read capture" in err and "Traceback" not in err
