"""Gesture engine tests on synthetic hands (pixel coords, mirrored view, scale = 100 px)."""
from types import SimpleNamespace

import numpy as np
import pytest

from airpen.gestures import Debounced, GestureEngine, Tuning, classify_pose, compute_metrics
from airpen.ledframe import COLS, ROWS, LedRenderer
from airpen.output import Output
from airpen.pointer import Pointer

W, H = 1280, 720
MCP = {"index": (-35, -95), "middle": (0, -100), "ring": (30, -95), "pinky": (55, -80)}
ORDER = ("index", "middle", "ring", "pinky")


def make_hand(label="Right", at=(640, 600), extended=("index", "middle", "ring", "pinky"),
              thumb_tip=None, bunch=False):
    p = np.zeros((21, 3))
    ox, oy = at
    p[0] = (0, 0, 0)
    p[1], p[2], p[3], p[4] = (-30, -30, 0), (-55, -55, 0), (-70, -80, 0), (-80, -105, 0)
    for k, name in enumerate(ORDER):
        base = 5 + 4 * k
        mx, my = MCP[name]
        p[base] = (mx, my, 0)
        if bunch:
            p[base + 1] = (mx * 0.7, my - 30, 0)
            p[base + 2] = (mx * 0.4, my - 45, 0)
            p[base + 3] = (-10 + k * 4, -150, 0)
        elif name in extended:
            p[base + 1], p[base + 2], p[base + 3] = (mx, my - 40, 0), (mx, my - 65, 0), (mx, my - 85, 0)
        else:
            p[base + 1], p[base + 2], p[base + 3] = (mx, my - 30, 0), (mx, my - 12, 0), (mx, my + 10, 0)
    if bunch:
        p[4] = (-4, -150, 0)
    if thumb_tip is not None:
        p[4] = (*thumb_tip, 0)
    p[:, 0] += ox
    p[:, 1] += oy
    return SimpleNamespace(label=label, score=0.99, pts=p, norm=p[:, :2] / np.array([W, H]))


def tip(name):
    mx, my = MCP[name]
    return (mx, my - 85)


def run(engine, frames, t0=0.0, dt=1 / 30):
    """frames: list of hand lists. Returns (all events, last GestureFrame, end time)."""
    events, gf, t = [], None, t0
    for hands in frames:
        gf = engine.update(hands, t)
        events += gf.events
        t += dt
    return events, gf, t


def kinds(events):
    return [(e.kind, e.button or e.dir) for e in events]


FIST = (10, -60)   # thumb tip lying across the curled fingers


def fist(label="Right", **kw):
    return make_hand(label, extended=(), thumb_tip=FIST, **kw)


def cursor_on(engine, label="Right", t0=0.0):
    ev, gf, t = run(engine, [[fist(label)]] * 34, t0)          # 1.1 s fist hold
    ev2, gf, t = run(engine, [[make_hand(label)]] * 3, t)       # open the hand again
    assert gf.cursor_on and gf.cursor_hand == label
    return t


TUNE = Tuning()


@pytest.mark.parametrize("kw,pose", [
    ({}, "open"),
    ({"extended": ("index",)}, "point"),
    ({"extended": ("index", "middle")}, "peace"),
    ({"extended": (), "thumb_tip": FIST}, "fist"),
    ({"extended": (), "thumb_tip": (-35, -88)}, "curled"),     # thumb on the curled index: a pinch, not a fist
])
def test_poses(kw, pose):
    h = make_hand(**kw)
    assert classify_pose(compute_metrics(h.pts, h.norm, TUNE), TUNE) == pose


def test_debounce_hysteresis():
    d = Debounced(0.3, 0.45, 2)
    assert d.update(0.2) is None and d.update(0.2) is True
    assert d.update(0.4) is None and d.update(0.4) is None  # inside hysteresis gap: still held
    assert d.update(0.5) is None and d.update(0.5) is False


def test_fist_hold_toggles_mouse_mode_on_then_off():
    eng = GestureEngine()
    t = cursor_on(eng)
    ev, gf, t = run(eng, [[fist()]] * 34, t)
    assert not gf.cursor_on
    assert ("cursor_off", "") in kinds(ev)


def test_short_fist_does_nothing():
    eng = GestureEngine()
    ev, gf, _ = run(eng, [[fist()]] * 8 + [[make_hand()]] * 3)   # ~0.27 s
    assert not gf.cursor_on


def test_no_clicks_while_cursor_off():
    eng = GestureEngine()
    ev, _, _ = run(eng, [[make_hand(thumb_tip=tip("index"))]] * 5)
    assert not [e for e in ev if e.kind == "press"]


def test_index_tap_is_left_click_with_rewind():
    eng = GestureEngine()
    t = cursor_on(eng)
    tap = make_hand(thumb_tip=tip("index"))
    gf1 = eng.update([tap], t)
    assert not gf1.events                                   # one frame is never enough
    presses = []
    for i in range(1, 5):                                   # lands within ~130 ms
        gf = eng.update([tap], t + 0.033 * i)
        if ("press", "left") in kinds(gf.events):
            presses.append(gf.rewind)
    assert presses == [True]
    ev, _, _ = run(eng, [[make_hand()]] * 5, t + 0.2)
    assert ("release", "left") in kinds(ev)


def test_middle_tap_is_right_click_and_freezes_pointer():
    eng = GestureEngine()
    t = cursor_on(eng)
    ev, gf, t = run(eng, [[make_hand(thumb_tip=tip("middle"))]] * 3, t)
    assert ("press", "right") in kinds(ev)
    assert ("press", "left") not in kinds(ev)
    assert gf.freeze
    ev, gf, _ = run(eng, [[make_hand()]] * 3, t)
    assert ("release", "right") in kinds(ev) and not gf.freeze


def test_either_hand_can_click():
    eng = GestureEngine()
    t = cursor_on(eng, "Right")
    ev, _, _ = run(eng, [[make_hand("Right"), make_hand("Left", at=(300, 600), thumb_tip=tip("index"))]] * 3, t)
    assert any(e.kind == "press" and e.hand == "Left" and e.button == "left" for e in ev)


def test_lost_hand_releases_button():
    eng = GestureEngine()
    t = cursor_on(eng)
    ev, _, t = run(eng, [[make_hand(thumb_tip=tip("index"))]] * 3, t)
    assert ("press", "left") in kinds(ev)
    ev, _, _ = run(eng, [[]] * 12, t)  # 0.4 s without a hand
    assert ("release", "left") in kinds(ev)


TUCKED = (-5, -130)   # thumb tip resting on the folded middle finger


def rotate(h, deg):
    """Rotate a synthetic hand about its wrist (180 = hand upside down, pointing down)."""
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    w = h.pts[0, :2].copy()
    h.pts[:, :2] = (h.pts[:, :2] - w) @ R.T + w
    h.norm = h.pts[:, :2] / np.array([W, H])
    return h


def test_point_up_scrolls_up_and_speeds_up():
    eng = GestureEngine()
    ev, gf, _ = run(eng, [[make_hand(extended=("index",), thumb_tip=TUCKED)]] * 60)  # 2 s
    dys = [e.dy for e in ev if e.kind == "scroll"]
    assert ("scroll_start", "") in kinds(ev) and gf.scrolling
    assert dys and all(dy > 0 for dy in dys)
    assert dys[-1] > dys[0] * 2                      # accelerates while held
    assert not [e for e in ev if e.kind.startswith("zoom")]


def test_point_down_scrolls_down():
    eng = GestureEngine()
    ev, _, _ = run(eng, [[rotate(make_hand(at=(640, 200), extended=("index",), thumb_tip=TUCKED), 180)]] * 20)
    dys = [e.dy for e in ev if e.kind == "scroll"]
    assert dys and all(dy < 0 for dy in dys)


def test_point_sideways_does_not_scroll():
    eng = GestureEngine()
    ev, _, _ = run(eng, [[rotate(make_hand(extended=("index",), thumb_tip=TUCKED), 90)]] * 20)
    assert not [e for e in ev if e.kind == "scroll"]


def test_point_scroll_in_mouse_mode_freezes_cursor_and_never_clicks():
    eng = GestureEngine()
    t = cursor_on(eng)
    ev, gf, _ = run(eng, [[make_hand(extended=("index",), thumb_tip=TUCKED)]] * 15, t)
    assert gf.scrolling and gf.freeze
    assert not [e for e in ev if e.kind == "press"]


def test_fist_never_clicks():
    eng = GestureEngine()
    t = cursor_on(eng)
    ev, _, _ = run(eng, [[make_hand(extended=(), thumb_tip=(-15, -95))]] * 10, t)
    assert not [e for e in ev if e.kind == "press"]


def test_fist_is_clutch():
    eng = GestureEngine()
    t = cursor_on(eng)
    ev, gf, _ = run(eng, [[fist()]] * 6, t)        # short fist: freeze, not yet a toggle
    assert gf.clutch and gf.freeze and gf.cursor_on


def test_open_palm_swipe_left():
    eng = GestureEngine()
    frames = [[make_hand(at=(900 - 60 * i, 500))] for i in range(8)]  # 1800 px/s = 18 scales/s
    ev, _, _ = run(eng, frames)
    swipes = [e.dir for e in ev if e.kind == "swipe"]
    assert swipes == ["left"]  # cooldown stops repeats


def test_slow_motion_is_not_a_swipe():
    eng = GestureEngine()
    ev, _, _ = run(eng, [[make_hand(at=(900 - 3 * i, 500))] for i in range(20)])
    assert not [e for e in ev if e.kind == "swipe"]


def test_cursor_hand_does_not_swipe():
    eng = GestureEngine()
    t = cursor_on(eng)
    ev, _, _ = run(eng, [[make_hand(at=(900 - 60 * i, 500))] for i in range(8)], t)
    assert not [e for e in ev if e.kind == "swipe"]


def test_two_finger_zoom_in_and_out():
    eng = GestureEngine()
    pt = ("index",)
    apart = [[make_hand("Left", at=(600 - 15 * i, 500), extended=pt),
              make_hand("Right", at=(700 + 15 * i, 500), extended=pt)] for i in range(8)]
    ev, gf, t = run(eng, apart)
    zin = [e.amount for e in ev if e.kind == "zoom"]
    assert gf.zooming and zin and all(a > 0 for a in zin)
    together = [[make_hand("Left", at=(495 + 15 * i, 500), extended=pt),
                 make_hand("Right", at=(805 - 15 * i, 500), extended=pt)] for i in range(6)]
    ev, _, t = run(eng, together, t)
    assert all(e.amount < 0 for e in ev if e.kind == "zoom")
    ev, gf, _ = run(eng, [[make_hand("Left", at=(400, 500)), make_hand("Right", at=(900, 500))]], t)
    assert ("zoom_end", "") in kinds(ev) and not gf.zooming


def test_two_open_palms_held_toggles_arm():
    eng = GestureEngine()
    both = [make_hand("Left", at=(400, 500)), make_hand("Right", at=(900, 500))]
    ev, _, _ = run(eng, [both] * 50)  # 1.67 s
    assert [e.kind for e in ev].count("arm") == 1


# ---------------------------------------------------------------- pointer
def test_pointer_freeze_and_offset_continuity():
    p = Pointer(box=(0, 0, 1, 1), min_cutoff=1000, beta=0)  # filter ~off
    a = p.update(np.array([0.5, 0.5]), 0.0)
    assert np.allclose(a, [0.5, 0.5])
    b = p.update(np.array([0.7, 0.5]), 0.1, freeze=True)
    assert np.allclose(b, [0.5, 0.5])                # frozen
    c = p.update(np.array([0.75, 0.5]), 0.2)
    assert np.allclose(c, [0.55, 0.5], atol=1e-3)    # resumes relative, no jump


def test_pointer_edge_resync():
    p = Pointer(box=(0, 0, 1, 1), min_cutoff=1000, beta=0)
    p.update(np.array([0.5, 0.5]), 0.0)
    p.update(np.array([0.9, 0.5]), 0.1, freeze=True)  # offset now -0.4
    p.update(np.array([1.4, 0.5]), 0.2)               # pushes past the right edge
    out = p.update(np.array([1.4, 0.5]), 0.3)
    assert out[0] == pytest.approx(1.0)
    back = p.update(np.array([1.3, 0.5]), 0.4)
    assert back[0] == pytest.approx(0.9, abs=1e-3)     # edge absorbed the offset


def test_pointer_rewind_on_tap():
    p = Pointer(box=(0, 0, 1, 1), min_cutoff=1000, beta=0, rewind_s=0.08, settle_s=0.15)
    for i in range(10):
        p.update(np.array([0.5 + 0.01 * i, 0.5]), i * 0.033)
    out = p.update(np.array([0.62, 0.5]), 0.33, rewind=True)
    assert out[0] < 0.59                               # jumped back before the finger moved
    held = p.update(np.array([0.70, 0.5]), 0.40)
    assert held[0] == pytest.approx(out[0])           # holds during settle


# ---------------------------------------------------------------- output
class Recorder(Output):
    name = "rec"

    def __init__(self):
        super().__init__(ws_next="true", ws_prev="true")
        self.log = []

    def _move(self, x, y): self.log.append(("move", x, y))
    def _button(self, b, down): self.log.append(("btn", b, down))
    def _wheel(self, v, h): self.log.append(("wheel", v, h))
    def _ctrl_key(self, down): self.log.append(("ctrl", down))


def test_output_scroll_accumulates_fractions():
    o = Recorder()
    for _ in range(4):
        o.scroll(0.4)
    assert sum(v for k, v, *_ in o.log if k == "wheel") == 1


def test_output_release_all_and_idempotent_press():
    o = Recorder()
    o.press("left"); o.press("left"); o.zoom(150)
    o.release_all()
    assert o.log.count(("btn", "left", True)) == 1
    assert ("btn", "left", False) in o.log and o.log[-1] == ("ctrl", False)


def test_led_frame_shape():
    led = LedRenderer()
    led.trigger("click", 0.0)
    f = led.render(0.05, armed=True, cursor_on=True, cursor=np.array([0.5, 0.5]), hands=1,
                   left_down=True, right_down=False, scrolling=False, zooming=False,
                   zoom_level=0, clutch=False)
    assert f.shape == (ROWS, COLS) and f.dtype == np.uint8 and f[:, 0].min() == 255


def test_one_hand_can_never_zoom():
    """Zoom is two-handed only: a single hand spreading / pinching thumb+index does nothing."""
    eng = GestureEngine()
    ix, iy = tip("index")
    frames = [[make_hand(extended=("index",), thumb_tip=(ix - 20 - 8 * i, iy))] for i in range(12)]
    frames += [[make_hand(extended=("index",), thumb_tip=(ix - 108 + 8 * i, iy))] for i in range(10)]
    ev, _, _ = run(eng, frames)
    assert not [e for e in ev if e.kind.startswith("zoom")]


def test_one_hand_pinch_does_not_zoom_in_mouse_mode():
    eng = GestureEngine()
    t = cursor_on(eng)
    ix, iy = tip("index")
    ev, _, _ = run(eng, [[make_hand(extended=("index",), thumb_tip=(ix - 20 - 8 * i, iy))] for i in range(12)], t)
    assert not [e for e in ev if e.kind.startswith("zoom")]


def test_zoom_output_sends_whole_ctrl_steps():
    o = Recorder()
    o.zoom(50); o.zoom(50)
    assert not [x for x in o.log if x[0] == "wheel"]       # under one step: nothing yet
    o.zoom(50)
    assert ("ctrl", True) in o.log and ("wheel", 120, 0) in o.log
    o.zoom_end()
    assert o.log[-1] == ("ctrl", False)


# ---------------------------------------------------------------- accuracy features
def _row(h):
    from airpen.calibrate import FIELDS
    m = compute_metrics(h.pts, h.norm, TUNE)
    r = {k: float(getattr(m, k)) for k in FIELDS}
    r["bends"] = list(m.bends)
    return r


def test_calibration_fit_places_thresholds_between_clusters():
    from airpen.calibrate import fit
    rng = np.random.default_rng(0)

    def jitter(h):
        h.pts[:, :2] += rng.normal(0, 1.0, (21, 2))
        return h

    def taps(finger):  # 4 frames touching, 3 apart, repeated
        return [_row(jitter(make_hand(thumb_tip=tip(finger) if i % 7 < 4 else None))) for i in range(42)]
    S = {
        "open": [_row(jitter(make_hand())) for _ in range(30)],
        "fist": [_row(jitter(fist())) for _ in range(30)],
        "tap_index": taps("index"),
        "tap_middle": taps("middle"),
        "point": [_row(jitter(make_hand(extended=("index",), thumb_tip=TUCKED))) for _ in range(30)],
        "zoom": [_row(jitter(make_hand(extended=("index",), thumb_tip=(-100, -150)))) for _ in range(30)],
        "rest": [_row(jitter(make_hand(extended=(), thumb_tip=(-40, -110)))) for _ in range(30)],
    }
    out, notes = fit(S, Tuning())
    touch = np.median([r["tap_index"] for r in S["tap_index"][:4]])
    apart = min(r["tap_index"] for r in S["tap_index"][4:7])
    assert touch < out["tap_on"] < out["tap_off"] < apart
    assert "right_on" in out and out["right_on"] < out["right_off"]
    assert 90 <= out["extend_bend"] <= 200
    assert out["pointer_ref_scale"] > 0


def test_distance_estimate_from_world_landmarks():
    # synthetic frame is W px wide; 70 deg FOV; hand placed at 50 cm
    h = make_hand()
    f_px = (W / 2) / np.tan(np.radians(35))
    px_per_m = f_px / 0.5
    world = (h.pts - h.pts[0]) / px_per_m          # same shape, metres, flat to the camera
    world[:, 2] = 0
    m = compute_metrics(h.pts, h.norm, TUNE, world)
    assert m.distance_cm == pytest.approx(50, rel=0.02)


def test_pointer_zone_grows_when_hand_is_close():
    p = Pointer(box=(0.25, 0.2, 0.75, 0.7), ref_scale=0.1, adapt_tau=0.01)
    p.update(np.array([0.5, 0.5]), 0.0, hand_scale=0.1)
    w_ref = p.box[2] - p.box[0]
    p.update(np.array([0.5, 0.5]), 1.0, hand_scale=0.2)   # hand twice as big = closer
    assert p.box[2] - p.box[0] == pytest.approx(2 * w_ref, rel=0.01)
    p.update(np.array([0.5, 0.5]), 2.0, hand_scale=0.02)  # far: clamped at half
    assert p.box[2] - p.box[0] == pytest.approx(0.5 * w_ref, rel=0.01)


def test_label_flip_is_stabilized():
    from airpen.tracker import HandTracker
    tr = HandTracker.__new__(HandTracker)
    tr._last_wrist = {}
    a = make_hand("Right")
    tr._stabilize_labels([a], 0.0)
    b = make_hand("Left")                      # same spot, label flipped by the model
    tr._stabilize_labels([b], 0.033)
    assert b.label == "Right"


def test_rotated_hand_keeps_its_pose():
    for deg in (0, 45, 90, 180):
        h = rotate(make_hand(extended=("index",)), deg)
        assert classify_pose(compute_metrics(h.pts, h.norm, TUNE), TUNE) == "point"


def test_real_hand_recording_replay():
    """Recorded from a real hand (calibration run). With the default thresholds every tap is
    caught and nothing else clicks."""
    import json
    from pathlib import Path
    from airpen.calibrate import report
    S = json.loads((Path(__file__).parent / "data" / "real_hand_2026-11-04.json").read_text())
    lines = {ln.split()[0]: ln for ln in report(S, Tuning())}
    assert "left taps 11" in lines["tap_index"]
    assert "right taps 1" in lines["tap_middle"] and int(lines["tap_middle"].split("right taps ")[1]) >= 12
    for step in ("fist", "point", "zoom", "rest", "bunch"):
        assert "taps" not in lines[step], lines[step]
    assert "fist 9" in lines["fist"]            # >= 90 % read as fist
    assert "point 100%" in lines["point"]
    assert "curled 9" in lines["rest"]          # resting hand is not a point (no accidental scroll)


def test_return_stroke_does_not_undo_swipe():
    eng = GestureEngine()
    out = [[make_hand(at=(900 - 60 * i, 500))] for i in range(8)]      # swipe left
    back = [[make_hand(at=(480 + 60 * i, 500))] for i in range(8)]     # hand travels back right
    ev, _, _ = run(eng, out + back)
    assert [e.dir for e in ev if e.kind == "swipe"] == ["left"]


def test_hand_too_close_is_ignored():
    eng = GestureEngine()
    t = cursor_on(eng)
    close = make_hand(thumb_tip=tip("index"))
    close.pts[:, :2] = (close.pts[:, :2] - close.pts[0, :2]) * 6 + np.array([640, 700])  # fills the frame
    close.norm = close.pts[:, :2] / np.array([W, H])
    ev, _, _ = run(eng, [[close]] * 10, t)
    assert not [e for e in ev if e.kind == "press"]


def test_vertical_swipe_is_ignored():
    eng = GestureEngine()
    ev, _, _ = run(eng, [[make_hand(at=(640, 700 - 60 * i))] for i in range(10)])
    assert not [e for e in ev if e.kind == "swipe"]


def test_100k_engine_frames_obey_the_rules():
    rng = np.random.default_rng(99)
    poses = [dict(), dict(extended=("index",)), dict(extended=("index",), thumb_tip=TUCKED),
             dict(extended=(), thumb_tip=FIST), dict(thumb_tip=tip("index")), dict(thumb_tip=tip("middle")),
             dict(extended=("index", "middle")), dict(extended=())]
    eng = GestureEngine()
    t = 0.0
    pos = {"Left": np.array([400.0, 500.0]), "Right": np.array([880.0, 500.0])}
    cur = {"Left": 0, "Right": 0}
    held = set()
    frames = 0
    while frames < 100_000:
        present = [lab for lab in ("Left", "Right") if rng.random() < 0.6]
        for lab in present:
            if rng.random() < 0.5:
                cur[lab] = int(rng.integers(len(poses)))
        vel = {lab: rng.normal(0, 25, 2) for lab in present}
        too_close = rng.random() < 0.05
        for _ in range(int(rng.integers(3, 40))):
            hs = []
            for lab in present:
                pos[lab] = np.clip(pos[lab] + vel[lab], [150, 250], [1130, 700])
                if cur[lab] % 3 == 2 and rng.random() < 0.5:     # sometimes holding a pen, tapping randomly
                    h = pen_hand(lab, at=tuple(pos[lab]), lift_index=float(rng.choice([0, 0, 45])),
                                 lift_middle=float(rng.choice([0, 0, 60])))
                else:
                    h = make_hand(lab, at=tuple(pos[lab]), **poses[cur[lab]])
                if too_close:
                    h.pts[:, :2] = (h.pts[:, :2] - h.pts[0, :2]) * 6 + np.array([640, 700])
                h.pts[:, :2] += rng.normal(0, 2, (21, 2))
                h.norm = h.pts[:, :2] / np.array([W, H])
                hs.append(h)
            gf = eng.update(hs, t)
            t += 1 / 30
            frames += 1
            for e in gf.events:
                if e.kind in ("zoom", "zoom_start"):
                    assert len(hs) == 2 and not too_close, "zoom with fewer than two usable hands"
                if e.kind == "swipe":
                    assert e.dir in ("left", "right") and not too_close
                if e.kind == "press":
                    assert not too_close and (e.hand, e.button) not in held
                    held.add((e.hand, e.button))
                if e.kind == "release":
                    held.discard((e.hand, e.button))
    gf = None
    for _ in range(20):
        gf = eng.update([], t)
        t += 1 / 30
        for e in gf.events:
            if e.kind == "release":
                held.discard((e.hand, e.button))
    assert not held, "button stuck after hands left"


# ---------------------------------------------------------------- pen mode
def pen_hand(label="Right", at=(640, 600), lift_index=0.0, lift_middle=0.0):
    """Tripod pen grip: index bent around the pen, thumb against it, other fingers curled."""
    h = make_hand(label, at=(0, 0), extended=("index",))
    mx, my = MCP["index"]
    h.pts[6, :2] = (mx, my - 40)
    h.pts[7, :2] = (mx + 20, my - 60)
    h.pts[8, :2] = (mx + 42, my - 66 - lift_index)      # lift = finger rises off the pen
    h.pts[4, :2] = (mx + 14, my - 58)                    # thumb tip pressed on the index side (the pen)
    if lift_middle:
        h.pts[12, :2] = h.pts[12, :2] + np.array([0, -lift_middle])
    h.pts[:, :2] += np.array(at)
    h.norm = h.pts[:, :2] / np.array([W, H])
    return h


def pen_on(eng, t0=0.0):
    ev, gf, t = run(eng, [[pen_hand()]] * 20, t0)
    assert gf.cursor_on and gf.pen_mode, kinds(ev)
    assert ("pen_on", "") in kinds(ev)
    return t


def test_holding_a_pen_turns_cursor_on_and_letting_go_turns_it_off():
    eng = GestureEngine()
    t = pen_on(eng)
    ev, gf, _ = run(eng, [[make_hand()]] * 30, t)          # open hand: pen put down
    assert not gf.cursor_on and ("cursor_off", "") in kinds(ev)


def test_pen_index_tap_is_left_click():
    eng = GestureEngine()
    t = pen_on(eng)
    tap = [[pen_hand(lift_index=45)]] * 4 + [[pen_hand()]] * 6
    ev, gf, _ = run(eng, tap, t)
    assert kinds(ev).count(("press", "left")) == 1 and kinds(ev).count(("release", "left")) == 1
    assert ("press", "right") not in kinds(ev)


def test_pen_middle_tap_is_right_click():
    eng = GestureEngine()
    t = pen_on(eng)
    tap = [[pen_hand(lift_middle=60)]] * 4 + [[pen_hand()]] * 6
    ev, _, _ = run(eng, tap, t)
    assert kinds(ev).count(("press", "right")) == 1 and ("press", "left") not in kinds(ev)


def test_pen_still_or_moving_never_clicks():
    eng = GestureEngine()
    t = pen_on(eng)
    ev, gf, _ = run(eng, [[pen_hand(at=(500 + 8 * i, 600 - 4 * i))] for i in range(60)], t)
    assert not [e for e in ev if e.kind == "press"]
    assert gf.pointer_xy is not None                      # cursor follows the grip


def test_pen_finger_lifted_and_held_is_not_a_click():
    eng = GestureEngine()
    t = pen_on(eng)
    ev, _, _ = run(eng, [[pen_hand(lift_index=45)]] * 40 + [[pen_hand()]] * 10, t)
    assert not [e for e in ev if e.kind == "press"]


def test_pen_cursor_freezes_during_tap():
    eng = GestureEngine()
    t = pen_on(eng)
    ev, gf, _ = run(eng, [[pen_hand(lift_index=45)]] * 2, t)
    assert gf.freeze


def test_three_pen_taps_three_clicks():
    eng = GestureEngine()
    t = pen_on(eng)
    seq = ([[pen_hand(lift_index=45)]] * 4 + [[pen_hand()]] * 8) * 3
    ev, _, _ = run(eng, seq, t)
    assert kinds(ev).count(("press", "left")) == 3
