"""ActionEngine on synthetic hands (pixel coords, mirrored view, hand scale = 100 px)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_gestures import FIST, make_hand, tip  # noqa: E402

from airpen.actions import (  # noqa: E402
    ACTIONS, ActionEngine, ActionTuning, is_fist, is_peace_shape, tap_candidate,
)
from airpen.gestures import Tuning  # noqa: E402

W, H = 1280, 720
DT = 1 / 30
M_PER_PX = 0.0009           # world scale: a 100 px hand-scale = 9 cm
PEACE_THUMB = (-20, -55)    # thumb folded across the palm


def curl(p, z, fingers):
    """Fold fingers like a real fist: every joint bends ~80 deg towards the camera (total ~240 deg,
    real recorded fists read 240-270). test_gestures.make_hand only folds them ~180."""
    for k in fingers:
        base = 5 + 4 * k
        y, zz = p[base, 1], 0.0
        for j, ln in enumerate((35, 25, 20)):
            ang = np.radians(80 * (j + 1))
            y -= ln * np.cos(ang)
            zz += ln * np.sin(ang)
            p[base + 1 + j, 1], z[base + 1 + j] = y, zz


def mk(label="Right", at=(640, 600), pose="open", thumb=(0.0, 0.0), lift=0.0, world=True, mirror=None):
    """Synthetic hand. `thumb` is an image-space offset (px) of the thumb tip, in the USER's frame
    (+x = right) whichever hand it is; `lift` raises the thumb off the fist (3D when `world`,
    otherwise an image-space shift). A Left hand is the Right hand mirrored in x."""
    if pose == "open":
        h = make_hand(label, at=at)
    elif pose == "fist":
        h = make_hand(label, at=at, extended=(), thumb_tip=FIST)
    elif pose == "peace":
        h = make_hand(label, at=at, extended=("index", "middle"), thumb_tip=PEACE_THUMB)
        h.pts[8, 0] -= 25                      # V: index tip leans left, middle tip right
        h.pts[12, 0] += 25
    elif pose == "tap_index":
        h = make_hand(label, at=at, thumb_tip=tip("index"))
    elif pose == "tap_middle":
        h = make_hand(label, at=at, thumb_tip=tip("middle"))
    else:
        raise ValueError(pose)
    p = h.pts
    z = np.zeros(21)
    if pose == "fist":
        curl(p, z, (0, 1, 2, 3))
    elif pose == "peace":
        curl(p, z, (2, 3))
    if (mirror if mirror is not None else label == "Left"):
        p[:, 0] = 2 * at[0] - p[:, 0]
    wrist = p[0].copy()
    if pose == "fist":
        p[4, 0] += thumb[0]
        p[4, 1] += thumb[1]
        if lift and world:
            z[4] = -lift * 100                 # px-equivalent: towards the camera
        elif lift:
            p[4, 1] -= lift * 110
    p[:, 2] = z / Tuning().z_weight                # image-only fallback sees the same 3D shape
    h.norm = p[:, :2] / np.array([W, H])
    h.world = np.column_stack([(p[:, :2] - wrist[:2]) * M_PER_PX, z * M_PER_PX]) if world else None
    return h


def drive(eng, seq, t0=0.0):
    """seq: list of frames; a frame is a hand or a list of hands. Returns ([ids], last ActionFrame, end t)."""
    ids, af, t = [], None, t0
    for fr in seq:
        hands = fr if isinstance(fr, list) else [fr]
        af = eng.update(hands, t)
        ids += [e.id for e in af.events]
        t += DT
    return ids, af, t


def swipe_seq(dx, n_stroke=5, hold=6, **kw):
    """Thumb travels dx px sideways in n_stroke frames and rests there."""
    return ([mk(pose="fist", thumb=(dx * (i + 1) / n_stroke, 0), **kw) for i in range(n_stroke)]
            + [mk(pose="fist", thumb=(dx, 0), **kw)] * hold)


def fist_settled(n=30, **kw):
    return [mk(pose="fist", **kw)] * n


HANDS = ["Right", "Left"]


def test_action_table():
    assert ACTIONS == {1: "index_tap", 2: "middle_tap", 3: "fist", 4: "thumb_swipe_right",
                       5: "thumb_swipe_left", 6: "peace", 7: "thumb_tap"}


@pytest.mark.parametrize("hand", HANDS)
def test_open_hand_still_emits_nothing(hand):
    ids, af, _ = drive(ActionEngine(hand), [mk(hand)] * 90)
    assert ids == [] and af.hand_seen and af.pose == "open" and not af.in_fist


@pytest.mark.parametrize("hand", HANDS)
@pytest.mark.parametrize("world", [True, False])
def test_index_tap(hand, world):
    seq = [mk(hand, world=world)] * 10 + [mk(hand, pose="tap_index", world=world)] * 4 + [mk(hand, world=world)] * 6
    assert drive(ActionEngine(hand), seq)[0] == [1]


@pytest.mark.parametrize("hand", HANDS)
def test_middle_tap_is_never_also_an_index_tap(hand):
    seq = [mk(hand)] * 10 + [mk(hand, pose="tap_middle")] * 4 + [mk(hand)] * 6
    assert drive(ActionEngine(hand), seq)[0] == [2]


def test_repeated_taps_counted_individually():
    one = [mk(pose="tap_index")] * 4 + [mk()] * 5
    two = [mk(pose="tap_middle")] * 4 + [mk()] * 5
    ids, _, _ = drive(ActionEngine("Right"), [mk()] * 10 + one * 3 + two * 2)
    assert ids == [1, 1, 1, 2, 2]


def test_tap_held_down_fires_once():
    ids, _, _ = drive(ActionEngine("Right"), [mk()] * 5 + [mk(pose="tap_index")] * 40 + [mk()] * 5)
    assert ids == [1]


@pytest.mark.parametrize("hand", HANDS)
def test_fist_alone_emits_only_3_and_latches(hand):
    eng = ActionEngine(hand)
    ids, af, _ = drive(eng, [mk(hand)] * 5 + fist_settled(90, label=hand))
    assert ids == [3] and af.in_fist and af.pose == "fist"


def test_short_fist_is_not_a_fist():
    ids, af, _ = drive(ActionEngine("Right"), [mk()] * 5 + fist_settled(6) + [mk()] * 10)
    assert ids == [] and not af.in_fist


def test_fist_unlatches_when_hand_opens_and_can_fire_again():
    eng = ActionEngine("Right")
    ids, af, t = drive(eng, fist_settled(20))
    assert af.in_fist
    ids2, af, t = drive(eng, [mk()] * 10, t)
    assert not af.in_fist and ids2 == []
    ids3, af, _ = drive(eng, fist_settled(20), t)
    assert ids3 == [3] and af.in_fist


def test_one_frame_landmark_flicker_does_not_unlatch():
    eng = ActionEngine("Right")
    seq = fist_settled(20) + [mk()] + fist_settled(20)
    ids, af, _ = drive(eng, seq)
    assert ids == [3] and af.in_fist


@pytest.mark.parametrize("hand", HANDS)
def test_thumb_swipe_right_is_the_same_for_both_hands(hand):
    ids, _, _ = drive(ActionEngine(hand), fist_settled(25, label=hand) + swipe_seq(+40, label=hand))
    assert ids == [3, 4]


@pytest.mark.parametrize("hand", HANDS)
def test_thumb_swipe_left_is_the_same_for_both_hands(hand):
    ids, _, _ = drive(ActionEngine(hand), fist_settled(25, label=hand) + swipe_seq(-40, label=hand))
    assert ids == [3, 5]


@pytest.mark.parametrize("dx,aid", [(+40, 4), (-40, 5)])
def test_swipe_return_stroke_not_counted(dx, aid):
    back = [mk(pose="fist", thumb=(dx * (1 - (i + 1) / 5), 0)) for i in range(5)]
    seq = fist_settled(25) + swipe_seq(dx, hold=4) + back + fist_settled(10)
    ids, _, _ = drive(ActionEngine("Right"), seq)
    assert ids == [3, aid]


def test_two_deliberate_swipes_both_count_after_the_return_block():
    right = swipe_seq(+40, hold=6)
    back = [mk(pose="fist", thumb=(40 * (1 - (i + 1) / 5), 0)) for i in range(5)]
    seq = fist_settled(25) + right + back + fist_settled(30) + swipe_seq(-40) + fist_settled(5)
    ids, _, _ = drive(ActionEngine("Right"), seq)
    assert ids[0] == 3 and ids[1] == 4 and ids[-1] == 5 and 7 not in ids


def test_small_thumb_wiggle_is_not_a_swipe():
    ids, _, _ = drive(ActionEngine("Right"), fist_settled(25) + swipe_seq(+10) + swipe_seq(-10))
    assert ids == [3]


def test_vertical_thumb_motion_is_not_a_swipe():
    seq = fist_settled(25) + [mk(pose="fist", thumb=(0, -6 * (i + 1))) for i in range(6)] + fist_settled(10)
    ids, _, _ = drive(ActionEngine("Right"), seq)
    assert 4 not in ids and 5 not in ids


def test_no_thumb_actions_without_a_fist():
    # open hand: thumb moves around freely, nothing happens
    seq = [mk()] * 10 + [make_hand(thumb_tip=(-80 + 5 * i, -105)) for i in range(20)] * 2
    assert drive(ActionEngine("Right"), seq)[0] == []


@pytest.mark.parametrize("hand", HANDS)
@pytest.mark.parametrize("world", [True, False])
def test_thumb_tap(hand, world):
    up = [mk(hand, pose="fist", lift=0.5, world=world)] * 4
    rest = [mk(hand, pose="fist", world=world)]
    ids, _, _ = drive(ActionEngine(hand), rest * 30 + up + rest * 8)
    assert ids == [3, 7]


def test_repeated_thumb_taps_counted_individually():
    up = [mk(pose="fist", lift=0.5)] * 4
    rest = [mk(pose="fist")]
    ids, _, _ = drive(ActionEngine("Right"), rest * 30 + (up + rest * 8) * 3)
    assert ids == [3, 7, 7, 7]


def test_thumb_held_up_is_not_a_tap():
    ids, _, _ = drive(ActionEngine("Right"), fist_settled(30) + [mk(pose="fist", lift=0.5)] * 40)
    assert ids == [3]


def test_swipe_is_never_also_a_tap():
    ids, _, _ = drive(ActionEngine("Right"), fist_settled(30) + swipe_seq(+40, hold=3)
                      + [mk(pose="fist", thumb=(40 * (1 - (i + 1) / 5), 0)) for i in range(5)] + fist_settled(10))
    assert ids == [3, 4]


def test_holding_a_fist_still_emits_no_thumb_actions():
    rng = np.random.default_rng(3)
    seq = [mk(pose="fist", thumb=tuple(rng.normal(0, 0.8, 2))) for _ in range(120)]    # ~0.8 px landmark jitter
    ids, _, _ = drive(ActionEngine("Right"), seq)
    assert ids == [3]


def test_whole_hand_translation_in_a_fist_is_not_a_swipe():
    for dx in (+9, -9):
        seq = [mk(pose="fist", at=(640 + dx * i, 600)) for i in range(40)]
        eng = ActionEngine("Right")
        ids, _, _ = drive(eng, fist_settled(25) + seq)
        assert ids == [3], (dx, ids)


def test_whole_hand_translation_open_is_nothing():
    ids, _, _ = drive(ActionEngine("Right"), [mk(at=(300 + 10 * i, 500)) for i in range(40)])
    assert ids == []


@pytest.mark.parametrize("hand", HANDS)
def test_peace_fires_once_and_is_not_a_tap(hand):
    ids, af, _ = drive(ActionEngine(hand), [mk(hand)] * 5 + [mk(hand, pose="peace")] * 90)
    assert ids == [6] and af.pose == "peace"


def test_peace_twice_counts_twice():
    p = [mk(pose="peace")] * 15
    ids, _, _ = drive(ActionEngine("Right"), p + [mk()] * 10 + p)
    assert ids == [6, 6]


def test_two_fingers_together_is_not_peace():
    h = make_hand(extended=("index", "middle"), thumb_tip=PEACE_THUMB)      # parallel fingers
    ids, af, _ = drive(ActionEngine("Right"), [h] * 30)
    assert ids == [] and af.pose != "peace"


def test_peace_then_fist_emits_no_taps():
    ids, _, _ = drive(ActionEngine("Right"), [mk(pose="peace")] * 15 + fist_settled(30))
    assert ids == [6, 3]


def test_fist_then_peace_emits_no_taps():
    ids, _, _ = drive(ActionEngine("Right"), fist_settled(30) + [mk(pose="peace")] * 15)
    assert ids == [3, 6]


def test_thumb_on_the_v_fingers_is_not_peace_nor_a_tap():
    h = make_hand(extended=("index", "middle"), thumb_tip=tip("middle"))
    ids, _, _ = drive(ActionEngine("Right"), [h] * 30)
    assert ids == []


def test_non_dominant_hand_is_ignored():
    eng = ActionEngine("Right")
    other = "Left"
    seq = ([mk(other)] * 10 + [mk(other, pose="tap_index")] * 4 + [mk(other)] * 6
           + fist_settled(30, label=other) + swipe_seq(+40, label=other) + [mk(other, pose="peace")] * 15)
    ids, af, _ = drive(eng, seq)
    assert ids == [] and not af.hand_seen


def test_non_dominant_hand_does_not_disturb_dominant():
    eng = ActionEngine("Right")
    left = mk("Left", at=(300, 600), pose="peace")
    ids, _, _ = drive(eng, [[mk("Right"), left]] * 10 + [[mk("Right", pose="tap_middle"), left]] * 4
                      + [[mk("Right"), left]] * 6)
    assert ids == [2]


def test_losing_the_hand_resets_state():
    eng = ActionEngine("Right")
    ids, af, t = drive(eng, fist_settled(25))
    assert af.in_fist
    ids, af, t = drive(eng, [[]] * 12, t)                  # 0.4 s with no hand
    assert not af.hand_seen and not af.in_fist
    ids, af, _ = drive(eng, [mk()] * 5, t)
    assert ids == [] and not af.in_fist


def test_short_dropout_keeps_the_latch():
    eng = ActionEngine("Right")
    _, _, t = drive(eng, fist_settled(25))
    _, af, t = drive(eng, [[]] * 3, t)                    # 0.1 s
    assert af.in_fist and not af.hand_seen
    _, af, _ = drive(eng, fist_settled(3), t)
    assert af.in_fist


def test_hand_too_close_to_camera_is_ignored():
    h = mk()
    c = h.pts[:, :2].mean(axis=0)
    h.pts[:, :2] = (h.pts[:, :2] - c) * 6 + c
    h.norm = h.pts[:, :2] / np.array([W, H])
    h.world = None
    ids, af, _ = drive(ActionEngine("Right"), [h] * 20)
    assert not af.hand_seen


def test_world_attribute_missing_is_fine():
    h = make_hand()                       # SimpleNamespace without .world
    assert not hasattr(h, "world")
    ids, af, _ = drive(ActionEngine("Right"), [h] * 5)
    assert af.hand_seen and af.metrics is not None


def test_reset_and_hand_names():
    eng = ActionEngine("left")
    assert eng.hand == "Left"
    drive(eng, fist_settled(25, label="Left"))
    assert eng.in_fist
    eng.reset()
    assert not eng.in_fist
    with pytest.raises(ValueError):
        ActionEngine("middle")


def test_tuning_json_overrides_reach_action_tuning(tmp_path, monkeypatch):
    import airpen.tuning_io as tio
    f = tmp_path / "tuning.json"
    f.write_text(json.dumps({"thumb_swipe_dist": 0.5, "fist_hold_s": 0.1, "tap_on": 0.4}))
    monkeypatch.setattr(tio, "CONFIG", f)
    a = tio.load_tuning(ActionTuning())
    assert a.thumb_swipe_dist == 0.5 and a.fist_hold_s == 0.1
    assert tio.load_tuning(Tuning()).tap_on == 0.4
    # a bigger swipe threshold really suppresses the 0.4-hand-scale swipe
    ids, _, _ = drive(ActionEngine("Right", act=a), fist_settled(25) + swipe_seq(+40))
    assert 4 not in ids


def test_no_name_collisions_between_tunings():
    from dataclasses import fields
    assert not {f.name for f in fields(ActionTuning)} & {f.name for f in fields(Tuning)}


# --------------------------------------------------------------- real recorded hand
REAL = json.loads((Path(__file__).parent / "data" / "real_hand_2026-11-04.json").read_text())


def _metrics(step):
    T = Tuning()
    return [SimpleNamespace(**{**s, "extended": tuple(b < T.extend_bend for b in s["bends"])})
            for s in REAL[step]]


def _frac(step, fn):
    ms = _metrics(step)
    return sum(bool(fn(m)) for m in ms) / len(ms)


def test_real_fist_reads_fist_and_other_poses_do_not():
    T, A = Tuning(), ActionTuning()
    assert _frac("fist", lambda m: is_fist(m, T)) >= 0.9
    for step in ("open", "point", "tap_index", "tap_middle"):
        assert _frac(step, lambda m: is_fist(m, T)) == 0.0, step
    assert _frac("rest", lambda m: is_fist(m, T)) < 0.1
    for step in ("open", "rest", "point", "tap_index", "tap_middle", "fist"):
        assert _frac(step, lambda m: is_peace_shape(m, T, A)) == 0.0, step


def test_real_taps_pick_the_right_finger():
    T = Tuning()
    ti = [tap_candidate(m, T) for m in _metrics("tap_index")]
    tm = [tap_candidate(m, T) for m in _metrics("tap_middle")]
    assert ti.count("index") >= 8 and "middle" not in ti
    assert tm.count("middle") >= 8 and "index" not in tm
    for step in ("fist", "point", "rest", "zoom"):
        assert not any(tap_candidate(m, T) for m in _metrics(step)), step


# ---- audit D#2: a hand reacquired after a gap is a NEW acquisition
def test_fist_hold_does_not_accumulate_across_a_gap():
    e = ActionEngine()
    e.update([mk(pose="fist")], 0)
    e.update([], .1)
    assert [v.id for v in e.update([mk(pose="fist")], 1).events] == []


def test_tap_debounce_does_not_accumulate_across_a_gap():
    e = ActionEngine(tune=Tuning(smooth=1))
    out = []
    for hs, t in [([mk(pose="tap_index")], 0), ([mk(pose="tap_index")], .03), ([], .1),
                  ([mk(pose="tap_index")], 1)]:
        out += [v.id for v in e.update(hs, t).events]
    assert out == []


def test_latched_fist_expires_after_a_gap():
    e = ActionEngine()
    ids, af, t = drive(e, fist_settled(30))
    assert ids == [3] and af.in_fist
    af = e.update([mk(pose="fist")], t + 4.0)              # same pose, but 4 s later: re-acquired
    assert not af.events and not af.in_fist                # not latched until it is held again


# ---- audit D#6: a whole-hand move on the tap's return frame is not a thumb tap
def test_moving_palm_on_thumb_return_frame_is_not_a_tap():
    e = ActionEngine()
    ids, _, t = drive(e, fist_settled(30) + [mk(pose="fist", lift=.5)] * 4)
    f = e.update([mk(pose="fist", at=(720, 600))], t)
    assert f.debug["palm_speed"] > ActionTuning().thumb_palm_still and 7 not in [v.id for v in f.events]
    assert 7 not in ids


def test_still_return_frame_is_still_a_tap():
    e = ActionEngine()
    ids, _, t = drive(e, fist_settled(30) + [mk(pose="fist", lift=.5)] * 4 + [mk(pose="fist")] * 2)
    assert ids.count(7) == 1


def test_tap_detector_cancels_when_hand_moves_on_return():
    from airpen.gestures import TapDetector
    d = TapDetector(Tuning())
    t = 0.0
    for _ in range(10):
        d.update(0.0, t, True); t += .03
    for _ in range(3):
        d.update(1.0, t, True); t += .03
    assert d.active
    assert d.update(0.0, t, False) is False and not d.active
