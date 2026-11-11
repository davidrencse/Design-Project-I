"""Regression cases for realistic thumb swipes rather than only perfect fast strokes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_actions import mk, fist_settled, swipe_seq, drive, W, H
from airpen.actions import ActionEngine


@pytest.mark.parametrize("hand", ["Left", "Right"])
@pytest.mark.parametrize("fps", [10, 15, 24, 30, 60])
@pytest.mark.parametrize("world", [False, True])
def test_both_directions_with_slow_strokes_at_different_frame_rates(hand, fps, world):
    def rest(seconds):
        return [mk(hand, pose="fist", world=world) for _ in range(round(fps * seconds))]
    def stroke(dx):
        n = max(3, round(fps * 0.35))
        return [mk(hand, pose="fist", world=world, thumb=(dx * (i + 1) / n, 0)) for i in range(n)]
    def hold(dx, seconds):
        return [mk(hand, pose="fist", world=world, thumb=(dx, 0)) for _ in range(round(fps * seconds))]
    def back(dx):
        n = max(3, round(fps * .4))
        return [mk(hand, pose="fist", world=world, thumb=(dx * (1 - (i + 1) / n), 0)) for i in range(n)]
    seq = rest(1.2) + stroke(40) + hold(40, .2) + back(40) + rest(.5) + stroke(-40) + hold(-40, .2)
    eng, ids = ActionEngine(hand), []
    for i, h in enumerate(seq):
        ids.extend(e.id for e in eng.update([h], i / fps).events)
    assert ids == [3, 4, 5]


@pytest.mark.parametrize("dx,aid", [(40, 4), (-40, 5)])
def test_very_slow_return_never_fires_opposite_swipe(dx, aid):
    back = [mk(pose="fist", thumb=(dx * (1 - (i + 1) / 60), 0)) for i in range(60)]
    ids, _, _ = drive(ActionEngine(), fist_settled(30) + swipe_seq(dx, hold=3) + back + fist_settled(20))
    assert ids == [3, aid]


def test_one_frame_thumb_teleport_is_not_a_swipe():
    ids, _, _ = drive(ActionEngine(), fist_settled(30) + [mk(pose="fist", thumb=(90, 0))] + fist_settled(30))
    assert ids == [3]


@pytest.mark.parametrize("hand", ["Left", "Right"])
def test_rotating_entire_fist_through_edge_on_does_not_swipe(hand):
    seq = fist_settled(30, label=hand)
    for angle in np.linspace(0, 135, 90):
        h = mk(hand, pose="fist")
        theta = np.radians(angle)
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        h.world = h.world @ rotation.T
        relative = h.pts - h.pts[0]
        relative[:, 2] *= .6  # undo the image-only depth weight used by the fixture
        relative = relative @ rotation.T
        h.pts = relative + h.pts[0]
        h.pts[:, 2] /= .6
        h.norm = h.pts[:, :2] / [W, H]
        seq.append(h)
    assert drive(ActionEngine(hand), seq)[0] == [3]


def test_unreliable_metric_distance_alone_does_not_discard_a_small_hand():
    import copy
    seq = [copy.deepcopy(h) for h in fist_settled(30) + swipe_seq(40)]
    for h in seq:
        h.world = h.world * .1
    ids, frame, _ = drive(ActionEngine(), seq)
    assert frame.metrics.distance_cm < 15
    assert ids == [3, 4]


def test_reacquisition_cannot_join_two_halves_of_a_swipe():
    seq = fist_settled(30) + [[]] * 3 + [mk(pose="fist", thumb=(40, 0))] * 20
    assert drive(ActionEngine(), seq)[0] == [3]


def test_guides_keep_camera_colors_and_resolution():
    from airpen.ui import camera_guide
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[:] = (20, 100, 180)
    guide = camera_guide(image, ["Record your hand", "Swipe thumb right"], .5)
    assert guide.shape == image.shape
    np.testing.assert_array_equal(guide[200:400], image[200:400])
    assert np.any(guide[:, :, 0] != guide[:, :, 2])
    np.testing.assert_array_equal(image[0, 0], [20, 100, 180])


@pytest.mark.parametrize("dx,aid", [(18, 4), (-18, 5)])
@pytest.mark.parametrize("hand", ["Left", "Right"])
def test_short_deliberate_thumb_strokes_are_detected(hand, dx, aid):
    seq = fist_settled(30, label=hand) + swipe_seq(dx, n_stroke=8, hold=5, label=hand)
    assert drive(ActionEngine(hand), seq)[0] == [3, aid]


@pytest.mark.parametrize("dx,aid", [(40, 4), (-40, 5)])
def test_delayed_return_after_old_cooldown_is_still_only_a_reset(dx, aid):
    back = [mk(pose="fist", thumb=(dx * (1 - (i + 1) / 5), 0)) for i in range(5)]
    seq = fist_settled(30) + swipe_seq(dx, hold=40) + back + fist_settled(20)
    assert drive(ActionEngine(), seq)[0] == [3, aid]
