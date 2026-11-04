"""Pen detector on synthetic camera images: white stylus in a hand, on dark and on white backgrounds."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_gestures import make_hand  # noqa: E402

from airpen.pen import PenDetector  # noqa: E402


def scene(bg, pen=True, skin=(120, 160, 210)):
    img = np.full((720, 1280, 3), bg, np.uint8)
    h = make_hand(at=(640, 600))
    P = h.pts[:, :2].astype(int)
    for a, b in ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17), (5, 8), (9, 12), (13, 16), (17, 20), (1, 4)):
        cv2.line(img, tuple(P[a]), tuple(P[b]), skin, 22)
    cv2.fillConvexPoly(img, cv2.convexHull(P[[0, 1, 5, 9, 13, 17]]), skin)
    grip = (P[4] + P[7] + P[8]) / 3
    tip = (grip + np.array([-90, -150])).astype(int)          # pen sticks out up-left past the fingers
    back = (grip + np.array([40, 70])).astype(int)
    if pen:
        cv2.line(img, tuple(back), tuple(tip), (245, 245, 245), 12, cv2.LINE_AA)
    return img, h, tip


def test_finds_pen_tip_on_dark_background():
    img, h, tip = scene((40, 35, 30))
    s = PenDetector().detect(img, h.pts)
    assert s is not None and np.linalg.norm(s.tip - tip) < 25


def test_finds_pen_on_grey_background():
    img, h, tip = scene((150, 150, 150))
    s = PenDetector().detect(img, h.pts)
    assert s is not None and np.linalg.norm(s.tip - tip) < 30


def test_white_wall_between_fingers_is_not_a_pen():
    img, h, _ = scene((235, 235, 235), pen=False)
    assert PenDetector().detect(img, h.pts) is None


def test_no_pen_no_detection():
    for bg in ((40, 35, 30), (235, 235, 235)):
        img, h, _ = scene(bg, pen=False)
        assert PenDetector().detect(img, h.pts) is None


def test_washed_out_skin_is_not_a_pen():
    img, h, _ = scene((40, 35, 30), pen=False, skin=(225, 228, 232))   # overexposed hand
    assert PenDetector().detect(img, h.pts) is None


def random_scene(rng, pen):
    bg = tuple(int(v) for v in rng.integers(10, 130, 3)) if rng.random() < 0.7 else (int(rng.integers(150, 250)),) * 3
    skin = tuple(int(v) for v in (rng.integers(80, 170), rng.integers(120, 200), rng.integers(170, 240)))
    img = np.full((720, 1280, 3), bg, np.uint8)
    h = make_hand(at=(int(rng.integers(400, 900)), int(rng.integers(500, 650))))
    P = h.pts[:, :2].astype(int)
    for a, b in ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17), (5, 8), (9, 12), (13, 16), (17, 20), (1, 4)):
        cv2.line(img, tuple(P[a]), tuple(P[b]), skin, int(rng.integers(18, 26)))
    cv2.fillConvexPoly(img, cv2.convexHull(P[[0, 1, 5, 9, 13, 17]]), skin)
    tip = None
    if pen:
        grip = (P[4] + P[7] + P[8]) / 3
        ang = rng.uniform(np.radians(200), np.radians(290))          # pointing up-left-ish, like writing
        L = rng.uniform(140, 230)
        u = np.array([np.cos(ang), np.sin(ang)])
        tip = (grip + u * L).astype(int)
        back = (grip - u * rng.uniform(40, 80)).astype(int)
        cv2.line(img, tuple(back), tuple(tip), (int(rng.integers(225, 256)),) * 3, int(rng.integers(9, 15)), cv2.LINE_AA)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, h, tip


def test_2000_random_scenes_detection_and_false_alarms():
    rng = np.random.default_rng(5)
    det = PenDetector()
    hits = misses = false_alarms = wrong_tip = 0
    dark_hits = dark_total = 0
    for i in range(2000):
        pen = i % 2 == 0
        img, h, tip = random_scene(rng, pen)
        s = det.detect(img, h.pts)
        if pen:
            dark = img[10, 10].mean() < 140
            dark_total += dark
            if s is None:
                misses += 1
            elif np.linalg.norm(s.tip - tip) > 35:
                wrong_tip += 1
            else:
                hits += 1
                dark_hits += dark
        elif s is not None:
            false_alarms += 1
    print(f"hits {hits}/1000  misses {misses}  wrong tip {wrong_tip}  false alarms {false_alarms}/1000  "
          f"dark-background hit rate {dark_hits}/{dark_total}")
    assert false_alarms <= 10                     # <= 1 % false pens
    assert dark_hits / dark_total >= 0.9          # pen in front of something darker: found >= 90 %
    assert wrong_tip <= 20
