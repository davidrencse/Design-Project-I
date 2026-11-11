"""User settings that are not thresholds (for now: which hand is dominant)."""
from __future__ import annotations

import json

from airpen.tuning_io import CONFIG, read_json_object

SETTINGS = CONFIG.with_name("settings.json")
HANDS = ("Left", "Right")


def _read() -> dict:
    return read_json_object(SETTINGS)


def norm_hand(value: str) -> str:
    """'left' / 'L' / 'Right' ... -> 'Left' / 'Right' (MediaPipe's handedness labels)."""
    v = str(value).strip().lower()
    if v in ("l", "left"):
        return "Left"
    if v in ("r", "right"):
        return "Right"
    raise ValueError(f"hand must be left or right, not {value!r}")


def load_hand(default: str = "Right") -> str:
    try:
        return norm_hand(_read().get("hand", default))
    except ValueError:
        return default


def save_hand(value: str) -> str:
    hand = norm_hand(value)
    data = _read()
    data["hand"] = hand
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(data, indent=2) + "\n")
    return hand
