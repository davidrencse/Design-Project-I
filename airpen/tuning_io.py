"""Loads personal thresholds written by the calibration wizard."""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import fields
from pathlib import Path

CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "airpen" / "tuning.json"
SAMPLES = CONFIG.with_name("calibration_samples.json")


def read_json_object(path: Path) -> dict:
    """A JSON file that must hold an object; anything else is warned about and ignored ({})."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f"[airpen] ignoring unreadable {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[airpen] ignoring {path}: expected a JSON object, got {type(data).__name__}", file=sys.stderr)
        return {}
    return data


def load_overrides() -> dict:
    return read_json_object(CONFIG)


def load_tuning(tune):
    """Apply overrides onto a Tuning instance. Unknown keys are ignored; a value that cannot be
    converted, is not finite, or is not positive (every threshold, duration and frame count is
    positive) is warned about and the default is kept."""
    defaults = {f.name: getattr(tune, f.name) for f in fields(tune)}
    names = set(defaults)
    for k, v in load_overrides().items():
        if k not in names:
            continue
        default = getattr(tune, k)
        try:
            if isinstance(v, bool) != isinstance(default, bool) or isinstance(v, (list, dict)):
                raise TypeError(f"expected {type(default).__name__}")
            if isinstance(default, int) and float(v) != int(float(v)):
                raise ValueError("frame counts must be integers")
            val = type(default)(v)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if not math.isfinite(val) or val <= 0:
                    raise ValueError("must be a finite number > 0")
        except (TypeError, ValueError, OverflowError) as exc:
            print(f"[airpen] ignoring {k}={v!r} in {CONFIG}: {exc}; keeping {default!r}", file=sys.stderr)
            continue
        setattr(tune, k, val)
    # Positive values alone cannot guarantee usable hysteresis or stable smoothing.
    for key in ("smooth", "min_hand_score"):
        if hasattr(tune, key) and getattr(tune, key) > 1:
            print(f"[airpen] ignoring {key}: must be <= 1", file=sys.stderr)
            setattr(tune, key, defaults[key])
    for lower, upper in (("tap_on", "tap_off"), ("right_on", "right_off"), ("tap_min", "tap_max")):
        if hasattr(tune, lower) and getattr(tune, lower) >= getattr(tune, upper):
            print(f"[airpen] ignoring {lower}/{upper}: thresholds must be increasing", file=sys.stderr)
            setattr(tune, lower, defaults[lower])
            setattr(tune, upper, defaults[upper])
    return tune
