"""Loads personal thresholds written by the calibration wizard."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import fields
from pathlib import Path

CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "airpen" / "tuning.json"
SAMPLES = CONFIG.with_name("calibration_samples.json")


def load_overrides() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f"[airpen] ignoring unreadable {CONFIG}: {exc}", file=sys.stderr)
        return {}


def load_tuning(tune):
    """Apply overrides onto a Tuning instance (unknown keys are ignored)."""
    names = {f.name for f in fields(tune)}
    for k, v in load_overrides().items():
        if k in names:
            setattr(tune, k, type(getattr(tune, k))(v))
    return tune
