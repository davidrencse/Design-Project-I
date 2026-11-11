"""Replay a recording (airpen/record.py) through the action engine and compare with what was expected.

    ./run.sh replay                       # newest capture under captures/
    .venv/bin/python -m airpen.replay captures/20261107-101500 --tol 1

Per step: each expected action count must be within +/- tol. Any action that was not expected
fails the step, except fist (3): a fist is the transitional pose of most gestures, so extra fist
fires are reported as info, not failures. Exit 0 if every step passes, 1 otherwise, 2 on bad input.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from airpen.record import ACTION_NAMES, N_PTS, ROOT
from airpen.settings import norm_hand

FIST_ID = 3


def newest_capture(root: Path) -> Path | None:
    dirs = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "meta.json").exists()) \
        if Path(root).is_dir() else []
    return dirs[-1] if dirs else None  # YYYYMMDD-HHMMSS names sort chronologically


def load_meta(cap_dir: Path) -> dict:
    """meta.json, validated: must be an object with a valid `hand` and a `steps` list of names."""
    meta = json.loads((Path(cap_dir) / "meta.json").read_text())
    if not isinstance(meta, dict):
        raise ValueError("meta.json is not a JSON object")
    for key in ("hand", "steps"):
        if key not in meta:
            raise ValueError(f"meta.json is missing required key {key!r}")
    meta["hand"] = norm_hand(meta["hand"])
    steps = meta["steps"]
    if not isinstance(steps, list) or not steps or not all(isinstance(s, str) and s and "/" not in s and "\\" not in s
                                              for s in steps):
        raise ValueError("meta.json 'steps' must be a nonempty list of step names")
    if len(set(steps)) != len(steps):
        raise ValueError("meta.json contains duplicate steps")
    if not isinstance(meta.get("expected", {}), dict):
        raise ValueError("meta.json 'expected' must be an object")
    return meta


def load_step(cap_dir: Path, step: str) -> dict[str, np.ndarray]:
    """One step's arrays, checked: t (n,), present (n,), pts (n,21,3), norm (n,21,2), world (n,21,3)."""
    with np.load(Path(cap_dir) / f"{step}.npz", allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    for key in ("t", "present", "pts", "norm", "world"):
        if key not in arr:
            raise ValueError(f"array {key!r} is missing")
    n = arr["t"].shape[0] if arr["t"].ndim == 1 else -1
    want = {"t": (n,), "present": (n,), "pts": (n, N_PTS, 3), "norm": (n, N_PTS, 2), "world": (n, N_PTS, 3)}
    for key, shape in want.items():
        if n < 0 or arr[key].shape != shape:
            raise ValueError(f"array {key!r} has shape {arr[key].shape}, expected "
                             f"{shape if n >= 0 else '(frames,)'}")
    if n == 0:
        raise ValueError("recording contains no frames")
    if not np.isfinite(arr["t"].astype(float)).all() or (np.diff(arr["t"].astype(float)) < 0).any():
        raise ValueError("timestamps 't' must be finite and non-decreasing")
    return arr


def hands_for_frame(arr: dict, i: int, hand: str) -> list:
    """Rebuild the tracker's hand objects for one recorded frame ([] when the hand was absent)."""
    if not bool(arr["present"][i]):
        return []
    world = arr["world"][i]
    return [SimpleNamespace(
        label=hand, score=1.0,
        pts=arr["pts"][i], norm=arr["norm"][i],
        world=None if np.isnan(world).any() else world,
    )]


def default_engine_factory(hand: str):
    """The engine exactly as the live service builds it (current personal thresholds)."""
    from airpen.actions import make_engine  # lazy: keeps this module importable without it
    return make_engine(hand)


def replay_step(arr: dict, hand: str, engine_factory, diagnostics=None) -> tuple[Counter, int]:
    """Feed one step to a fresh engine. Returns (action id counts, frames fed)."""
    eng = engine_factory(hand)
    got: Counter = Counter()
    ts = arr["t"]
    for i in range(len(ts)):
        hands = hands_for_frame(arr, i, hand)
        fr = eng.update(hands, float(ts[i]))
        if diagnostics is not None:
            if hands and not getattr(fr, "hand_seen", bool(hands)):
                diagnostics["rejected"] += 1
            if getattr(fr, "hand_seen", bool(hands)):
                diagnostics["accepted"] += 1
                diagnostics["tap_allowed"] += int(getattr(fr, "debug", {}).get("tap_ok", False))
        for ev in fr.events:
            got[int(ev.id)] += 1
    return got, len(ts)


def judge(expected: dict[int, int], got: Counter, tol: int) -> tuple[list[str], list[str]]:
    """Returns (problems, info). Problems fail the step."""
    problems, info = [], []
    for aid, exp in sorted(expected.items()):
        n = got.get(aid, 0)
        if abs(n - exp) > tol:
            problems.append(f"{ACTION_NAMES.get(aid, aid)} ({aid}): expected {exp}, got {n}")
    for aid, n in sorted(got.items()):
        if aid in expected or n == 0:
            continue
        if aid == FIST_ID:
            info.append(f"extra fist x{n} (tolerated)")
        else:
            problems.append(f"unexpected {ACTION_NAMES.get(aid, aid)} ({aid}) x{n}")
    return problems, info


def changed_tuning(meta: dict) -> list[str]:
    """Names of thresholds that differ between the recording's `tuning` and the current ones."""
    saved = meta.get("tuning")
    if not isinstance(saved, dict):
        return []
    from airpen.actions import effective_tuning
    now = effective_tuning()
    return sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))


def replay_dir(path, engine_factory=None, tol: int = 1) -> dict:
    """Replay every step of a capture. Returns a report dict; report['ok'] is the pass/fail verdict."""
    cap_dir = Path(path)
    factory = engine_factory or default_engine_factory
    meta = load_meta(cap_dir)
    hand = meta["hand"]
    report = {"capture": str(cap_dir), "hand": hand, "tol": tol, "steps": {}, "ok": True,
              "tuning_changed": changed_tuning(meta) if engine_factory is None else []}
    for step in meta["steps"]:
        entry = {"expected": {}, "got": {}, "frames": 0, "present": 0,
                 "ok": False, "problems": [], "info": []}
        try:
            expected = {int(k): int(v) for k, v in meta.get("expected", {}).get(step, {}).items()}
            entry["expected"] = expected
            arr = load_step(cap_dir, step)
            diagnostics = Counter()
            got, n = replay_step(arr, hand, factory, diagnostics)
            entry["diagnostics"] = dict(diagnostics)
            entry.update(got=dict(sorted(got.items())), frames=n, present=int(arr["present"].sum()))
            entry["problems"], entry["info"] = judge(expected, got, tol)
        except (OSError, KeyError, ValueError, TypeError, AttributeError, EOFError, zipfile.BadZipFile) as exc:
            entry["problems"] = [f"could not replay: {type(exc).__name__}: {exc}" if not str(exc)
                                 else f"could not replay: {exc}"]
        entry["ok"] = not entry["problems"]
        report["steps"][step] = entry
        report["ok"] = report["ok"] and entry["ok"]
    return report


def format_report(report: dict) -> str:
    lines = [f"capture {report['capture']}  hand {report['hand']}  tol +/-{report['tol']}", ""]
    if report.get("tuning_changed"):
        lines.insert(1, "note: replaying with CURRENT thresholds; they differ from the recording's: "
                        + ", ".join(report["tuning_changed"]))
    lines.append(f"{'step':<20}{'frames':>7}{'hand':>7}  result")
    for step, e in report["steps"].items():
        lines.append(f"{step:<20}{e['frames']:>7}{e['present']:>7}  {'OK' if e['ok'] else 'FAIL'}")
        ids = sorted(set(e["expected"]) | set(e["got"]))
        for aid in ids:
            name = ACTION_NAMES.get(aid, str(aid))
            exp, got = e["expected"].get(aid, 0), e["got"].get(aid, 0)
            if aid in e["expected"]:
                flag = "" if abs(got - exp) <= report["tol"] else "   <- FAIL"
            elif aid == FIST_ID:
                flag = "   (extra fist, tolerated)"
            else:
                flag = "   <- unexpected"
            lines.append(f"    {aid} {name:<20} expected {exp:>3}  got {got:>3}{flag}")
        diag = e.get("diagnostics", {})
        if diag.get("rejected", 0):
            lines.append(f"    input: {diag['rejected']} present frames rejected (distance, size, confidence or invalid landmarks)")
        if not e["ok"] and set(e["expected"]) & {1, 2} and not diag.get("tap_allowed", 0):
            lines.append("    input: no tap-eligible frames; keep ring finger extended and recalibrate")
        for msg in e["problems"]:
            lines.append(f"    ! {msg}")
        for msg in e["info"]:
            lines.append(f"    . {msg}")
    lines.append("")
    lines.append("PASS" if report["ok"] else "FAIL")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="airpen.replay", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", help="capture folder (default: newest under captures/)")
    p.add_argument("--tol", type=int, default=1, help="allowed +/- on each expected count (default 1)")
    args = p.parse_args(argv)
    if args.tol < 0:
        p.error("--tol must be >= 0")
    cap = Path(args.dir) if args.dir else newest_capture(ROOT / "captures")
    if cap is None or not (cap / "meta.json").exists():
        print(f"no capture found ({cap or ROOT / 'captures'}); run ./run.sh record first", file=sys.stderr)
        return 2
    try:
        report = replay_dir(cap, tol=args.tol)
    except (OSError, ValueError) as exc:
        print(f"cannot read capture {cap}: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
