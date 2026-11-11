# AirPen audit — 2026-11-11

## Result

The live seven-action recognizer now connects to configurable PC controls. Existing uncommitted
work was retained. No desktop actions were executed during validation: integration tests use a
recording backend. No background service was started or restarted.

Recognition is **not yet validated for unattended desktop control**. The saved webcam session
`captures/20261107-220745` still fails its expected counts for six of nine steps. Neutral,
open-palm and fist steps pass. Both thumb-swipe steps still fail the full evaluation, including
no matching left events in the saved left-swipe clip; this is not a claim of solved real-world accuracy.
Synthetic tests passing does not establish real-world accuracy.

## Changes

- Added `airpen/controls.py` and an editable JSON binding example. Defaults provide left/right
  click, cursor toggle, workspace switching, media play/pause and pause/resume. Bindings also
  support shortcuts, scroll and program launcher argument arrays. Configuration errors fail
  startup before input devices open.
- Connected controls to the actual live loop and existing uinput/pynput backends. The adaptive
  pointer freezes during tap formation and fist/thumb controls, preserves its position across
  hand loss, and now uses the pointer sensitivity saved by calibration.
- Added repeat suppression and a bounded command worker so external programs do not block
  recognition. Commands never use a shell. Queued commands from before a pause are discarded;
  a command already started may finish. Commands exceeding three seconds are killed, so use
  short-lived launchers for applications.
- Added keyboard shortcuts/media keys to both input backends, attempted key release on partial
  failures, and closed partially constructed uinput devices. Automatic Wayland fallback now
  reports recognition-only operation rather than silently using an ineffective X11 backend.
- Input failure disables PC output while recognition continues. Camera recovery clears pointer
  tracking. Cleanup attempts every resource even if another device fails to close.
- Reject malformed/nonfinite/degenerate landmarks and uncertain handedness. Duplicate timestamps
  cannot accumulate confirmation frames. Recently tracked identities take precedence over
  screen position when two hands cross; ambiguous duplicate labels are not assigned by position.
- Vectorized the twelve finger joint-angle calculations. No change to angle definitions or
  classification thresholds. Existing gesture and geometry tests validate the shared metrics.
- Tightened tuning validation for hysteresis ordering, smoothing bounds and integer frame counts.
  Fixed idle frame limits and immediate preview-signal handling. Limited concurrent notification
  subprocesses. Added a timeout to camera framerate setup.
- Replayed recorded landmarks and added rejected-frame and tap-gating diagnostics. Empty or
  duplicate-step captures no longer pass as successful evaluations.
- Corrected the stylus width allowance after morphology: a narrow, antialiased diagonal stroke
  can be wider than the original cutoff. Kept the existing ridge, aspect and hand-outline checks.
  Pen detection belongs to the older pen engine; the live seven-action path does not use it.
- Replaced the direct OpenCV dependency with `opencv-contrib-python`, which MediaPipe already
  requires, avoiding overlapping OpenCV distributions on new installs. Model downloads now use
  a temporary file, so an interrupted download does not become the installed model.

## Remaining findings

1. **Personal tap calibration and pose mismatch.** The recording's index-to-thumb distances do
   not reliably enter the current press threshold. Its middle-tap poses fold the ring finger,
   which disables taps under the current recognizer rules and can resemble peace. Follow the
   documented pose and run `./run.sh calibrate`, then make a new recording. Supporting relaxed,
   folded-finger taps needs independently labeled examples of taps, peace and ordinary hand
   motion; simply removing the exclusion risks unintended clicks.
2. **Thumb detection still needs live verification.** The swipe implementation now keeps history
   in anatomical 3D coordinates and maps the final direction to camera left/right, preventing
   an edge-on palm from flipping the history. It uses a settled rest, a three-sample median,
   coherent short-stroke confirmation, and position-based return suppression rather than a
   blind timer. Short tracking dropouts clear motion history. The estimated distance guard now
   needs corroborating image-size evidence before rejecting an otherwise modest-sized hand.
   Controlled regressions cover both hands/directions, 10–60 fps, small deliberate strokes,
   palm rotation, landmark jumps and delayed/slow returns. The saved clips still contain misses
   and false actions, especially on the left. Fresh live validation is required; the aggregate
   expected counts were retained, not adjusted to make replay pass.
3. **Aggregate targets are not accuracy labels.** Each recording step contains a target count,
   not exact event times. Counts alone cannot establish precision, recall or latency. Some peace
   steps may contain fewer complete release/repeat cycles than the target assumes. Keep complete
   releases between repetitions and collect independent sessions before claiming an accuracy rate.
4. **Gesture overlap is visible in default mappings.** Fist toggles the cursor and is also the
   prerequisite for thumb gestures. Set the fist binding to `none` for shortcut-only use. Peace
   pauses/resumes only when that binding remains assigned; terminal stop always remains available.
   Clicks are discrete; there is no drag operation in this seven-action path.
5. **Platform/hardware limits.** The launcher uses Linux-specific locks, signals and `/proc`.
   A portable pynput backend does not make the whole application Windows/macOS compatible.
   Native Wayland input needs working uinput permissions. Real cursor delivery, monitor mapping,
   camera disconnect/replug, lighting variation and Arduino UNO Q hardware remain unverified.

## Validation

- Full automated suite after the swipe/camera follow-up: **226 passed**
  (`.venv/bin/python -m pytest -q`). Eight targeted new regressions were also run against
  the pre-fix recognizer and failed there, confirming they exercise actual prior defects.
  Includes the unchanged 2,000-scene stylus stress test and live-loop integration tests.
- Actual MediaPipe model initialization, CPU inference on a blank image and tracker cleanup passed.
- Shell syntax, Python compilation, example binding validation, CLI argument checks, dependency
  compatibility and whitespace checks passed. The duplicate OpenCV package was also removed
  from the local environment; the contrib package was reinstalled at the same tested version.
- Saved webcam replay was executed and intentionally remains FAIL; targets were not weakened.

Start recognition without desktop control:

```sh
./run.sh fg --backend none
./run.sh calibrate
./run.sh record
./run.sh replay
```

After checking recognition, start PC controls with `./run.sh fg --backend auto`. Edit the
bindings described in README.md to select programs and shortcuts on your own desktop.

## Follow-up: thumb swipes and normal camera preview

The default preview displays the original mirrored color camera image with a lightweight
color hand/finger overlay. The monochrome HandFX renderer and its keyboard toggle remain out
of the live preview path. Calibration and recording retain color and native frame resolution,
with hand landmarks, instruction text and progress added. Preview-only frame processing/FPS
bookkeeping from the monochrome renderer was removed from the main loop.

The thumb-swipe follow-up changes above address reproducible algorithm defects. They do not
establish reliable left/right recognition on the saved real-world clips. Try a new guided
recording after the update, including a clear reset to the starting thumb position between
swipes, and check replay before enabling workspace bindings.

## Follow-up: hand UI and action notifications

Restored hand skeletons, distinct finger colors, joint markers and fingertip highlights over the
unaltered color camera background. Dominant/ignored hand labels and the dominant hand pose are
shown. A two-second on-screen message names each recognized action, including when the desktop
notification service is absent or disabled. Desktop notifications now name the gesture and hand.
An overlay failure falls back to the plain camera while recognition continues.

Validation: 72 focused overlay, runtime, replay and robustness tests passed, plus compilation
and whitespace checks. A synthetic two-hand frame was visually inspected. No real desktop
notifications or PC controls were fired during tests. The gesture recognition limitations above
remain unchanged by this UI update.
