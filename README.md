# AirPen: control your Arch laptop with hand gestures

Your webcam watches your hands. Gestures move the real cursor, click, scroll, zoom and switch
workspaces in any app. It runs in the background, so no window has to stay open.

## Start / stop

```
./run.sh            # start in the background (first run sets everything up)
./run.sh stop       # stop
./run.sh show       # open or hide the camera preview (closing it does NOT stop AirPen)
./run.sh status     # is it running?
./run.sh fg         # run in this terminal with the preview open, Ctrl+C to quit
./run.sh calibrate  # fit every gesture threshold to YOUR hand (about 1 minute). Do this first.
```

The preview draws holographic linework on your hands: fingertip and palm reticles, finger
traces, and live rotation / distance / grip readouts. Press `l` to switch to the plain camera.

A desktop notification tells you whenever the mode changes.

To start it at every login, add this line to `~/.config/hypr/hyprland.conf`:

```
exec-once = "/home/czr/projects/DESIGN PROJ/Design-Project-I/run.sh" start
```

## Two modes

AirPen starts in **gesture mode**. **Hold a fist for about 0.6 s** to switch to **mouse mode**,
and do it again to switch back. If your hand leaves the camera for 2.5 s, AirPen drops back
to gesture mode by itself.

### Gesture mode (default)

| Do this | Get this |
|---|---|
| Open palm, swipe left / right | Next / previous workspace |
| Thumb + index spread apart / pinch together (other fingers folded) | Zoom in / out |
| Point index finger up / down and hold it still | Scroll up / down. Hold to scroll faster |

### Mouse mode

| Do this | Get this |
|---|---|
| Move your hand | Move the cursor |
| Tap thumb + index (keep the other fingers relaxed and straight) | Left click (hold = drag) |
| Tap thumb + middle | Right click |
| Point index finger up / down and hold it still | Scroll up / down |
| Short fist | Freeze the cursor (reposition your hand) |

### Any mode

| Do this | Get this |
|---|---|
| Both index fingers pointing, move them apart / together | Zoom in / out |
| Both palms open and still for 1.5 s | Pause / resume AirPen |

Zoom sends Ctrl + scroll wheel, which works in browsers, image viewers and most editors.
Workspaces switch with `hyprctl dispatch workspace e±1`.

**How it avoids false triggers:**
- **Ring finger:** in a real tap it stays straight. In a resting hand, a fist or a point it's
  curled, so those never click.
- **Zoom vs scroll:** zoom starts only when the thumb–index distance actually *moves*. A
  pointing finger has to be held still to scroll.
- **Pointing:** needs a fully straight index finger, so a relaxed, half-curled hand doesn't
  scroll.

## Accuracy and calibration

`./run.sh calibrate` records your hand doing each gesture (about 1 minute) and fits every
threshold to it. Taps are split into "touching" and "released" frames. The press threshold
sits above your touching plateau; the release threshold sits under your *quickest* release,
since fast taps only open for a single frame. Then it replays your recording through the
classifier and prints what each step was read as, along with any taps found:

```
tap_index   read as: open 100% | left taps 11      <- every tap caught
tap_middle  read as: open 97%  | right taps 13
fist        read as: fist 98%                      <- no false clicks
rest        read as: curled 97%                    <- resting hand doesn't scroll
```

The results go to `~/.config/airpen/tuning.json`, and deleting that file resets to the
built-in defaults. `python -m airpen.calibrate --refit` re-fits from the saved recording
without recording again. The built-in defaults come from a real calibration run, which is
also kept as a regression test (`tests/data/`).

## How it works

```
webcam 640x480 ─► MediaPipe hand landmarks (21 points, 2 hands) ─► gesture engine ─► smoothing (One Euro)
                                                                                    └► /dev/uinput virtual mouse + keyboard
```

- The cursor uses a virtual absolute tablet device, so it works on Wayland. You need to be in
  the `input` group, which you already are.
- The cursor tracks your knuckles, not your fingertip, so it stays still while you tap.
- Hand shape (touches, finger bends) is measured on MediaPipe's metric 3D world landmarks
  using joint angles. That works at any distance and rotation, and fingers lined up toward the
  camera don't register as touching.
- Hand distance is estimated in cm (pinhole model, real palm length vs its image length). The
  cursor zone scales with it, so control feels the same near or far.
- Measurements are lightly smoothed over frames. A new pose has to win 2 of the last 3 frames
  to count. Left/right labels that flip for a frame are corrected, and dim frames are
  brightened before detection.
- After about 2 s with no hands in view, AirPen checks only every third frame to save battery.
- On Linux, AirPen stops the webcam from dropping to 15 fps in dim light
  (`exposure_dynamic_framerate`). Pass `--dynamic-fps` to leave the camera alone.

Tests: `.venv/bin/python -m pytest -q` runs every gesture against synthetic hands.

## Optional: Arduino UNO Q

`unoq_app/` is an App Lab app that shows AirPen's state on the UNO Q LED matrix. It's off unless
you start with `--unoq <board-ip>`. It is not tested on hardware yet.

## Agent skills

`.claude/skills/` holds 10 vetted OpenCV and computer-vision skills for Claude Code.
`yolo-inference` is AGPL and `add-hand-tracking` has no license, so don't redistribute those two.
