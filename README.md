# AirPen: hand gestures from your webcam

Your webcam watches your dominant hand and recognizes seven actions. Actions control the PC,
are written to the log, and optionally send desktop notifications. It runs in the
background, so no window has to stay open. Use `--backend none` to check recognition without
controlling the desktop. Recognition still needs personal calibration: the bundled webcam
recording does **not** pass all seven gesture checks; see [AUDIT.md](AUDIT.md).

## Start / stop

```
./run.sh            # start in the background (first run sets everything up)
./run.sh stop       # stop
./run.sh show       # open or hide the camera preview (closing it does NOT stop AirPen)
./run.sh status     # is it running?
./run.sh fg         # run in this terminal with the preview open, Ctrl+C to quit
./run.sh calibrate  # fit the hand-shape thresholds to YOUR hand (about 1 minute)
```

The preview keeps the normal, mirrored color camera feed, with hand bones, colored finger
joints and fingertip highlights overlaid. Your dominant hand is labeled with its current pose;
the other hand is marked ignored. Every recognized gesture shows its name in a two-second
on-screen notification, even if desktop notifications are unavailable. Desktop notifications
also name the gesture and hand by default; `--no-notify` disables only those desktop popups.
Recording and calibration keep the camera in color with hand tracking and instructions.
Closing the preview hides it; use `./run.sh stop` to stop gesture control.

To start it at every login, add this line to `~/.config/hypr/hyprland.conf`:

```
exec-once = "/home/czr/projects/DESIGN PROJ/Design-Project-I/run.sh" start
```

## Choose your dominant hand

AirPen watches only your dominant hand. Other hands in view are ignored. The default is right.

```
./run.sh hand           # print the current hand
./run.sh hand left      # use the left hand (saved to ~/.config/airpen/settings.json)
./run.sh hand right
```

Restart AirPen after changing it (`./run.sh stop && ./run.sh start`). You can also pass
`--hand left` to `python -m airpen.record` and `./run.sh record --hand left` for a one-off
recording.

## The 7 actions

Each action is logged as `ACTION <id> <name> <hand>`, e.g. `ACTION 4 thumb_swipe_right Right`.

| id | Name | Do this with your dominant hand |
|---|---|---|
| 1 | index_tap | Tap thumb and index finger together |
| 2 | middle_tap | Tap thumb and middle finger together |
| 3 | fist | Make a fist |
| 4 | thumb_swipe_right | Make a fist, then swipe the thumb to your right |
| 5 | thumb_swipe_left | Make a fist, then swipe the thumb to your left |
| 6 | peace | Raise index and middle fingers (peace sign) |
| 7 | thumb_tap | Make a fist, then tap the thumb |

For thumb swipes, make a fist, let the thumb rest briefly, swipe it sideways, then return it
to the starting position before the next swipe. Return strokes are ignored even if they are
slow. Opening the hand also resets the swipe state.
The old `thumb_return_block_s` tuning field has been replaced by position-based return detection
(`thumb_return_range`); a timer no longer decides whether a movement is a return.

Keep the rest of your hand relaxed while you do these. Other hand poses should not fire
anything. That is what the recording check below measures.

## PC controls and custom programs

Default bindings:

| Gesture | PC action |
|---|---|
| Index tap | Left click at current cursor |
| Middle tap | Right click at current cursor |
| Fist | Toggle hand-driven cursor; fist also freezes movement while thumb gestures are used |
| Thumb swipe right / left | Next / previous workspace |
| Peace | Pause / resume PC controls |
| Thumb tap | Media play / pause |

Move your open hand after toggling the cursor on. Removing the hand keeps the cursor in place;
reacquiring it preserves its position. Tap formation freezes the cursor. Clicks are discrete;
holding a pinch does not drag. Fist is also the prerequisite for thumb gestures, so making a new
fist toggles the cursor even when you intend to use a thumb gesture. Remap `fist` to `none` if
you only want shortcuts/workspace controls.

```sh
./run.sh fg --backend none       # recognize only; recommended for initial calibration
./run.sh fg --backend auto       # PC controls (also the normal start default)
./run.sh stop                    # stop background control from a terminal
mkdir -p ~/.config/airpen
cp config/bindings.example.json ~/.config/airpen/bindings.json
```

Edit `bindings.json`, then restart AirPen. Omitted gestures retain their defaults; use
`{"type": "none"}` to disable a gesture. `--bindings /path/to/file.json` selects a different
file. Invalid bindings stop startup before the camera or desktop backend is opened.
For example, replace only the bindings you want:

```json
{
  "thumb_tap": {"type": "hotkey", "keys": ["ctrl", "l"]},
  "thumb_swipe_right": {"type": "scroll", "amount": -120},
  "thumb_swipe_left": {"type": "command", "argv": ["gtk-launch", "firefox"]}
}
```

`ctrl+l` focuses the location bar in many browsers. Program commands use an argument array,
without a shell: use your installed program's launcher or desktop file ID. Commands run on one
worker with a four-command queue and a three-second timeout; use a launcher that returns
promptly. Long-running commands are terminated. Supported named keys: `ctrl`, `shift`, `alt`,
`super`, arrows, `space`, `enter`, `esc`, `tab`, `backspace`, `play_pause`, `volume_up`,
`volume_down`, `mute`; letters and digits also work.

On Linux, `auto` first tries `/dev/uinput` for native Wayland/X11 controls. Without permission,
Wayland falls back to recognition only and logs the reason. X11 can fall back to `pynput`.
An explicitly requested unavailable backend fails startup. This launcher currently targets Linux
(`fcntl`, POSIX signals, `/proc`); the `pynput` backend alone is not full Windows/macOS support.

## Checking accuracy on your own hand

Recognition depends on your hand, camera and lighting, so check it with your own recording.

```
./run.sh record                 # guided: 9 steps, about 1.5 minutes in total
./run.sh record --reps 8        # you do 8 reps per gesture step (sets the expected counts)
./run.sh replay                 # replay the newest recording and compare with what you did
./run.sh replay captures/<dir>  # replay a specific one
```

`record` stops AirPen while it runs and leaves it stopped afterwards. Restart with `./run.sh start`. Each step shows an
instruction, counts down 2 seconds, then records about 8 seconds. Press `q` or Esc to abort.
Raw landmarks go to `captures/<time>/`, one file per step plus `meta.json`. Nothing leaves your
machine.

`replay` feeds each step to the recognizer and prints expected and actual counts per action:

```
index_tap   ...  1 index_tap        expected   5  got   5
            ...  3 fist             (extra fist, tolerated)
```

A step fails when an expected count is off by more than 1, or when an action fires that the step
did not expect (an extra fist is only reported). Exit code 0 means pass, 1 means fail.
`--tol 0` makes it strict. Use it to find the gestures that need a clearer motion. Replay uses your
current thresholds, exactly like the live service. `meta.json` also stores the thresholds that
were active when you recorded, and replay prints a note if they have changed since.

## Hand-shape calibration

`./run.sh calibrate` records your hand in a few poses (about 1 minute) and fits the
hand-shape thresholds to it. Taps are split into "touching" and "released" frames. The press
threshold sits above your touching plateau; the release threshold sits under your quickest
release, since fast taps only open for a single frame. It then replays the recording and prints
what each pose was read as:

```
tap_index   read as: open 100% | left taps 11
tap_middle  read as: open 97%  | right taps 13
fist        read as: fist 98%
rest        read as: curled 97%
```

The results go to `~/.config/airpen/tuning.json`. Deleting that file resets to the built-in
defaults. `python -m airpen.calibrate --refit` re-fits from the saved recording without
recording again.

## How it works

```
webcam ─► MediaPipe hand landmarks (21 points, one dominant hand) ─► hand-shape measures ─► action engine
                                                                                        ├► preview
                                                                                        ├► log + notification
                                                                                        └► configured PC controls
```

- Hand shape (touches, finger bends) is measured on MediaPipe's metric 3D world landmarks using
  joint angles. That works at any distance and rotation.
- The ring finger must stay straight for a tap to count, so a fist or a relaxed hand does not
  register taps.
- Measurements are lightly smoothed over frames. A new pose has to win 2 of the last 3 frames
  to count. Left/right labels that flip for a frame are corrected, and dim frames are
  brightened before detection.
- After about 2 s with no hand in view, AirPen checks only every third frame to save battery.
- On Linux, AirPen stops the webcam from dropping to 15 fps in dim light
  (`exposure_dynamic_framerate`). Pass `--dynamic-fps` to leave the camera alone.

Tests: `.venv/bin/python -m pytest -q`. Synthetic tests cover gestures, PC mapping, camera
recovery and malformed inputs; they do not establish accuracy on your hand.
Dependencies use `opencv-contrib-python` (which includes `cv2`) to avoid installing two
OpenCV distributions into the same environment.

## Optional: Arduino UNO Q

`unoq_app/` is an App Lab app that shows AirPen's state on the UNO Q LED matrix. It's off unless
you start with `--unoq <board-ip>`. It is not tested on hardware yet.

## Agent skills

`.claude/skills/` holds 10 vetted OpenCV and computer-vision skills for Claude Code.
`yolo-inference` is AGPL and `add-hand-tracking` has no license, so don't redistribute those two.
