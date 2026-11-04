# AirPen: real-time stroke input from camera + Arduino UNO Q

Evolution of the IMU stylus in `Write-Up.pdf`. The write-up's main limitation is that IMU
position drifts, because it comes from double-integrating acceleration. A camera measures
**absolute** fingertip position, so it is the primary input. The UNO Q supplies motion when
the hand is off camera, and later it can provide a physical pen-down signal.

```
webcam ─► MediaPipe HandLandmarker (21 pts) ─► gestures ─► One Euro filter ─┐
                                                                            ├─► OS mouse (OneNote, Paint, any app)
UNO Q (IMU on MCU ─Bridge─► Linux ─UDP─►) ─► fallback (dx,dy,pen) ──────────┘    or built-in canvas
```

## Run

```
run.bat                 # opens in preview mode: nothing is controlled yet
run.bat --mode canvas   # draw in AirPen's own whiteboard
run.bat --mode mouse    # control the real cursor
```

| Gesture | Action |
|---|---|
| Hold thumb + index like a pen; the pen tip is the midpoint between them | Move cursor |
| Pinch thumb + index | Pen down / left button held (writes in OneNote's Draw tab) |
| Pinch thumb + middle | Right click |
| Fist | Clutch: cursor freezes; reposition your hand, then open it to continue |
| Index + middle up, others folded | Scroll (move hand up/down) |

Keys: `1/2/3` preview / mouse / canvas · `r` recenter · `x` clear · `s` save canvas · `q` quit.
**F8 works from any app and toggles mouse control off. It is your panic button.** F10 quits.

Text generation for the demo: write in OneNote with AirPen in mouse mode, then use OneNote's
**Draw → Ink to Text**. That gives handwriting recognition without training a model.

Test the off-camera fallback without hardware: move your hand out of view, then run
`.venv\Scripts\python.exe tools\fake_unoq.py`. The cursor will trace a figure-eight.

Tuning lives at the top of `airpen/main.py` (`BOX`: active region) and `airpen/gestures.py`
(`PINCH_ON/OFF`). Smoothing is set where `OneEuro2D(min_cutoff, beta)` is called.

## UNO Q ideation (not built yet)

The UNO Q has two brains: a Qualcomm Linux MPU and an STM32U585 MCU, joined by the Bridge RPC.
Its USB-C port connects to the Linux side, so the simplest PC link is Wi-Fi UDP using the
protocol already implemented in `airpen/fallback.py`: `dx,dy,pen\n`.

1. **IMU wand / ring (best fit with the write-up).** Plug a Modulino Movement (LSM6DSOX) into
   the Qwiic port, or use an MPU-6050 over I2C. The MCU reads at about 200 Hz and uses the
   **gyroscope as an air mouse** (angular rate maps to dx,dy). This avoids double integration,
   so it doesn't drift the way the write-up feared. A button gives pen-down. The MCU calls
   `Bridge.notify`, Python on the board receives it and sends UDP to the PC.
2. **Sensor fusion (the strong "design project" story).** While the hand is on camera, use
   camera position to estimate the IMU's bias continuously. That is the "online bias
   estimation" from the write-up, with the camera as ground truth. Off camera, the IMU
   coasts on the corrected model.
3. **Physical pen-down.** A force sensor (FSR) or button on the fingertip tells you exactly
   when ink should flow. Pinch detection is the weakest part of camera-only input.
4. **Feedback.** Use the 8×13 LED matrix to show mode and pen state (camera, IMU, or lost),
   with a buzzer tick on click.
5. **Stretch: run the vision on the UNO Q itself.** Plug a USB webcam into the board, run
   MediaPipe on its Linux side, and send cursor coordinates to the PC. That makes it a
   standalone device; expect lower fps.

## Prior art reviewed

- Virtual mouse (MediaPipe + pyautogui/pynput): [hemil13/Virtual-Mouse](https://github.com/hemil13/Virtual-Mouse),
  [Vyomrana02/Virtual_Mouse](https://github.com/Vyomrana02/Virtual_Mouse),
  [Gokul9404/Hand-controlled-VR-Mouse](https://github.com/Gokul9404/Hand-controlled-VR-Mouse),
  [yunwoong7/virtual_mouse](https://github.com/yunwoong7/virtual_mouse).
  Borrowed: active-region mapping, pinch-to-click, scroll gesture.
- Air writing/canvas: [Ankur3509/AirDraw](https://github.com/Ankur3509/AirDraw),
  [machine-canvas-developers/air-canvas](https://github.com/machine-canvas-developers/air-canvas),
  prnvbaraiya/Air-Writing-and-Character-Recognition. Borrowed: pinch = pen down, canvas overlay.
- Smoothing: One Euro Filter (Casiez et al., CHI 2012). Most hobby repos skip it; it is the
  main reason the cursor here is steady enough to write with.
- UNO Q: [Modulino Movement + UNO Q Motion Monitor](https://projecthub.arduino.cc/mario-r/modulino-movement-uno-q-motion-monitor-20cb6f)
  (`Bridge.notify` streaming pattern), [CWTI UNO Q knowledge base](https://github.com/CWTI-Ltd/arduino_uno_q_knowledge_base_and_playground).

The code here is a fresh implementation of those patterns; no third-party repo code is vendored.
"# Design-Project-I" 
