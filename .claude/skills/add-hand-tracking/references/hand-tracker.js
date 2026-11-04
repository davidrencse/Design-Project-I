/**
 * MediaPipe Hand Landmarker wrapper.
 *
 * Tracks the index finger tip for cursor position and detects
 * fist gestures for clicking. This is the core ML layer.
 *
 * DEPENDENCY: @mediapipe/tasks-vision (npm install @mediapipe/tasks-vision)
 *
 * ARCHITECTURE:
 *   Webcam (640x480) → MediaPipe HandLandmarker (30fps, GPU)
 *     → 21 hand landmarks per frame
 *     → Extract index finger tip (landmark 8) → cursor x,y
 *     → Detect fist (all fingertips below MCP joints) → click trigger
 *     → Sensitivity remap (0.30–0.70 → 0–1) → small palm = full screen
 *
 * HAND LANDMARK MAP (21 points):
 *   0: Wrist
 *   1-4: Thumb (CMC, MCP, IP, TIP)
 *   5-8: Index finger (MCP, PIP, DIP, TIP) ← TIP (8) = cursor
 *   9-12: Middle finger
 *   13-16: Ring finger
 *   17-20: Pinky
 *
 * FIST DETECTION:
 *   A fist is detected when ALL four fingertips (8, 12, 16, 20) are
 *   below (higher y-value than) their corresponding MCP joints (5, 9, 13, 17).
 *   The thumb is excluded because it curls sideways, not downward.
 *
 * SENSITIVITY:
 *   Raw MediaPipe coords are 0–1 across the full camera frame. In practice,
 *   your palm only moves within ~40% of that range (0.30–0.70). The remap()
 *   function expands this to fill 0–1, so small palm movements cover the
 *   entire screen. Adjust RAW_MIN/RAW_MAX if users need more/less range.
 *
 * ADAPT THIS FILE to your framework's module system (ES modules shown below).
 * The import path for @mediapipe/tasks-vision may need adjustment based on
 * your bundler.
 */

import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// --- Landmark indices ---
const INDEX_FINGER_TIP = 8;
const INDEX_FINGER_MCP = 5;
const MIDDLE_FINGER_TIP = 12;
const MIDDLE_FINGER_MCP = 9;
const RING_FINGER_TIP = 16;
const RING_FINGER_MCP = 13;
const PINKY_TIP = 20;
const PINKY_MCP = 17;

// --- Sensitivity range ---
// Tighter range = more sensitive (less hand movement needed)
// 0.30–0.70 means the middle 40% of camera frame maps to full screen
const RAW_MIN = 0.30;
const RAW_MAX = 0.70;
const RAW_RANGE = RAW_MAX - RAW_MIN;

function remap(value) {
  return Math.max(0, Math.min(1, (value - RAW_MIN) / RAW_RANGE));
}

/**
 * Detect fist: all four fingertips curled below their MCP joints.
 * In MediaPipe's coordinate system, y increases downward,
 * so tip.y > mcp.y means the finger is curled.
 */
function detectFist(hand) {
  return (
    hand[INDEX_FINGER_TIP].y > hand[INDEX_FINGER_MCP].y &&
    hand[MIDDLE_FINGER_TIP].y > hand[MIDDLE_FINGER_MCP].y &&
    hand[RING_FINGER_TIP].y > hand[RING_FINGER_MCP].y &&
    hand[PINKY_TIP].y > hand[PINKY_MCP].y
  );
}

export class HandTracker {
  constructor() {
    this.landmarker = null;
    this.videoEl = null;
    this.stream = null;
    this.animationId = 0;
    this.lastDetectTime = 0;
    this.detectInterval = 1000 / 30; // 30fps detection
    this.onGaze = null;
    this._isRunning = false;
  }

  get isRunning() {
    return this._isRunning;
  }

  /**
   * Initialize MediaPipe Hand Landmarker.
   * Downloads WASM runtime (~2MB) and model (~10MB) from CDN on first call.
   * Subsequent calls use browser cache.
   */
  async init() {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    this.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 1
    });
  }

  /**
   * Request webcam access and create a hidden video element.
   * The video element must be in the DOM for MediaPipe to read frames.
   */
  async startCamera() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });

    this.videoEl = document.createElement('video');
    this.videoEl.srcObject = this.stream;
    this.videoEl.setAttribute('playsinline', '');
    this.videoEl.style.display = 'none';
    document.body.appendChild(this.videoEl);
    await this.videoEl.play();

    return this.videoEl;
  }

  /**
   * Start the detection loop.
   * @param {function} callback — Called each frame with:
   *   { x: number, y: number, confidence: number, isFist: boolean }
   *   x,y are 0–1, already remapped for sensitivity and mirror-flipped.
   */
  start(callback) {
    this.onGaze = callback;
    this._isRunning = true;
    this._tick();
  }

  _tick = () => {
    if (!this._isRunning) return;
    this.animationId = requestAnimationFrame(this._tick);

    const now = performance.now();
    if (now - this.lastDetectTime < this.detectInterval) return;
    this.lastDetectTime = now;

    if (!this.landmarker || !this.videoEl || this.videoEl.readyState < 2) return;

    const result = this.landmarker.detectForVideo(this.videoEl, now);

    if (result.landmarks && result.landmarks.length > 0) {
      const hand = result.landmarks[0];
      const tip = hand[INDEX_FINGER_TIP];

      const x = remap(1 - tip.x); // Mirror flip: hand right → cursor right
      const y = remap(tip.y);
      const isFist = detectFist(hand);

      this.onGaze({ x, y, confidence: 1, isFist });
    }
  };

  stop() {
    this._isRunning = false;
    if (this.animationId) cancelAnimationFrame(this.animationId);
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.videoEl) this.videoEl.remove();
  }

  destroy() {
    this.stop();
    if (this.landmarker) this.landmarker.close();
    this.landmarker = null;
  }
}
