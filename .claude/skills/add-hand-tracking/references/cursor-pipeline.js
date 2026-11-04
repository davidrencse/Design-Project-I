/**
 * Cursor Pipeline — central state manager for hand-tracking cursor.
 *
 * This is the BRAIN. It wires together:
 *   HandTracker (ML detection) → OneEuroFilter (smoothing) → Cursor position
 *   + Fist detection → Click dispatch
 *   + Mouse fallback
 *
 * ADAPT THIS to your framework's state management:
 *   - React: useState/useRef in a context provider, or zustand/jotai store
 *   - Vue: reactive()/ref() in a composable (useHandTracking.ts)
 *   - Svelte: $state runes in a .svelte.ts module
 *   - Vanilla: plain object with getters, or EventTarget for reactivity
 *
 * The logic below is framework-agnostic. Replace the state variables
 * with your framework's reactive primitives.
 *
 * FIST CLICK BEHAVIOR:
 *   - Fires ONCE on the open→closed transition (not continuously while fist held)
 *   - Uses document.elementFromPoint() to find the element under cursor
 *   - Dispatches a real MouseEvent('click') with correct clientX/clientY
 *   - 500ms cooldown between clicks to prevent rapid-fire
 *   - This means fist clicks work with ANY existing click handlers on the page
 */

import { OneEuroFilter } from './one-euro-filter.js';
import { HandTracker } from './hand-tracker.js';

// ============================================================
// STATE — replace these with your framework's reactive primitives
// ============================================================
let gazeX = 0.5;
let gazeY = 0.5;
let isTracking = false;
let isLoading = false;
let useMouseFallback = false;
let hasPermission = false;
let isFist = false;

// ============================================================
// INTERNAL
// ============================================================

// OneEuroFilters — one per axis
// minCutoff=0.4: heavy smoothing at rest (kills jitter)
// beta=0.002: stays smooth during movement (no jerky jumps)
const filterX = new OneEuroFilter(0.4, 0.002);
const filterY = new OneEuroFilter(0.4, 0.002);

let tracker = null;
let mouseCleanup = null;

// Fist click state
let wasFist = false;
const FIST_COOLDOWN_MS = 500;
let lastFistClickTime = 0;

function handleGaze(raw) {
  const now = performance.now() / 1000;
  gazeX = filterX.filter(raw.x, now);
  gazeY = filterY.filter(raw.y, now);
  isFist = raw.isFist;

  // Fire click on fist transition (open → closed)
  const nowMs = performance.now();
  if (raw.isFist && !wasFist && nowMs - lastFistClickTime > FIST_COOLDOWN_MS) {
    lastFistClickTime = nowMs;
    fireFistClick();
  }
  wasFist = raw.isFist;

  // FRAMEWORK HOOK: trigger a re-render or notify subscribers here
  // React: setState({ gazeX, gazeY, isFist })
  // Vue: the refs auto-trigger
  // Svelte: $state auto-triggers
  // Vanilla: dispatchEvent or call registered callbacks
}

function fireFistClick() {
  const screenX = gazeX * window.innerWidth;
  const screenY = gazeY * window.innerHeight;

  const el = document.elementFromPoint(screenX, screenY);
  if (el) {
    el.dispatchEvent(
      new MouseEvent('click', {
        bubbles: true,
        clientX: screenX,
        clientY: screenY,
        view: window,
      })
    );
  }
}

function handleMouseMove(e) {
  gazeX = e.clientX / window.innerWidth;
  gazeY = e.clientY / window.innerHeight;
  // FRAMEWORK HOOK: trigger re-render here
}

// ============================================================
// PUBLIC API
// ============================================================

/**
 * Get current state. In a reactive framework, return reactive refs instead.
 */
export function getGaze() {
  return {
    get x() { return gazeX; },
    get y() { return gazeY; },
    get isTracking() { return isTracking; },
    get isLoading() { return isLoading; },
    get useMouseFallback() { return useMouseFallback; },
    get hasPermission() { return hasPermission; },
    get isFist() { return isFist; },
  };
}

/**
 * Initialize hand tracking. Downloads model (~10MB, cached after first load),
 * requests webcam permission, starts detection loop.
 * Falls back to mouse automatically on any failure.
 */
export async function initGaze() {
  isLoading = true;
  try {
    tracker = new HandTracker();

    // Timeout after 15s — CDN download can be slow on first load
    const initPromise = tracker.init();
    const timeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('MediaPipe init timed out')), 15000)
    );
    await Promise.race([initPromise, timeout]);

    await tracker.startCamera();
    hasPermission = true;
    tracker.start(handleGaze);
    isTracking = true;
  } catch (err) {
    console.warn('Hand tracking unavailable, using mouse fallback:', err);
    enableMouseFallback();
  } finally {
    isLoading = false;
  }
}

/**
 * Switch to mouse control. Can be called at any time.
 * Also used as automatic fallback when hand tracking fails.
 */
export function enableMouseFallback() {
  useMouseFallback = true;
  hasPermission = true;

  if (mouseCleanup) mouseCleanup();
  window.addEventListener('mousemove', handleMouseMove);
  mouseCleanup = () => window.removeEventListener('mousemove', handleMouseMove);
}

/**
 * Clean up everything. Call on unmount/destroy.
 */
export function destroyGaze() {
  if (tracker) tracker.destroy();
  tracker = null;
  if (mouseCleanup) mouseCleanup();
  isTracking = false;
}
