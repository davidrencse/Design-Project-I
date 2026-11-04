/**
 * Edge-Zone Scroller — scrolls the page based on vertical cursor position.
 *
 * When the hand (or mouse) cursor enters the top or bottom zone of the
 * screen, the page scrolls in that direction. Speed ramps up quadratically
 * the further into the zone the cursor goes.
 *
 * ZONES:
 *   ┌─────────────────────┐
 *   │   TOP ZONE (0-35%)  │  ← scroll up, faster near top
 *   │                     │
 *   │  DEAD ZONE (35-65%) │  ← no scrolling
 *   │                     ��
 *   │ BOTTOM ZONE (65-100%)│  ← scroll down, faster near bottom
 *   └─────────────────────┘
 *
 * USAGE:
 *   // On mount, set the scroll container:
 *   setScrollContainer(document.querySelector('.my-scroll-wrapper'));
 *   // or for full-page scroll:
 *   setScrollContainer(document.documentElement);
 *
 *   // On every animation frame, call:
 *   updateScroll(gazeY);  // gazeY is 0-1, from the cursor pipeline
 *
 * TUNING:
 *   DEAD_ZONE_TOP/BOTTOM: Adjust the trigger zones. Wider dead zone = less
 *     accidental scrolling but you have to move further to trigger.
 *   MAX_SCROLL_SPEED: Pixels per frame at maximum intensity. 45 feels fast
 *     but controllable. Increase for long pages, decrease for short ones.
 */

let scrollContainer = null;
let isScrolling = false;
let scrollDirection = null; // 'up' | 'down' | null

const DEAD_ZONE_TOP = 0.35;
const DEAD_ZONE_BOTTOM = 0.65;
const MAX_SCROLL_SPEED = 45; // px per frame

/**
 * Get current scroll state. Adapt to your framework's reactivity.
 */
export function getScroll() {
  return {
    get isScrolling() { return isScrolling; },
    get direction() { return scrollDirection; },
  };
}

/**
 * Set the scrollable container element.
 * For full-page scrolling: document.documentElement or document.body
 * For a scroll wrapper: the wrapper element itself
 */
export function setScrollContainer(el) {
  scrollContainer = el;
}

/**
 * Call this on every animation frame with the current gazeY (0-1).
 * It calculates scroll speed and direction, then scrolls the container.
 */
export function updateScroll(gazeY) {
  if (!scrollContainer) return;

  let speed = 0;

  if (gazeY > DEAD_ZONE_BOTTOM) {
    // Bottom zone — scroll down
    const intensity = (gazeY - DEAD_ZONE_BOTTOM) / (1 - DEAD_ZONE_BOTTOM);
    speed = intensity * intensity * MAX_SCROLL_SPEED;
    scrollDirection = 'down';
  } else if (gazeY < DEAD_ZONE_TOP) {
    // Top zone — scroll up
    const intensity = (DEAD_ZONE_TOP - gazeY) / DEAD_ZONE_TOP;
    speed = -intensity * intensity * MAX_SCROLL_SPEED;
    scrollDirection = 'up';
  } else {
    // Dead zone — no scrolling
    scrollDirection = null;
  }

  isScrolling = Math.abs(speed) > 0.5;

  if (isScrolling) {
    scrollContainer.scrollBy(0, speed);
  }
}
