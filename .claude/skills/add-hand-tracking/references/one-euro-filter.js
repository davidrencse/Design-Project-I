/**
 * 1€ (One Euro) Filter — adaptive low-pass filter for noisy hand tracking signals.
 * Reduces jitter when hand is still, stays responsive when moving fast.
 *
 * Reference: https://cristal.univ-lille.fr/~casiez/1euro/
 *
 * USAGE:
 *   const filterX = new OneEuroFilter(0.4, 0.002);
 *   const filterY = new OneEuroFilter(0.4, 0.002);
 *
 *   // Call on every tracking frame:
 *   const smoothX = filterX.filter(rawX, performance.now() / 1000);
 *   const smoothY = filterY.filter(rawY, performance.now() / 1000);
 *
 * TUNING GUIDE:
 *   minCutoff (default 0.4):
 *     Lower = more smoothing when still. Kills jitter but adds slight lag.
 *     Range: 0.1 (ultra smooth) to 3.0 (almost raw)
 *     0.4 is the sweet spot for hand tracking.
 *
 *   beta (default 0.002):
 *     Higher = more responsive to fast movements (less lag when moving).
 *     Range: 0.0001 (very smooth) to 0.1 (very reactive)
 *     0.002 prevents jerky jumps while still allowing intentional movement.
 *
 *   dCutoff (default 1.0):
 *     Derivative filter cutoff. Almost never needs changing.
 */

function smoothingFactor(te, cutoff) {
  const r = 2 * Math.PI * cutoff * te;
  return r / (r + 1);
}

class LowPassFilter {
  constructor(initial = 0) {
    this.y = initial;
    this.s = initial;
    this.initialized = false;
  }

  filter(value, alpha) {
    if (!this.initialized) {
      this.y = value;
      this.s = value;
      this.initialized = true;
      return value;
    }
    this.y = value;
    this.s = alpha * value + (1 - alpha) * this.s;
    return this.s;
  }

  last() {
    return this.s;
  }

  reset(value) {
    this.y = value;
    this.s = value;
    this.initialized = false;
  }
}

export class OneEuroFilter {
  /**
   * @param {number} minCutoff — Lower = more smoothing at rest. Default 0.4
   * @param {number} beta — Higher = more responsive when moving. Default 0.002
   * @param {number} dCutoff — Derivative cutoff. Default 1.0
   */
  constructor(minCutoff = 0.4, beta = 0.002, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
    this.lastTime = -1;
  }

  filter(value, timestamp) {
    const now = timestamp ?? performance.now() / 1000;

    if (this.lastTime < 0) {
      this.lastTime = now;
      return this.xFilter.filter(value, 1);
    }

    const te = now - this.lastTime;
    if (te <= 0) return this.xFilter.last();
    this.lastTime = now;

    const dx = (value - this.xFilter.last()) / te;
    const edx = this.dxFilter.filter(dx, smoothingFactor(te, this.dCutoff));

    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const alpha = smoothingFactor(te, cutoff);

    return this.xFilter.filter(value, alpha);
  }

  reset(value = 0.5) {
    this.xFilter.reset(value);
    this.dxFilter.reset(0);
    this.lastTime = -1;
  }
}
