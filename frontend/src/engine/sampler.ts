/**
 * Core mathematical utilities for browser-side DIAMOND diffusion sampler
 */

export const ACTION_ORDER = ["LEFT", "RIGHT", "UP", "DOWN", "JUMP", "ATTACK", "HEAL"] as const;

export const TRACKED_KEYS: Record<string, string> = {
  a: "LEFT",
  arrowleft: "LEFT",
  d: "RIGHT",
  arrowright: "RIGHT",
  w: "UP",
  arrowup: "UP",
  s: "DOWN",
  arrowdown: "DOWN",
  k: "ATTACK",
  j: "HEAL",
  " ": "JUMP",
  space: "JUMP",
};

/**
 * Encodes active keyboard inputs into a single integer bitmask.
 * Matches PyTorch ACTION_ORDER:
 * LEFT: 1, RIGHT: 2, UP: 4, DOWN: 8, JUMP: 16, ATTACK: 32, HEAL: 64
 */
export function encodeAction(pressedKeys: Set<string>): number {
  let encoded = 0;
  let mult = 1;

  for (const name of ACTION_ORDER) {
    let isActive = false;
    for (const [key, actionName] of Object.entries(TRACKED_KEYS)) {
      if (actionName === name && pressedKeys.has(key)) {
        isActive = true;
        break;
      }
    }
    if (isActive) {
      encoded += mult;
    }
    mult *= 2;
  }

  return encoded;
}

/**
 * Builds the Karras sigma schedule for EDM sampling.
 * Matches world_model.model.diffusion_sampler.build_sigmas:
 * s_i = (max_inv + l * (min_inv - max_inv)) ** rho
 */
export function buildSigmas(
  numSteps = 3,
  sigmaMin = 0.002,
  sigmaMax = 5.0,
  rho = 7
): Float32Array {
  const minInvRho = Math.pow(sigmaMin, 1 / rho);
  const maxInvRho = Math.pow(sigmaMax, 1 / rho);
  const sigmas = new Float32Array(numSteps + 1);

  for (let i = 0; i < numSteps; i++) {
    const l = numSteps <= 1 ? 0 : i / (numSteps - 1);
    const s = maxInvRho + l * (minInvRho - maxInvRho);
    sigmas[i] = Math.pow(s, rho);
  }
  sigmas[numSteps] = 0.0;
  return sigmas;
}

/**
 * Generates Gaussian noise using the Box-Muller transform:
 * Z ~ N(mean, std^2)
 */
export function boxMullerRandom(out: Float32Array, mean = 0.0, std = 1.0): void {
  const len = out.length;
  for (let i = 0; i < len; i += 2) {
    let u1 = Math.random();
    let u2 = Math.random();
    while (u1 <= 1e-15) u1 = Math.random(); // avoid log(0)

    const r = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 2.0 * Math.PI * u2;

    out[i] = mean + std * (r * Math.cos(theta));
    if (i + 1 < len) {
      out[i + 1] = mean + std * (r * Math.sin(theta));
    }
  }
}

/**
 * Converts a planar CHW float32 tensor [0, 1] into interleaved HWC RGBA Uint8ClampedArray [0, 255].
 * Fast single loop with clamp and rounding.
 */
export function chwToRgba(
  chw: Float32Array,
  rgba: Uint8ClampedArray,
  h: number,
  w: number
): void {
  const planeSize = h * w;
  const rOffset = 0;
  const gOffset = planeSize;
  const bOffset = planeSize * 2;

  let outIdx = 0;
  for (let i = 0; i < planeSize; i++) {
    const r = chw[rOffset + i];
    const g = chw[gOffset + i];
    const b = chw[bOffset + i];

    rgba[outIdx]     = r <= 0 ? 0 : r >= 1 ? 255 : (r * 255 + 0.5) | 0;
    rgba[outIdx + 1] = g <= 0 ? 0 : g >= 1 ? 255 : (g * 255 + 0.5) | 0;
    rgba[outIdx + 2] = b <= 0 ? 0 : b >= 1 ? 255 : (b * 255 + 0.5) | 0;
    rgba[outIdx + 3] = 255;
    outIdx += 4;
  }
}
