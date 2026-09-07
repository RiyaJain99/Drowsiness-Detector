import { describe, expect, it } from 'vitest';
import {
  distance,
  eyeAspectRatio,
  faceMetrics,
  headPitchDegrees,
  mouthAspectRatio,
  trimmedMean,
} from '../geometry';
import { CHIN, EYE_OUTER, FOREHEAD, LEFT_EYE, MOUTH_HORIZONTAL, MOUTH_VERTICAL } from '../landmarks';

/** Build a sparse landmark array with only the indices a test needs. */
function mesh(points: Record<number, { x: number; y: number }>) {
  const arr = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
  for (const [i, p] of Object.entries(points)) arr[Number(i)] = p;
  return arr;
}

/** An eye of the given width and lid separation, centred on the origin. */
function eye(width: number, opening: number) {
  const [p1, p2, p3, p4, p5, p6] = LEFT_EYE;
  return mesh({
    [p1]: { x: 0, y: 0 },
    [p4]: { x: width, y: 0 },
    [p2]: { x: width / 3, y: -opening / 2 },
    [p3]: { x: (2 * width) / 3, y: -opening / 2 },
    [p6]: { x: width / 3, y: opening / 2 },
    [p5]: { x: (2 * width) / 3, y: opening / 2 },
  });
}

describe('distance', () => {
  it('measures a 3-4-5 triangle', () => {
    expect(distance({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(5);
  });
});

describe('eyeAspectRatio', () => {
  it('equals opening over width for a symmetric eye', () => {
    // Both vertical spans are `opening`, so EAR = (o + o) / (2w) = o / w.
    expect(eyeAspectRatio(eye(0.1, 0.03), LEFT_EYE)).toBeCloseTo(0.3, 6);
  });

  it('is invariant to distance from the camera', () => {
    // The same eye twice as far away: every coordinate halves, ratio must not.
    const near = eyeAspectRatio(eye(0.1, 0.03), LEFT_EYE);
    const far = eyeAspectRatio(eye(0.05, 0.015), LEFT_EYE);
    expect(far).toBeCloseTo(near, 6);
  });

  it('falls toward zero as the lids close', () => {
    const open = eyeAspectRatio(eye(0.1, 0.03), LEFT_EYE);
    const shut = eyeAspectRatio(eye(0.1, 0.002), LEFT_EYE);
    expect(shut).toBeLessThan(open);
    expect(shut).toBeCloseTo(0.02, 6);
  });

  it('returns 0 rather than dividing by zero on degenerate tracking', () => {
    expect(eyeAspectRatio(mesh({}), LEFT_EYE)).toBe(0);
  });
});

describe('mouthAspectRatio', () => {
  it('rises as the jaw opens', () => {
    const shut = mesh({
      [MOUTH_HORIZONTAL[0]]: { x: 0, y: 0 },
      [MOUTH_HORIZONTAL[1]]: { x: 0.1, y: 0 },
      [MOUTH_VERTICAL[0]]: { x: 0.05, y: 0 },
      [MOUTH_VERTICAL[1]]: { x: 0.05, y: 0.005 },
    });
    const yawning = mesh({
      [MOUTH_HORIZONTAL[0]]: { x: 0, y: 0 },
      [MOUTH_HORIZONTAL[1]]: { x: 0.1, y: 0 },
      [MOUTH_VERTICAL[0]]: { x: 0.05, y: 0 },
      [MOUTH_VERTICAL[1]]: { x: 0.05, y: 0.07 },
    });
    expect(mouthAspectRatio(shut)).toBeCloseTo(0.05, 6);
    expect(mouthAspectRatio(yawning)).toBeCloseTo(0.7, 6);
  });
});

describe('trimmedMean', () => {
  it('ignores outliers at both ends', () => {
    // A blink during calibration would sit at the bottom of this list.
    expect(trimmedMean([0.01, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 9], 0.2)).toBeCloseTo(0.3, 6);
  });

  it('falls back to a plain mean when trimming would empty the set', () => {
    expect(trimmedMean([1, 3], 0.5)).toBe(2);
  });
});

describe('headPitchDegrees', () => {
  const neutral = { foreshortening: 2.0, balance: 1.0 };

  it('reads zero at the calibrated pose', () => {
    expect(headPitchDegrees(neutral, neutral)).toBeCloseTo(0, 6);
  });

  it('recovers the rotation angle from vertical foreshortening', () => {
    // A rigid segment rotated 30 degrees out of plane projects to cos(30) of
    // its length, so the inverse cosine must give 30 back.
    const down = { foreshortening: 2.0 * Math.cos(Math.PI / 6), balance: 1.4 };
    expect(headPitchDegrees(down, neutral)).toBeCloseTo(-30, 4);
  });

  it('signs a forward nod negative and a raised chin positive', () => {
    const shrunk = 2.0 * Math.cos(Math.PI / 9);
    expect(headPitchDegrees({ foreshortening: shrunk, balance: 1.4 }, neutral)).toBeLessThan(0);
    expect(headPitchDegrees({ foreshortening: shrunk, balance: 0.7 }, neutral)).toBeGreaterThan(0);
  });

  it('does not blow up before calibration has produced a baseline', () => {
    expect(headPitchDegrees(neutral, { foreshortening: 0, balance: 0 })).toBe(0);
  });
});

describe('faceMetrics', () => {
  it('is invariant to how close the driver sits', () => {
    const build = (scale: number) =>
      mesh({
        [FOREHEAD]: { x: 0, y: -1 * scale },
        [CHIN]: { x: 0, y: 1 * scale },
        [EYE_OUTER[0]]: { x: -0.5 * scale, y: 0 },
        [EYE_OUTER[1]]: { x: 0.5 * scale, y: 0 },
      });
    expect(faceMetrics(build(1)).foreshortening).toBeCloseTo(faceMetrics(build(2.5)).foreshortening, 6);
  });
});
