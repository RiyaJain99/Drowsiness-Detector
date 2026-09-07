import {
  CHIN,
  EYE_OUTER,
  FOREHEAD,
  LEFT_EYE,
  MOUTH_HORIZONTAL,
  MOUTH_VERTICAL,
  RIGHT_EYE,
} from './landmarks';

export interface Point {
  x: number;
  y: number;
}

export const distance = (a: Point, b: Point): number => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * Eye aspect ratio: mean vertical lid separation over horizontal eye width.
 *
 *     EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
 *
 * Dividing by eye width is the whole point — it makes the value independent of
 * how far the driver is from the camera and of absolute face size, so one
 * threshold holds as they shift in their seat.
 */
export function eyeAspectRatio(landmarks: Point[], indices: readonly number[]): number {
  const [p1, p2, p3, p4, p5, p6] = indices.map((i) => landmarks[i]);
  const width = distance(p1, p4);
  if (width === 0) return 0;
  return (distance(p2, p6) + distance(p3, p5)) / (2 * width);
}

/** Mean EAR across both eyes. One eye occluded by head turn skews this, so both are read. */
export function meanEyeAspectRatio(landmarks: Point[]): number {
  return (eyeAspectRatio(landmarks, LEFT_EYE) + eyeAspectRatio(landmarks, RIGHT_EYE)) / 2;
}

/** Mouth aspect ratio — vertical lip gap over mouth width. Drives yawn detection. */
export function mouthAspectRatio(landmarks: Point[]): number {
  const width = distance(landmarks[MOUTH_HORIZONTAL[0]], landmarks[MOUTH_HORIZONTAL[1]]);
  if (width === 0) return 0;
  return distance(landmarks[MOUTH_VERTICAL[0]], landmarks[MOUTH_VERTICAL[1]]) / width;
}

/**
 * Trimmed mean — discards the lowest and highest `trim` fraction before averaging.
 * Calibration uses this so a blink or a tracking glitch during the baseline
 * capture cannot drag the threshold down.
 */
export function trimmedMean(values: number[], trim = 0.2): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const cut = Math.floor(sorted.length * trim);
  const kept = sorted.length - cut * 2 > 0 ? sorted.slice(cut, sorted.length - cut) : sorted;
  return kept.reduce((sum, v) => sum + v, 0) / kept.length;
}

/** Scale-free description of head geometry in the image plane. */
export interface FaceMetrics {
  /**
   * Face height over interocular width. Pitch rotation foreshortens the
   * vertical axis while leaving the horizontal one alone, so this shrinks as
   * the head turns away from square-on — and it is immune to the driver simply
   * sitting closer to the camera.
   */
  foreshortening: number;
  /**
   * Upper face height over lower face height. Tilting down expands the visible
   * forehead and compresses the jaw, so this rises. It supplies the sign that
   * foreshortening alone cannot.
   */
  balance: number;
}

export function faceMetrics(landmarks: Point[]): FaceMetrics {
  const eyeMid = {
    x: (landmarks[EYE_OUTER[0]].x + landmarks[EYE_OUTER[1]].x) / 2,
    y: (landmarks[EYE_OUTER[0]].y + landmarks[EYE_OUTER[1]].y) / 2,
  };
  const interocular = distance(landmarks[EYE_OUTER[0]], landmarks[EYE_OUTER[1]]);
  const upper = distance(landmarks[FOREHEAD], eyeMid);
  const lower = distance(eyeMid, landmarks[CHIN]);

  return {
    foreshortening: interocular === 0 ? 0 : (upper + lower) / interocular,
    balance: lower === 0 ? 0 : upper / lower,
  };
}

/**
 * Signed head pitch in degrees, relative to the driver's calibrated neutral pose.
 *
 * A rigid segment rotated by θ out of the image plane projects to cos(θ) of its
 * length, so inverting the foreshortening ratio recovers the angle directly.
 * Negative is nodding forward, which is the direction that matters here.
 */
export function headPitchDegrees(current: FaceMetrics, baseline: FaceMetrics): number {
  if (baseline.foreshortening === 0) return 0;
  const ratio = Math.min(1, Math.max(-1, current.foreshortening / baseline.foreshortening));
  const magnitude = (Math.acos(ratio) * 180) / Math.PI;
  const noddingForward = current.balance > baseline.balance;
  return noddingForward ? -magnitude : magnitude;
}
