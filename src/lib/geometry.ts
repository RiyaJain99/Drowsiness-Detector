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
  /** MediaPipe depth, roughly in units of image width; smaller is closer to the camera. */
  z?: number;
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

/** Absolute head orientation in degrees, from 3-D landmarks. */
export interface HeadPose {
  /** Negative is nodding forward (chin to chest). */
  pitch: number;
  /** Positive is turning toward the driver's left. */
  yaw: number;
  /** Positive is tilting so the driver's left eye drops in the image. */
  roll: number;
}

const DEG = 180 / Math.PI;

/**
 * Full three-axis head pose from MediaPipe's 3-D landmarks.
 *
 * MediaPipe gives x and y normalised to image width and height and z in
 * (roughly) image-width units. Rescaling y by the aspect ratio puts all three
 * axes in the same metric, after which two rigid facial segments are enough:
 *
 *  - the interocular axis (outer eye corners) rotates with yaw and roll;
 *  - the vertical axis (forehead → chin) rotates with pitch.
 *
 * Each angle is the out-of-plane (depth) component against the in-plane
 * length, so it is independent of distance from the camera, and pitch is
 * measured against the in-plane length rather than the y component alone so
 * that tilting the head sideways (roll) does not leak into it. Returns null
 * when the landmarks carry no depth.
 */
export function headPose(landmarks: Point[], aspect = 4 / 3): HeadPose | null {
  const a = landmarks[EYE_OUTER[0]];
  const b = landmarks[EYE_OUTER[1]];
  const top = landmarks[FOREHEAD];
  const bottom = landmarks[CHIN];
  if (a?.z === undefined || b?.z === undefined || top?.z === undefined || bottom?.z === undefined) return null;

  const scaleY = 1 / (aspect || 1);
  const rx = b.x - a.x;
  const ry = (b.y - a.y) * scaleY;
  const rz = b.z - a.z;
  const ux = top.x - bottom.x;
  const uy = (top.y - bottom.y) * scaleY;
  const uz = top.z - bottom.z;

  // Nodding forward brings the forehead toward the camera (z falls) relative
  // to the chin — that is negative pitch in this project's convention.
  const pitch = Math.atan2(uz, Math.hypot(ux, uy)) * DEG;
  const yaw = Math.atan2(rz, Math.hypot(rx, ry)) * DEG;
  const roll = Math.atan2(ry, Math.abs(rx)) * DEG;
  return { pitch, yaw, roll };
}

/** Axis-aligned bounds of a set of landmarks, normalised 0–1. */
export function bounds(landmarks: Point[], indices: readonly number[]) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const i of indices) {
    const p = landmarks[i];
    if (!p) continue;
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY, width: Math.max(0, maxX - minX), height: Math.max(0, maxY - minY) };
}

/** Median of a small array — used as a glitch filter on per-frame EAR. */
export function median(values: readonly number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((x, y) => x - y);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Population standard deviation. */
export function standardDeviation(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
}
