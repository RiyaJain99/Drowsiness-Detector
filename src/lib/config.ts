export interface MonitorConfig {
  /** Closure threshold as a fraction of the driver's calibrated open-eye EAR. */
  closureRatio: number;
  /** How long to sample the open-eye baseline, in ms of *face-visible* time. */
  calibrationMs: number;
  /** Calibration gives up if it cannot collect a baseline within this long. */
  calibrationTimeoutMs: number;
  /** Fewest frames a baseline may be built from. */
  calibrationMinFrames: number;
  /** A baseline EAR below this almost certainly means eyes closed or bad tracking. */
  calibrationMinEar: number;
  /** Max coefficient of variation of baseline EAR — above it the driver was not holding still. */
  calibrationMaxVariation: number;
  /** Max absolute head yaw during calibration — the driver must face the camera. */
  calibrationMaxYaw: number;
  /** Max absolute head pitch during calibration (dash cameras sit below eye level, so generous). */
  calibrationMaxPitch: number;
  /** Frames in the EAR median filter. Removes single-frame landmark glitches. */
  earSmoothingFrames: number;
  /** Mouth aspect ratio counting as an open mouth (floor; calibration may raise it). */
  yawnRatio: number;
  /** Personal yawn threshold sits this far above the calibrated resting MAR. */
  yawnMarMargin: number;
  /** Mouth must stay open this long to be a yawn rather than speech. */
  yawnHoldMs: number;
  /** Eyes shut at least this long is a microsleep, not a blink. */
  microsleepMs: number;
  /** Closures at or under this are counted as ordinary blinks. */
  blinkMaxMs: number;
  /** Blinks per minute above which blinking is called excessive. */
  highBlinkRate: number;
  /** Blinks per minute below which blinking is called reduced (staring / fatigue). */
  lowBlinkRate: number;
  /** Rolling window PERCLOS is computed over. */
  perclosWindowMs: number;
  /** PERCLOS percentage at which the driver is considered drowsy. */
  drowsyPerclos: number;
  /** PERCLOS percentage that escalates to critical. */
  criticalPerclos: number;
  /** Yawns within the session before drowsiness is called on yawning alone. */
  yawnsForDrowsy: number;
  /** Degrees of forward head pitch below neutral that counts as a nod. */
  nodPitchDegrees: number;
  /** How long the head must stay down before the nod is called. */
  nodHoldMs: number;
  /** Degrees of head yaw away from neutral that counts as looking away. */
  distractionYawDegrees: number;
  /** How long the head must stay turned before it is called a distraction. */
  distractionHoldMs: number;
  /** Tracking may drop out this long before it is treated as "no face". */
  faceLostGraceMs: number;
  /** Drowsiness score (0–100) that alone escalates to drowsy. */
  scoreDrowsy: number;
  /** Drowsiness score (0–100) that alone escalates to critical. */
  scoreCritical: number;
  /** Window over which microsleeps / yawns / nods feed the drowsiness score. */
  scoreWindowMs: number;
  /** Minimum time in a state before it may de-escalate, to stop flicker. */
  dwellMs: number;
  /** How long a driver acknowledgement suppresses repeat alarms. */
  snoozeMs: number;
  /** Normalised frame luminance below which tracking is called unreliable. */
  lowLightLuma: number;
  /** Milliseconds of EAR history kept for the trace. */
  traceWindowMs: number;
  /** Minimum gap between repeated spoken warnings of the same level. */
  voiceCooldownMs: number;
  /** Consecutive drowsy verdicts from the trained model before it raises a warning. */
  modelConsecutive: number;
  /** How often the trained model is evaluated. */
  modelIntervalMs: number;
}

export const DEFAULT_CONFIG: MonitorConfig = {
  // 0.78 rather than a fixed EAR value: published implementations hard-code
  // ~0.25, which is really "0.25 worked for the author's eyes". Scaling the
  // driver's own baseline generalises across eye shape, glasses and seating.
  closureRatio: 0.78,
  calibrationMs: 4000,
  calibrationTimeoutMs: 15_000,
  calibrationMinFrames: 30,
  calibrationMinEar: 0.12,
  // Trimmed EAR of a still, open-eyed face varies by a few percent; 25%
  // means the eyes were opening and closing or the head was moving.
  calibrationMaxVariation: 0.25,
  calibrationMaxYaw: 25,
  calibrationMaxPitch: 35,
  // A 3-frame median rejects one-frame spikes but delays a real closure by
  // only one frame (~33 ms), well under the 500 ms microsleep criterion.
  earSmoothingFrames: 3,
  yawnRatio: 0.55,
  yawnMarMargin: 0.25,
  // Speech opens the mouth briefly and repeatedly; a yawn holds it open.
  yawnHoldMs: 900,
  // Past 500 ms a closure is a microsleep rather than a blink; escalates
  // while the eyes are still shut, not after they reopen.
  microsleepMs: 500,
  blinkMaxMs: 400,
  // Resting adults blink ~12–20/min; fatigue shows as either extreme.
  highBlinkRate: 32,
  lowBlinkRate: 6,
  perclosWindowMs: 60_000,
  // PERCLOS above 15% is the standard drowsiness criterion.
  drowsyPerclos: 15,
  criticalPerclos: 30,
  yawnsForDrowsy: 2,
  // Checking mirrors and instruments dips the head briefly; a fatigue nod holds.
  nodPitchDegrees: 18,
  nodHoldMs: 800,
  // A mirror check turns the head ~30–45° for well under two seconds.
  distractionYawDegrees: 32,
  distractionHoldMs: 2200,
  // MediaPipe drops the odd frame under motion blur; that is not "no driver".
  faceLostGraceMs: 1500,
  scoreDrowsy: 45,
  scoreCritical: 80,
  scoreWindowMs: 300_000,
  dwellMs: 2500,
  snoozeMs: 60_000,
  lowLightLuma: 0.18,
  traceWindowMs: 20_000,
  voiceCooldownMs: 20_000,
  // One window can be wrong; five in a row (about 5 s) is a sustained pattern.
  modelConsecutive: 5,
  modelIntervalMs: 1000,
};

/** User-tunable subset exposed in the settings drawer, with sane bounds. */
export interface TunableSpec {
  key: keyof MonitorConfig;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  /** Display transform (e.g. ms → s). */
  scale?: number;
}

export const TUNABLES: TunableSpec[] = [
  { key: 'closureRatio', label: 'Eye-closure sensitivity', hint: 'Closed when EAR < baseline × this. Higher = more sensitive.', min: 0.6, max: 0.9, step: 0.01, unit: '×' },
  { key: 'microsleepMs', label: 'Microsleep duration', hint: 'Eye closure longer than this triggers a critical alert.', min: 300, max: 1500, step: 50, unit: 's', scale: 0.001 },
  { key: 'drowsyPerclos', label: 'PERCLOS warning level', hint: 'Percent of time eyes closed (last 60 s) that raises a warning.', min: 8, max: 30, step: 1, unit: '%' },
  { key: 'criticalPerclos', label: 'PERCLOS critical level', hint: 'Percent of time eyes closed that escalates to critical.', min: 15, max: 50, step: 1, unit: '%' },
  { key: 'yawnHoldMs', label: 'Yawn hold time', hint: 'Mouth must stay wide open this long to count as a yawn.', min: 500, max: 2500, step: 100, unit: 's', scale: 0.001 },
  { key: 'nodPitchDegrees', label: 'Head-nod angle', hint: 'Forward head drop (vs. calibrated pose) that counts as a nod.', min: 10, max: 35, step: 1, unit: '°' },
  { key: 'distractionYawDegrees', label: 'Looking-away angle', hint: 'Head turn away from the road that counts as distraction.', min: 20, max: 60, step: 1, unit: '°' },
  { key: 'calibrationMs', label: 'Calibration length', hint: 'Seconds of open-eye baseline captured at start.', min: 2000, max: 10_000, step: 500, unit: 's', scale: 0.001 },
  { key: 'voiceCooldownMs', label: 'Voice alert cooldown', hint: 'Minimum gap between repeated spoken warnings.', min: 5000, max: 120_000, step: 5000, unit: 's', scale: 0.001 },
];

/** Clamp untrusted (e.g. localStorage) overrides into the allowed ranges. */
export function sanitizeConfig(overrides: Partial<Record<string, unknown>>): Partial<MonitorConfig> {
  const out: Partial<MonitorConfig> = {};
  for (const spec of TUNABLES) {
    const raw = overrides[spec.key];
    if (typeof raw !== 'number' || !Number.isFinite(raw)) continue;
    out[spec.key] = Math.min(spec.max, Math.max(spec.min, raw));
  }
  return out;
}
