export interface MonitorConfig {
  /** Closure threshold as a fraction of the driver's calibrated open-eye EAR. */
  closureRatio: number;
  /** How long to sample the open-eye baseline, in ms. */
  calibrationMs: number;
  /** Mouth aspect ratio counting as an open mouth. */
  yawnRatio: number;
  /** Mouth must stay open this long to be a yawn rather than speech. */
  yawnHoldMs: number;
  /** Eyes shut at least this long is a microsleep, not a blink. */
  microsleepMs: number;
  /** Closures at or under this are counted as ordinary blinks. */
  blinkMaxMs: number;
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
  /** Minimum time in a state before it may de-escalate, to stop flicker. */
  dwellMs: number;
  /** How long a driver acknowledgement suppresses repeat alarms. */
  snoozeMs: number;
  /** Normalised frame luminance below which tracking is called unreliable. */
  lowLightLuma: number;
  /** Milliseconds of EAR history kept for the trace. */
  traceWindowMs: number;
}

export const DEFAULT_CONFIG: MonitorConfig = {
  // 0.78 rather than a fixed EAR value: published implementations hard-code
  // ~0.25, which is really "0.25 worked for the author's eyes". Scaling the
  // driver's own baseline generalises across eye shape, glasses and seating.
  closureRatio: 0.78,
  calibrationMs: 4000,
  yawnRatio: 0.55,
  // Speech opens the mouth briefly and repeatedly; a yawn holds it open.
  yawnHoldMs: 900,
  // Past 500 ms a closure is a microsleep rather than a blink; escalates
  // while the eyes are still shut, not after they reopen.
  microsleepMs: 500,
  blinkMaxMs: 400,
  perclosWindowMs: 60_000,
  // PERCLOS above 15% is the standard drowsiness criterion.
  drowsyPerclos: 15,
  criticalPerclos: 30,
  yawnsForDrowsy: 2,
  // Checking mirrors and instruments dips the head briefly; a fatigue nod holds.
  nodPitchDegrees: 18,
  nodHoldMs: 800,
  dwellMs: 2500,
  snoozeMs: 60_000,
  lowLightLuma: 0.18,
  traceWindowMs: 20_000,
};
