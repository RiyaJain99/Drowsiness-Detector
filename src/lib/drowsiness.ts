import { DEFAULT_CONFIG, type MonitorConfig } from './config';
import {
  headPitchDegrees,
  median,
  standardDeviation,
  trimmedMean,
  type FaceMetrics,
  type HeadPose,
} from './geometry';

export type DriverState = 'idle' | 'calibrating' | 'alert' | 'drowsy' | 'critical';
export type Severity = 'ok' | 'warn' | 'crit';
export type EventKind =
  | 'system'
  | 'calibration'
  | 'state'
  | 'microsleep'
  | 'yawn'
  | 'nod'
  | 'blinking'
  | 'distraction'
  | 'face';
export type Attention = 'inactive' | 'calibrating' | 'attentive' | 'distracted' | 'fatigued' | 'critical' | 'no-face';
export type CalibrationPhase = 'idle' | 'running' | 'success' | 'failed';

export interface MonitorEvent {
  /** ms since the session origin. */
  at: number;
  label: string;
  detail: string;
  severity: Severity;
  kind: EventKind;
  /**
   * 0–1 detection confidence, derived from how far the triggering signal
   * cleared its threshold and from the current tracking quality. Null for
   * bookkeeping events that are not detections.
   */
  confidence: number | null;
  /** Driver state at the moment the event was recorded. */
  status: DriverState;
}

export interface Sample {
  t: number;
  ear: number;
  closed: boolean;
}

/** One frame's worth of measurements, already reduced to scale-free ratios. */
export interface Frame {
  ear: number;
  mar: number;
  face: FaceMetrics;
  earLeft?: number;
  earRight?: number;
  /** Absolute 3-D head pose, when the landmarks carry depth. */
  pose?: HeadPose | null;
  /** 0–1 tracking quality (light level, face size). Scales event confidence. */
  quality?: number;
}

/** Points each signal contributes to the 0–100 drowsiness score. */
export interface ScoreParts {
  perclos: number;
  closure: number;
  microsleeps: number;
  yawns: number;
  nods: number;
  blinkRate: number;
}

export interface CalibrationStatus {
  phase: CalibrationPhase;
  /** 0–1 */
  progress: number;
  /** Outcome text once finished, or live guidance while running. */
  message: string;
  /** 0–1 baseline stability once calibrated. */
  quality: number | null;
}

export interface Reading {
  state: DriverState;
  reason: string;
  ear: number;
  earLeft: number;
  earRight: number;
  mar: number;
  closed: boolean;
  /** How long the eyes have currently been shut, ms (0 when open). */
  closedForMs: number;
  perclos: number;
  blinksPerMinute: number;
  totalBlinks: number;
  yawns: number;
  microsleeps: number;
  nods: number;
  distractions: number;
  /** Times the driver entered the drowsy (warning) state. */
  warnings: number;
  /** Times the driver entered the critical state. */
  criticals: number;
  /** Signed degrees from the calibrated neutral pose; negative is nodding forward. */
  pitch: number;
  /** Degrees from neutral; positive is toward the driver's left. */
  yaw: number;
  roll: number;
  /** 0–100 composite drowsiness score. */
  score: number;
  scoreParts: ScoreParts;
  attention: Attention;
  faceVisible: boolean;
  quality: number;
  threshold: number | null;
  baseline: number | null;
  yawnThreshold: number | null;
  calibration: CalibrationStatus;
  acknowledged: boolean;
  /** True on the frame a state transition happens, so the caller can alarm once. */
  transitioned: boolean;
}

const ZERO_PARTS: ScoreParts = { perclos: 0, closure: 0, microsleeps: 0, yawns: 0, nods: 0, blinkRate: 0 };
const EMPTY_FRAME: Frame = { ear: 0, mar: 0, face: { foreshortening: 0, balance: 0 } };
/** Above this frame interval the median filter would eat whole blinks, so it is bypassed. */
const SMOOTHING_MAX_DT = 50;

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const round2 = (v: number) => Math.round(v * 100) / 100;

/**
 * Consumes per-frame facial ratios and decides how awake the driver is.
 *
 * Deliberately knows nothing about cameras, canvases or React: every method
 * takes the current timestamp as an argument rather than reading a clock. That
 * is what makes the whole decision path unit-testable without a browser.
 */
export class DrowsinessAnalyzer {
  private cfg: MonitorConfig;

  private state: DriverState = 'idle';
  private stateSince = 0;
  private reason = '';

  // ── calibration
  private baseline: number | null = null;
  private threshold: number | null = null;
  private yawnThreshold: number | null = null;
  private neutralPose: FaceMetrics | null = null;
  private neutralHeadPose: HeadPose | null = null;
  private calibrationEar: number[] = [];
  private calibrationMar: number[] = [];
  private calibrationPose: FaceMetrics[] = [];
  private calibrationHeadPose: HeadPose[] = [];
  private calibrationStart = 0;
  private calibrationValidMs = 0;
  private calibrationLastT: number | null = null;
  private calibrationPhase: CalibrationPhase = 'idle';
  private calibrationMessage = '';
  private calibrationQuality: number | null = null;

  // ── signal tracking
  private samples: Sample[] = [];
  private earWindow: number[] = [];
  private lastT: number | null = null;
  private blinkTimes: number[] = [];
  private closedSince: number | null = null;
  private closureEarSum = 0;
  private closureFrames = 0;
  private mouthOpenSince: number | null = null;
  private mouthPeak = 0;
  private yawnCounted = false;
  private headDownSince: number | null = null;
  private nodCounted = false;
  private turnedSince: number | null = null;
  private distractionCounted = false;
  private faceLostSince: number | null = null;
  private faceLostHandled = false;
  private monitoringSince: number | null = null;
  private lastBlinkWarning = -Infinity;

  // ── counters
  private yawns = 0;
  private microsleeps = 0;
  private nods = 0;
  private distractions = 0;
  private totalBlinks = 0;
  private warnings = 0;
  private criticals = 0;
  private yawnTimes: number[] = [];
  private microsleepTimes: number[] = [];
  private nodTimes: number[] = [];

  private pitch = 0;
  private yaw = 0;
  private roll = 0;
  private score = 0;
  private scoreParts: ScoreParts = ZERO_PARTS;
  private quality = 1;
  private perclosNow = 0;
  private lastFrame: Frame = EMPTY_FRAME;
  private lastNow = 0;
  private acknowledgedUntil = 0;
  private modelStreak = 0;
  private modelProbability = 0;
  private log: MonitorEvent[] = [];
  private origin = 0;

  constructor(config: Partial<MonitorConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
  }

  get config(): Readonly<MonitorConfig> {
    return this.cfg;
  }

  get events(): readonly MonitorEvent[] {
    return this.log;
  }

  get trace(): readonly Sample[] {
    return this.samples;
  }

  get currentThreshold(): number | null {
    return this.threshold;
  }

  get neutral(): FaceMetrics | null {
    return this.neutralPose;
  }

  get currentState(): DriverState {
    return this.state;
  }

  /** Apply new thresholds mid-session; derived personal thresholds are recomputed. */
  configure(partial: Partial<MonitorConfig>): void {
    this.cfg = { ...this.cfg, ...partial };
    if (this.baseline !== null) this.threshold = this.baseline * this.cfg.closureRatio;
  }

  /** Begin (or restart) baseline capture. Existing counters are preserved. */
  beginCalibration(now: number): void {
    if (this.origin === 0) this.origin = now;
    this.calibrationEar = [];
    this.calibrationMar = [];
    this.calibrationPose = [];
    this.calibrationHeadPose = [];
    this.calibrationStart = now;
    this.calibrationValidMs = 0;
    this.calibrationLastT = null;
    this.calibrationPhase = 'running';
    this.calibrationMessage = 'Look straight at the camera with your eyes open';
    this.interrupt();
    this.transition('calibrating', 'measuring your open-eye baseline', now, null, 'calibration');
  }

  /**
   * The driver has confirmed they are awake. Suppresses repeat alarms for a
   * minute — without this the alarm fires every frame the eyes are shut, which
   * trains drivers to mute the system entirely.
   */
  acknowledge(now: number): void {
    this.acknowledgedUntil = now + this.cfg.snoozeMs;
    this.record('Acknowledged', `driver confirmed alert; alarm muted ${Math.round(this.cfg.snoozeMs / 1000)} s`, 'ok', now, 'system');
  }

  /**
   * Latest verdict from the trained classifier; null when it has none. The model
   * is one more input to `decide`: it can raise a warning the rules missed, but
   * the rules keep sole authority over critical, so it can never mute them.
   */
  setModelVerdict(verdict: { drowsy: boolean; probability: number } | null): void {
    this.modelStreak = verdict?.drowsy ? this.modelStreak + 1 : 0;
    this.modelProbability = verdict?.probability ?? 0;
  }

  isAcknowledged(now: number): boolean {
    return now < this.acknowledgedUntil;
  }

  reset(): void {
    const cfg = this.cfg;
    Object.assign(this, new DrowsinessAnalyzer(cfg));
  }

  /**
   * Forget any closure, yawn, nod or head-turn in progress. Used when the
   * signal is interrupted (camera paused, long tracking loss) so that the gap
   * is not later mistaken for one very long eye closure.
   */
  interrupt(): void {
    this.closedSince = null;
    this.closureEarSum = this.closureFrames = 0;
    this.mouthOpenSince = null;
    this.mouthPeak = 0;
    this.yawnCounted = false;
    this.headDownSince = null;
    this.nodCounted = false;
    this.turnedSince = null;
    this.distractionCounted = false;
    this.earWindow = [];
    this.lastT = null;
    this.modelStreak = 0;
  }

  /**
   * Called on frames where tracking found no face. A short dropout (motion
   * blur, a hand across the face) is ridden out; past the grace period the
   * driver is treated as not visible — and if their eyes were closing or head
   * dropping when they vanished, that is escalated rather than forgotten,
   * because slumping out of frame is exactly what a microsleep looks like.
   */
  faceLost(now: number): Reading {
    this.lastNow = now;
    this.faceLostSince ??= now;

    if (this.state === 'calibrating') {
      if (this.calibrationPhase === 'running' && now - this.calibrationStart > this.cfg.calibrationTimeoutMs) {
        this.failCalibration('No face detected — sit facing the camera and make sure your face is well lit', now);
      } else if (this.calibrationPhase === 'running') {
        this.calibrationMessage = 'Looking for your face…';
      }
      return this.snapshot(this.lastFrame, false, this.perclosNow, false);
    }

    let transitioned = false;
    const lostFor = now - this.faceLostSince;
    if (!this.faceLostHandled && lostFor >= this.cfg.faceLostGraceMs && this.threshold !== null) {
      this.faceLostHandled = true;
      const eyesWereClosed = this.closedSince !== null;
      const headWasDown = this.headDownSince !== null;
      if (eyesWereClosed) {
        const duration = now - this.closedSince!;
        this.microsleeps += 1;
        this.microsleepTimes.push(now);
        this.record('Microsleep', `${(duration / 1000).toFixed(1)} s with eyes shut, then face left frame`, 'crit', now, 'microsleep', round2(0.7 * this.quality));
        transitioned = this.transition('critical', 'eyes closed, then face lost', now, round2(0.7 * this.quality));
      } else if (headWasDown) {
        transitioned = this.transition('critical', 'head dropped out of view', now, round2(0.65 * this.quality));
      } else if (this.state === 'alert') {
        transitioned = this.transition('drowsy', 'no face in frame', now, null, 'face');
      } else {
        this.record('Face lost', 'no face in frame', 'warn', now, 'face');
      }
      this.interrupt();
    }
    return this.snapshot(this.lastFrame, false, this.perclosNow, transitioned);
  }

  /** Feed one frame. Returns the full derived picture for rendering. */
  push(frame: Frame, now: number): Reading {
    this.lastNow = now;
    this.lastFrame = frame;
    this.faceLostSince = null;
    this.faceLostHandled = false;
    // Exponential smoothing so one dark frame does not swing event confidence.
    this.quality = this.quality * 0.9 + clamp01(frame.quality ?? 1) * 0.1;

    if (this.state === 'calibrating') return this.calibrate(frame, now);
    if (this.threshold === null || this.neutralPose === null) {
      return this.snapshot(frame, false, 0, false);
    }

    const ear = this.smoothEar(frame.ear, now);
    this.trackPose(frame);

    const closed = ear < this.threshold;
    this.trackClosure(closed, ear, now);
    this.trackYawn(frame.mar, now);
    this.trackNod(now);
    this.trackDistraction(frame.pose != null && this.neutralHeadPose !== null, now);

    this.samples.push({ t: now, ear, closed });
    const cutoff = now - this.cfg.perclosWindowMs;
    while (this.samples.length > 0 && this.samples[0].t < cutoff) this.samples.shift();
    this.blinkTimes = this.blinkTimes.filter((t) => t > now - 60_000);
    const recent = now - this.cfg.scoreWindowMs;
    this.yawnTimes = this.yawnTimes.filter((t) => t > recent);
    this.microsleepTimes = this.microsleepTimes.filter((t) => t > recent);
    this.nodTimes = this.nodTimes.filter((t) => t > recent);
    this.trackBlinkRate(now);

    const perclos = this.perclos();
    this.perclosNow = perclos;
    // The live reading always shows PERCLOS, but it only drives alerts and the
    // score once the window holds enough data to mean something.
    const usable = this.perclosReliable(now) ? perclos : 0;
    this.computeScore(usable, now);
    const transitioned = this.decide(usable, now);
    return this.snapshot({ ...frame, ear }, closed, perclos, transitioned);
  }

  /** Percentage of sampled frames in the window with eyes closed. */
  perclos(): number {
    if (this.samples.length === 0) return 0;
    const shut = this.samples.reduce((n, s) => n + (s.closed ? 1 : 0), 0);
    return (shut / this.samples.length) * 100;
  }

  // ---------------------------------------------------------------- internals

  private smoothEar(raw: number, now: number): number {
    const dt = this.lastT === null ? 0 : now - this.lastT;
    this.lastT = now;
    const n = this.cfg.earSmoothingFrames;
    if (n <= 1 || dt > SMOOTHING_MAX_DT) {
      this.earWindow = [raw];
      return raw;
    }
    this.earWindow.push(raw);
    if (this.earWindow.length > n) this.earWindow.shift();
    return median(this.earWindow);
  }

  private trackPose(frame: Frame): void {
    if (frame.pose && this.neutralHeadPose) {
      this.pitch = frame.pose.pitch - this.neutralHeadPose.pitch;
      this.yaw = frame.pose.yaw - this.neutralHeadPose.yaw;
      this.roll = frame.pose.roll - this.neutralHeadPose.roll;
    } else {
      // No depth available: fall back to the 2-D foreshortening estimate.
      this.pitch = headPitchDegrees(frame.face, this.neutralPose!);
      this.yaw = 0;
      this.roll = 0;
    }
  }

  private calibrate(frame: Frame, now: number): Reading {
    if (this.calibrationPhase !== 'running') return this.snapshot(frame, false, 0, false);

    const dt = this.calibrationLastT === null ? 0 : Math.min(200, now - this.calibrationLastT);
    this.calibrationLastT = now;
    this.calibrationValidMs += dt;
    this.calibrationEar.push(frame.ear);
    this.calibrationMar.push(frame.mar);
    this.calibrationPose.push(frame.face);
    if (frame.pose) this.calibrationHeadPose.push(frame.pose);

    // Live guidance from the frame in hand, so the driver can correct early.
    const { calibrationMinEar, calibrationMaxYaw, calibrationMaxPitch } = this.cfg;
    this.calibrationMessage =
      frame.pose && Math.abs(frame.pose.yaw) > calibrationMaxYaw
        ? 'Turn your head toward the camera'
        : frame.pose && Math.abs(frame.pose.pitch) > calibrationMaxPitch
          ? 'Adjust the camera so it sees your face straight on'
          : frame.ear < calibrationMinEar
            ? 'Keep your eyes open'
            : 'Hold still and look at the camera';

    if (this.calibrationValidMs >= this.cfg.calibrationMs) return this.finishCalibration(frame, now);
    if (now - this.calibrationStart > this.cfg.calibrationTimeoutMs) {
      this.failCalibration('Could not collect a steady baseline in time — hold still and try again', now);
    }
    return this.snapshot(frame, false, 0, false);
  }

  private finishCalibration(frame: Frame, now: number): Reading {
    const c = this.cfg;
    if (this.calibrationEar.length < c.calibrationMinFrames) {
      this.failCalibration('Too few camera frames — close other apps using the camera and try again', now);
      return this.snapshot(frame, false, 0, false);
    }

    const baseline = trimmedMean(this.calibrationEar);
    const sorted = [...this.calibrationEar].sort((a, b) => a - b);
    const cut = Math.floor(sorted.length * 0.2);
    const kept = sorted.slice(cut, sorted.length - cut || undefined);
    const variation = baseline > 0 ? standardDeviation(kept) / baseline : Infinity;

    let failure: string | null = null;
    if (!(baseline >= c.calibrationMinEar)) {
      failure = 'Eyes appear closed or hidden — look at the camera with eyes open (remove sunglasses)';
    } else if (variation > c.calibrationMaxVariation) {
      failure = 'Readings were unstable — hold your head still and try not to blink repeatedly';
    } else if (this.calibrationHeadPose.length > 0) {
      const yaw = median(this.calibrationHeadPose.map((p) => p.yaw));
      const pitch = median(this.calibrationHeadPose.map((p) => p.pitch));
      if (Math.abs(yaw) > c.calibrationMaxYaw) failure = `Head turned ${Math.abs(yaw).toFixed(0)}° — face the camera directly`;
      else if (Math.abs(pitch) > c.calibrationMaxPitch) failure = 'Camera angle too steep — position it closer to eye level';
    }
    if (failure) {
      this.failCalibration(failure, now);
      return this.snapshot(frame, false, 0, false);
    }

    this.baseline = baseline;
    this.threshold = baseline * c.closureRatio;
    const restingMar = trimmedMean(this.calibrationMar);
    this.yawnThreshold = Math.max(c.yawnRatio, restingMar + c.yawnMarMargin);
    this.neutralPose = {
      foreshortening: trimmedMean(this.calibrationPose.map((p) => p.foreshortening)),
      balance: trimmedMean(this.calibrationPose.map((p) => p.balance)),
    };
    this.neutralHeadPose =
      this.calibrationHeadPose.length > 0
        ? {
            pitch: trimmedMean(this.calibrationHeadPose.map((p) => p.pitch)),
            yaw: trimmedMean(this.calibrationHeadPose.map((p) => p.yaw)),
            roll: trimmedMean(this.calibrationHeadPose.map((p) => p.roll)),
          }
        : null;
    this.calibrationQuality = round2(clamp01(1 - variation / c.calibrationMaxVariation) * 0.5 + 0.5);
    this.calibrationPhase = 'success';
    this.calibrationMessage = `Baseline EAR ${baseline.toFixed(3)} · closure threshold ${this.threshold.toFixed(3)}`;
    this.monitoringSince ??= now;
    this.record(
      'Calibrated',
      `baseline ${baseline.toFixed(3)}, threshold ${this.threshold.toFixed(3)}`,
      'ok',
      now,
      'calibration',
      this.calibrationQuality,
    );
    this.interrupt();
    this.transition('alert', 'eyes open', now);
    return this.snapshot(frame, false, 0, true);
  }

  private failCalibration(message: string, now: number): void {
    this.calibrationPhase = 'failed';
    this.calibrationMessage = message;
    this.record('Calibration failed', message, 'warn', now, 'calibration');
    // A recalibration that fails keeps the previous, known-good baseline
    // rather than leaving a driver mid-journey unmonitored.
    if (this.threshold !== null) this.transition('alert', 'kept previous calibration', now);
  }

  /** Classify a completed eye closure as either a blink or a microsleep. */
  private trackClosure(closed: boolean, ear: number, now: number): void {
    if (closed) {
      this.closedSince ??= now;
      this.closureEarSum += ear;
      this.closureFrames += 1;
      return;
    }
    if (this.closedSince === null) return;

    const duration = now - this.closedSince;
    const meanEar = this.closureFrames > 0 ? this.closureEarSum / this.closureFrames : ear;
    this.closedSince = null;
    this.closureEarSum = this.closureFrames = 0;

    if (duration >= this.cfg.microsleepMs) {
      this.microsleeps += 1;
      this.microsleepTimes.push(now);
      const depth = clamp01((1 - meanEar / this.threshold!) / 0.4);
      const length = clamp01((duration - this.cfg.microsleepMs) / this.cfg.microsleepMs);
      this.record(
        'Microsleep',
        `${(duration / 1000).toFixed(1)} s with eyes shut`,
        'crit',
        now,
        'microsleep',
        round2((0.6 + 0.25 * depth + 0.15 * length) * this.quality),
      );
    } else if (duration <= this.cfg.blinkMaxMs) {
      this.blinkTimes.push(now);
      this.totalBlinks += 1;
    }
  }

  private trackYawn(mar: number, now: number): void {
    const limit = this.yawnThreshold ?? this.cfg.yawnRatio;
    if (mar <= limit) {
      this.mouthOpenSince = null;
      this.yawnCounted = false;
      this.mouthPeak = 0;
      return;
    }
    this.mouthOpenSince ??= now;
    this.mouthPeak = Math.max(this.mouthPeak, mar);
    if (!this.yawnCounted && now - this.mouthOpenSince >= this.cfg.yawnHoldMs) {
      this.yawns += 1;
      this.yawnTimes.push(now);
      this.yawnCounted = true;
      this.record(
        'Yawn',
        `mouth held open ${(this.cfg.yawnHoldMs / 1000).toFixed(1)} s (MAR ${this.mouthPeak.toFixed(2)})`,
        'warn',
        now,
        'yawn',
        round2((0.6 + 0.4 * clamp01((this.mouthPeak - limit) / 0.3)) * this.quality),
      );
    }
  }

  private trackNod(now: number): void {
    if (this.pitch > -this.cfg.nodPitchDegrees) {
      this.headDownSince = null;
      this.nodCounted = false;
      return;
    }
    this.headDownSince ??= now;
    if (!this.nodCounted && now - this.headDownSince >= this.cfg.nodHoldMs) {
      this.nods += 1;
      this.nodTimes.push(now);
      this.nodCounted = true;
      this.record(
        'Head nod',
        `head down ${Math.abs(this.pitch).toFixed(0)}° and holding`,
        'crit',
        now,
        'nod',
        round2((0.6 + 0.4 * clamp01((Math.abs(this.pitch) - this.cfg.nodPitchDegrees) / 15)) * this.quality),
      );
    }
  }

  private trackDistraction(havePose: boolean, now: number): void {
    if (!havePose || Math.abs(this.yaw) < this.cfg.distractionYawDegrees) {
      this.turnedSince = null;
      this.distractionCounted = false;
      return;
    }
    this.turnedSince ??= now;
    if (!this.distractionCounted && now - this.turnedSince >= this.cfg.distractionHoldMs) {
      this.distractions += 1;
      this.distractionCounted = true;
      this.record(
        'Looking away',
        `head turned ${Math.abs(this.yaw).toFixed(0)}° for over ${(this.cfg.distractionHoldMs / 1000).toFixed(1)} s`,
        'warn',
        now,
        'distraction',
        round2((0.6 + 0.4 * clamp01((Math.abs(this.yaw) - this.cfg.distractionYawDegrees) / 20)) * this.quality),
      );
    }
  }

  private get distracted(): boolean {
    return this.distractionCounted && this.turnedSince !== null;
  }

  /** True once the PERCLOS window spans at least `perclosMinMs` of samples. */
  private perclosReliable(now: number): boolean {
    return this.samples.length > 0 && now - this.samples[0].t >= this.cfg.perclosMinMs;
  }

  /** A full minute of data is needed before a per-minute rate means anything. */
  private blinkRateReliable(now: number): boolean {
    return this.monitoringSince !== null && now - this.monitoringSince >= 60_000;
  }

  private trackBlinkRate(now: number): void {
    const rate = this.blinkTimes.length;
    if (!this.blinkRateReliable(now) || rate <= this.cfg.highBlinkRate) return;
    if (now - this.lastBlinkWarning < 60_000) return;
    this.lastBlinkWarning = now;
    this.record(
      'Excessive blinking',
      `${rate} blinks in the last minute (normal 12–20)`,
      'warn',
      now,
      'blinking',
      round2((0.55 + 0.45 * clamp01((rate - this.cfg.highBlinkRate) / this.cfg.highBlinkRate)) * this.quality),
    );
  }

  /**
   * 0–100 composite. Every term is a real measured signal, normalised against
   * its own threshold, so the number can be explained part by part:
   *
   *   PERCLOS        up to 40  (linear to the critical PERCLOS level)
   *   eye closure    up to 20  (current closure longer than a blink)
   *   microsleeps    up to 20  (3+ in the last 5 minutes = full)
   *   yawns          up to 15  (3+ in the last 5 minutes = full)
   *   head nods      up to 15  (2+ in the last 5 minutes = full)
   *   blink rate     up to 10  (outside the normal band, after 60 s of data)
   */
  private computeScore(perclos: number, now: number): void {
    const c = this.cfg;
    const closedFor = this.closedSince !== null ? now - this.closedSince : 0;
    const rate = this.blinkTimes.length;
    let blink = 0;
    if (this.blinkRateReliable(now)) {
      if (rate > c.highBlinkRate) blink = clamp01((rate - c.highBlinkRate) / 15);
      else if (rate < c.lowBlinkRate) blink = clamp01((c.lowBlinkRate - rate) / c.lowBlinkRate);
    }
    const parts: ScoreParts = {
      perclos: clamp01(perclos / c.criticalPerclos) * 40,
      closure: closedFor > c.blinkMaxMs ? clamp01((closedFor - c.blinkMaxMs) / 1600) * 20 : 0,
      microsleeps: clamp01(this.microsleepTimes.length / 3) * 20,
      yawns: clamp01(this.yawnTimes.length / 3) * 15,
      nods: clamp01(this.nodTimes.length / 2) * 15,
      blinkRate: blink * 10,
    };
    this.scoreParts = parts;
    this.score = Math.min(100, Object.values(parts).reduce((s, v) => s + v, 0));
  }

  private margin(value: number, threshold: number, span: number): number {
    return round2((0.55 + 0.45 * clamp01((value - threshold) / span)) * this.quality);
  }

  private decide(perclos: number, now: number): boolean {
    const { microsleepMs, criticalPerclos, drowsyPerclos, yawnsForDrowsy, dwellMs, scoreCritical, scoreDrowsy } = this.cfg;
    const closedFor = this.closedSince !== null ? now - this.closedSince : 0;
    const shutNow = closedFor >= microsleepMs;
    const noddingNow = this.nodCounted && this.headDownSince !== null;
    const recentYawns = this.yawnTimes.length;

    if (shutNow) return this.transition('critical', 'eyes have been shut', now, this.margin(closedFor, microsleepMs, microsleepMs));
    if (noddingNow) return this.transition('critical', 'head has dropped forward', now, this.margin(-this.pitch, this.cfg.nodPitchDegrees, 15));
    if (perclos >= criticalPerclos) {
      return this.transition('critical', `PERCLOS ${perclos.toFixed(0)}%`, now, this.margin(perclos, criticalPerclos, criticalPerclos / 2));
    }
    if (this.score >= scoreCritical) {
      return this.transition('critical', `drowsiness score ${this.score.toFixed(0)}`, now, this.margin(this.score, scoreCritical, 20));
    }
    if (perclos >= drowsyPerclos) {
      return this.transition('drowsy', `PERCLOS ${perclos.toFixed(0)}%`, now, this.margin(perclos, drowsyPerclos, drowsyPerclos));
    }
    // Yawns count over a rolling window, not the whole session — otherwise
    // two yawns early in a long drive would pin the driver at drowsy forever.
    if (recentYawns >= yawnsForDrowsy) return this.transition('drowsy', 'repeated yawning', now, this.margin(recentYawns, yawnsForDrowsy, 2));
    if (this.score >= scoreDrowsy) {
      return this.transition('drowsy', `drowsiness score ${this.score.toFixed(0)}`, now, this.margin(this.score, scoreDrowsy, 20));
    }
    if (this.modelStreak >= this.cfg.modelConsecutive) {
      const percent = (this.modelProbability * 100).toFixed(0);
      return this.transition('drowsy', `ML model: drowsy pattern (${percent}%)`, now, round2(this.modelProbability * this.quality));
    }
    // De-escalation waits out the dwell time. Without this the display flickers
    // between states on readings sitting right at the threshold.
    if (now - this.stateSince < dwellMs) return false;
    return this.transition('alert', 'eyes open', now);
  }

  private transition(
    next: DriverState,
    reason: string,
    now: number,
    confidence: number | null = null,
    kind: EventKind = 'state',
  ): boolean {
    this.reason = reason;
    if (next === this.state) return false;

    const previous = this.state;
    this.state = next;
    this.stateSince = now;

    if (next === 'critical') {
      this.criticals += 1;
      this.record('Critical', reason, 'crit', now, kind, confidence);
    } else if (next === 'drowsy') {
      this.warnings += 1;
      this.record('Drowsy', reason, 'warn', now, kind, confidence);
    } else if (next === 'alert' && previous !== 'calibrating') {
      this.record('Recovered', reason, 'ok', now, kind);
    }
    return true;
  }

  private record(
    label: string,
    detail: string,
    severity: Severity,
    now: number,
    kind: EventKind,
    confidence: number | null = null,
  ): void {
    if (this.origin === 0) this.origin = now;
    this.log.push({ at: now - this.origin, label, detail, severity, kind, confidence, status: this.state });
  }

  private attention(): Attention {
    if (this.state === 'idle') return 'inactive';
    if (this.state === 'calibrating') return 'calibrating';
    if (this.faceLostSince !== null && this.lastNow - this.faceLostSince >= this.cfg.faceLostGraceMs) return 'no-face';
    if (this.state === 'critical') return 'critical';
    if (this.distracted) return 'distracted';
    if (this.state === 'drowsy') return this.reason === 'no face in frame' ? 'distracted' : 'fatigued';
    return 'attentive';
  }

  private snapshot(frame: Frame, closed: boolean, perclos: number, transitioned: boolean): Reading {
    const progress =
      this.calibrationPhase === 'success'
        ? 1
        : this.calibrationPhase === 'idle'
          ? 0
          : clamp01(this.calibrationValidMs / this.cfg.calibrationMs);
    return {
      state: this.state,
      reason: this.reason,
      ear: frame.ear,
      earLeft: frame.earLeft ?? frame.ear,
      earRight: frame.earRight ?? frame.ear,
      mar: frame.mar,
      closed,
      closedForMs: this.closedSince !== null ? this.lastNow - this.closedSince : 0,
      perclos,
      blinksPerMinute: this.blinkTimes.length,
      totalBlinks: this.totalBlinks,
      yawns: this.yawns,
      microsleeps: this.microsleeps,
      nods: this.nods,
      distractions: this.distractions,
      warnings: this.warnings,
      criticals: this.criticals,
      pitch: this.pitch,
      yaw: this.yaw,
      roll: this.roll,
      score: this.score,
      scoreParts: this.scoreParts,
      attention: this.attention(),
      faceVisible: this.faceLostSince === null,
      quality: this.quality,
      threshold: this.threshold,
      baseline: this.baseline,
      yawnThreshold: this.yawnThreshold,
      calibration: {
        phase: this.calibrationPhase,
        progress,
        message: this.calibrationMessage,
        quality: this.calibrationQuality,
      },
      acknowledged: this.isAcknowledged(this.lastNow),
      transitioned,
    };
  }
}

/** Reading before anything has run. */
export const IDLE_READING: Reading = new DrowsinessAnalyzer().push(EMPTY_FRAME, 0);
