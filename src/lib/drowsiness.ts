import { DEFAULT_CONFIG, type MonitorConfig } from './config';
import { headPitchDegrees, trimmedMean, type FaceMetrics } from './geometry';

export type DriverState = 'idle' | 'calibrating' | 'alert' | 'drowsy' | 'critical';
export type Severity = 'ok' | 'warn' | 'crit';

export interface MonitorEvent {
  at: number;
  label: string;
  detail: string;
  severity: Severity;
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
}

export interface Reading {
  state: DriverState;
  reason: string;
  ear: number;
  mar: number;
  closed: boolean;
  perclos: number;
  blinksPerMinute: number;
  yawns: number;
  microsleeps: number;
  nods: number;
  /** Signed degrees from the calibrated neutral pose; negative is nodding forward. */
  pitch: number;
  threshold: number | null;
  baseline: number | null;
  acknowledged: boolean;
  /** True on the frame a state transition happens, so the caller can alarm once. */
  transitioned: boolean;
}

/**
 * Consumes per-frame facial ratios and decides how awake the driver is.
 *
 * Deliberately knows nothing about cameras, canvases or React: every method
 * takes the current timestamp as an argument rather than reading a clock. That
 * is what makes the whole decision path unit-testable without a browser.
 */
export class DrowsinessAnalyzer {
  private readonly config: MonitorConfig;

  private state: DriverState = 'idle';
  private stateSince = 0;
  private reason = '';

  private baseline: number | null = null;
  private threshold: number | null = null;
  private neutralPose: FaceMetrics | null = null;
  private calibrationEar: number[] = [];
  private calibrationPose: FaceMetrics[] = [];
  private calibrationStart = 0;

  private samples: Sample[] = [];
  private blinkTimes: number[] = [];
  private closedSince: number | null = null;
  private mouthOpenSince: number | null = null;
  private yawnCounted = false;
  private headDownSince: number | null = null;
  private nodCounted = false;

  private yawns = 0;
  private microsleeps = 0;
  private nods = 0;
  private pitch = 0;
  private acknowledgedUntil = 0;
  private log: MonitorEvent[] = [];
  private origin = 0;

  constructor(config: Partial<MonitorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
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

  /** Begin (or restart) baseline capture. Existing counters are preserved. */
  beginCalibration(now: number): void {
    if (this.origin === 0) this.origin = now;
    this.calibrationEar = [];
    this.calibrationPose = [];
    this.calibrationStart = now;
    this.transition('calibrating', 'measuring your open-eye baseline', now);
  }

  /**
   * The driver has confirmed they are awake. Suppresses repeat alarms for a
   * minute — without this the alarm fires every frame the eyes are shut, which
   * trains drivers to mute the system entirely.
   */
  acknowledge(now: number): void {
    this.acknowledgedUntil = now + this.config.snoozeMs;
    this.record('Acknowledged', 'driver confirmed alert; alarm muted 60 s', 'ok', now);
  }

  isAcknowledged(now: number): boolean {
    return now < this.acknowledgedUntil;
  }

  reset(): void {
    this.state = 'idle';
    this.baseline = this.threshold = null;
    this.neutralPose = null;
    this.samples = [];
    this.blinkTimes = [];
    this.closedSince = this.mouthOpenSince = this.headDownSince = null;
    this.yawnCounted = this.nodCounted = false;
    this.yawns = this.microsleeps = this.nods = 0;
    this.pitch = 0;
    this.acknowledgedUntil = 0;
    this.log = [];
    this.origin = 0;
  }

  /** Called when tracking is lost, so a driver who has looked away is not read as alert. */
  faceLost(now: number): void {
    if (this.state === 'alert') this.transition('drowsy', 'no face in frame', now);
  }

  /** Feed one frame. Returns the full derived picture for rendering. */
  push(frame: Frame, now: number): Reading {
    if (this.state === 'calibrating') return this.calibrate(frame, now);
    if (this.threshold === null || this.neutralPose === null) {
      return this.snapshot(frame, false, 0, false);
    }

    this.pitch = headPitchDegrees(frame.face, this.neutralPose);
    const closed = frame.ear < this.threshold;
    this.trackClosure(closed, now);
    this.trackYawn(frame.mar, now);
    this.trackNod(now);

    this.samples.push({ t: now, ear: frame.ear, closed });
    const cutoff = now - this.config.perclosWindowMs;
    while (this.samples.length > 0 && this.samples[0].t < cutoff) this.samples.shift();
    this.blinkTimes = this.blinkTimes.filter((t) => t > now - 60_000);

    const perclos = this.perclos();
    const transitioned = this.decide(perclos, now);
    return this.snapshot(frame, closed, perclos, transitioned);
  }

  /** Percentage of sampled frames in the window with eyes closed. */
  perclos(): number {
    if (this.samples.length === 0) return 0;
    const shut = this.samples.reduce((n, s) => n + (s.closed ? 1 : 0), 0);
    return (shut / this.samples.length) * 100;
  }

  // ---------------------------------------------------------------- internals

  private calibrate(frame: Frame, now: number): Reading {
    this.calibrationEar.push(frame.ear);
    this.calibrationPose.push(frame.face);
    if (now - this.calibrationStart < this.config.calibrationMs) {
      return this.snapshot(frame, false, 0, false);
    }
    this.baseline = trimmedMean(this.calibrationEar);
    this.threshold = this.baseline * this.config.closureRatio;
    this.neutralPose = {
      foreshortening: trimmedMean(this.calibrationPose.map((p) => p.foreshortening)),
      balance: trimmedMean(this.calibrationPose.map((p) => p.balance)),
    };
    this.record(
      'Calibrated',
      `baseline ${this.baseline.toFixed(3)}, threshold ${this.threshold.toFixed(3)}`,
      'ok',
      now,
    );
    this.transition('alert', 'eyes open', now);
    return this.snapshot(frame, false, 0, true);
  }

  /** Classify a completed eye closure as either a blink or a microsleep. */
  private trackClosure(closed: boolean, now: number): void {
    if (closed) {
      this.closedSince ??= now;
      return;
    }
    if (this.closedSince === null) return;

    const duration = now - this.closedSince;
    this.closedSince = null;

    if (duration >= this.config.microsleepMs) {
      this.microsleeps += 1;
      this.record('Microsleep', `${(duration / 1000).toFixed(1)} s with eyes shut`, 'crit', now);
    } else if (duration <= this.config.blinkMaxMs) {
      this.blinkTimes.push(now);
    }
  }

  private trackYawn(mar: number, now: number): void {
    if (mar <= this.config.yawnRatio) {
      this.mouthOpenSince = null;
      this.yawnCounted = false;
      return;
    }
    this.mouthOpenSince ??= now;
    if (!this.yawnCounted && now - this.mouthOpenSince >= this.config.yawnHoldMs) {
      this.yawns += 1;
      this.yawnCounted = true;
      this.record('Yawn', `mouth held open ${(this.config.yawnHoldMs / 1000).toFixed(1)} s`, 'warn', now);
    }
  }

  private trackNod(now: number): void {
    if (this.pitch > -this.config.nodPitchDegrees) {
      this.headDownSince = null;
      this.nodCounted = false;
      return;
    }
    this.headDownSince ??= now;
    if (!this.nodCounted && now - this.headDownSince >= this.config.nodHoldMs) {
      this.nods += 1;
      this.nodCounted = true;
      this.record('Head nod', `head down ${Math.abs(this.pitch).toFixed(0)}° and holding`, 'crit', now);
    }
  }

  private decide(perclos: number, now: number): boolean {
    const { microsleepMs, criticalPerclos, drowsyPerclos, yawnsForDrowsy, dwellMs } = this.config;
    const shutNow = this.closedSince !== null && now - this.closedSince >= microsleepMs;
    const noddingNow = this.nodCounted && this.headDownSince !== null;

    if (shutNow || noddingNow || perclos >= criticalPerclos) {
      const why = shutNow
        ? 'eyes have been shut'
        : noddingNow
          ? 'head has dropped forward'
          : `PERCLOS ${perclos.toFixed(0)}%`;
      return this.transition('critical', why, now);
    }
    if (perclos >= drowsyPerclos || this.yawns >= yawnsForDrowsy) {
      return this.transition(
        'drowsy',
        perclos >= drowsyPerclos ? `PERCLOS ${perclos.toFixed(0)}%` : 'repeated yawning',
        now,
      );
    }
    // De-escalation waits out the dwell time. Without this the display flickers
    // between states on readings sitting right at the threshold.
    if (now - this.stateSince < dwellMs) return false;
    return this.transition('alert', 'eyes open', now);
  }

  private transition(next: DriverState, reason: string, now: number): boolean {
    this.reason = reason;
    if (next === this.state) return false;

    const previous = this.state;
    this.state = next;
    this.stateSince = now;

    if (next === 'critical') this.record('Critical', reason, 'crit', now);
    else if (next === 'drowsy') this.record('Drowsy', reason, 'warn', now);
    else if (next === 'alert' && previous !== 'calibrating') this.record('Recovered', reason, 'ok', now);
    return true;
  }

  private record(label: string, detail: string, severity: Severity, now: number): void {
    if (this.origin === 0) this.origin = now;
    this.log.push({ at: now - this.origin, label, detail, severity });
  }

  private snapshot(frame: Frame, closed: boolean, perclos: number, transitioned: boolean): Reading {
    return {
      state: this.state,
      reason: this.reason,
      ear: frame.ear,
      mar: frame.mar,
      closed,
      perclos,
      blinksPerMinute: this.blinkTimes.length,
      yawns: this.yawns,
      microsleeps: this.microsleeps,
      nods: this.nods,
      pitch: this.pitch,
      threshold: this.threshold,
      baseline: this.baseline,
      acknowledged: false,
      transitioned,
    };
  }
}
