import { beforeEach, describe, expect, it } from 'vitest';
import { DrowsinessAnalyzer, type Frame } from '../drowsiness';
import type { FaceMetrics } from '../geometry';

const OPEN = 0.32;
const SHUT = 0.1;
const CLOSED_MOUTH = 0.05;
const OPEN_MOUTH = 0.7;

/** Head square-on to the camera. */
const NEUTRAL: FaceMetrics = { foreshortening: 2.0, balance: 1.0 };
/** Head dropped forward: vertical axis foreshortened, forehead more visible. */
const HEAD_DOWN: FaceMetrics = { foreshortening: 2.0 * Math.cos((25 * Math.PI) / 180), balance: 1.4 };

const frame = (ear: number, mar = CLOSED_MOUTH, face = NEUTRAL): Frame => ({ ear, mar, face });

/**
 * The analyzer takes `now` as an argument rather than reading a clock, so these
 * tests drive several minutes of driving in a few milliseconds.
 */
class Driver {
  t = 0;
  constructor(readonly analyzer: DrowsinessAnalyzer) {}

  /** Advance `ms` of footage at 30 fps with fixed facial measurements. */
  run(ms: number, ear: number, mar = CLOSED_MOUTH, face = NEUTRAL) {
    const step = 1000 / 30;
    for (let elapsed = 0; elapsed < ms; elapsed += step) {
      this.t += step;
      this.analyzer.push(frame(ear, mar, face), this.t);
    }
    return this;
  }

  read(face = NEUTRAL) {
    return this.analyzer.push(frame(OPEN, CLOSED_MOUTH, face), this.t);
  }

  calibrate() {
    this.analyzer.beginCalibration(this.t);
    return this.run(4500, OPEN);
  }

  get state() {
    return this.read().state;
  }

  labels() {
    return this.analyzer.events.map((e) => e.label);
  }
}

let driver: Driver;
beforeEach(() => {
  driver = new Driver(new DrowsinessAnalyzer());
});

describe('calibration', () => {
  it('derives the threshold from the driver rather than a fixed constant', () => {
    driver.calibrate();
    expect(driver.analyzer.currentThreshold).toBeCloseTo(OPEN * 0.78, 4);
  });

  it('adapts to a driver whose open eyes read much narrower', () => {
    const narrow = new Driver(new DrowsinessAnalyzer());
    narrow.analyzer.beginCalibration(0);
    narrow.run(4500, 0.19);
    // A hard-coded 0.25 cutoff would report this driver as permanently asleep.
    expect(narrow.analyzer.currentThreshold!).toBeLessThan(0.25);
    expect(narrow.run(2000, 0.19).state).toBe('alert');
  });

  it('emits nothing but a calibration event before it finishes', () => {
    driver.analyzer.beginCalibration(0);
    driver.run(1000, SHUT);
    expect(driver.labels()).toEqual([]);
  });
});

describe('blink versus microsleep', () => {
  it('does not log an ordinary blink as an event', () => {
    driver.calibrate().run(200, SHUT).run(2000, OPEN);
    expect(driver.labels()).not.toContain('Microsleep');
  });

  it('logs a closure past 500 ms as a microsleep', () => {
    driver.calibrate().run(900, SHUT).run(500, OPEN);
    expect(driver.labels()).toContain('Microsleep');
  });

  it('counts blinks toward the per-minute rate', () => {
    driver.calibrate();
    for (let i = 0; i < 5; i++) driver.run(150, SHUT).run(1000, OPEN);
    expect(driver.read().blinksPerMinute).toBe(5);
  });
});

describe('escalation', () => {
  it('starts alert after calibration', () => {
    expect(driver.calibrate().state).toBe('alert');
  });

  it('goes critical while the eyes are still shut, not after they reopen', () => {
    driver.calibrate().run(700, SHUT);
    expect(driver.state).toBe('critical');
  });

  it('calls drowsy once PERCLOS crosses 15 percent', () => {
    driver.calibrate();
    // Roughly a fifth of each cycle spent with eyes closed, none long enough
    // on its own to be a microsleep.
    for (let i = 0; i < 30; i++) driver.run(300, SHUT).run(1200, OPEN);
    const reading = driver.read();
    expect(reading.perclos).toBeGreaterThan(15);
    expect(reading.state).toBe('drowsy');
  });

  it('does not call drowsy on one normal blink right after calibration', () => {
    // The PERCLOS window is nearly empty here, so a 230 ms blink is ~30% of it.
    driver.calibrate().run(300, OPEN).run(230, SHUT).run(100, OPEN);
    const reading = driver.read();
    expect(reading.perclos).toBeGreaterThan(15);
    expect(reading.state).toBe('alert');
    expect(reading.scoreParts.perclos).toBe(0);
  });

  it('still escalates on a microsleep before the PERCLOS window has filled', () => {
    driver.calibrate().run(300, OPEN).run(700, SHUT);
    expect(driver.state).toBe('critical');
  });

  it('calls drowsy on repeated yawning even with PERCLOS low', () => {
    driver.calibrate();
    for (let i = 0; i < 2; i++) driver.run(1500, OPEN, OPEN_MOUTH).run(1000, OPEN);
    const reading = driver.read();
    expect(reading.yawns).toBe(2);
    expect(reading.state).toBe('drowsy');
  });

  it('ignores brief mouth movement, so talking is not a yawn', () => {
    driver.calibrate();
    for (let i = 0; i < 10; i++) driver.run(300, OPEN, OPEN_MOUTH).run(300, OPEN);
    expect(driver.read().yawns).toBe(0);
  });
});

describe('de-escalation', () => {
  it('holds a critical state briefly instead of flickering back', () => {
    driver.calibrate().run(700, SHUT).run(300, OPEN);
    expect(driver.state).toBe('critical');
  });

  it('recovers once the driver is reliably alert again', () => {
    driver.calibrate().run(700, SHUT).run(60_000, OPEN);
    expect(driver.state).toBe('alert');
    expect(driver.labels()).toContain('Recovered');
  });
});

describe('face tracking loss', () => {
  it('degrades to drowsy rather than reporting a driver it cannot see as alert', () => {
    driver.calibrate();
    driver.analyzer.faceLost(driver.t);
    driver.analyzer.faceLost(driver.t + 2000);
    expect(driver.analyzer.events.at(-1)?.detail).toBe('no face in frame');
  });

  it('rides out a brief tracking dropout without raising anything', () => {
    driver.calibrate();
    const reading = driver.analyzer.faceLost(driver.t + 300);
    expect(reading.state).toBe('alert');
    expect(driver.labels()).not.toContain('Drowsy');
  });

  it('escalates to critical when the face vanishes while the eyes are shut', () => {
    driver.calibrate().run(300, SHUT);
    driver.analyzer.faceLost(driver.t + 100);
    const reading = driver.analyzer.faceLost(driver.t + 2000);
    expect(reading.state).toBe('critical');
    expect(driver.labels()).toContain('Microsleep');
  });

  it('does not count a long dropout as one giant eye closure when the face returns', () => {
    driver.calibrate().run(200, SHUT);
    driver.analyzer.faceLost(driver.t + 50);
    driver.analyzer.faceLost(driver.t + 3000);
    driver.t += 3000;
    driver.run(5000, OPEN);
    // One microsleep from "eyes shut then face lost"; reopening must not add another.
    expect(driver.read().microsleeps).toBe(1);
  });
});

describe('calibration quality', () => {
  it('fails when the eyes are closed throughout, instead of learning a bogus baseline', () => {
    driver.analyzer.beginCalibration(0);
    const reading = driver.run(4500, 0.05).read();
    expect(reading.calibration.phase).toBe('failed');
    expect(driver.analyzer.currentThreshold).toBeNull();
    expect(driver.labels()).toContain('Calibration failed');
  });

  it('fails when the head is turned well away from the camera', () => {
    driver.analyzer.beginCalibration(0);
    for (let t = 0; t < 4500; t += 1000 / 30) {
      driver.t = t;
      driver.analyzer.push({ ...frame(OPEN), pose: { pitch: 0, yaw: 40, roll: 0 } }, t);
    }
    expect(driver.read().calibration.phase).toBe('failed');
  });

  it('times out when no face is ever seen', () => {
    driver.analyzer.beginCalibration(0);
    driver.analyzer.faceLost(1000);
    const reading = driver.analyzer.faceLost(16_000);
    expect(reading.calibration.phase).toBe('failed');
  });

  it('keeps the previous baseline when a recalibration fails', () => {
    driver.calibrate();
    const before = driver.analyzer.currentThreshold;
    driver.analyzer.beginCalibration(driver.t);
    driver.run(4500, 0.05);
    expect(driver.analyzer.currentThreshold).toBe(before);
    // Monitoring resumes on the old baseline — and those shut eyes now alarm.
    expect(driver.analyzer.currentState).not.toBe('calibrating');
    expect(driver.labels()).toContain('Calibration failed');
  });

  it('reports progress while running', () => {
    driver.analyzer.beginCalibration(0);
    driver.run(2000, OPEN);
    const progress = driver.analyzer.push(frame(OPEN), driver.t).calibration.progress;
    expect(progress).toBeGreaterThan(0.4);
    expect(progress).toBeLessThan(0.6);
  });
});

describe('signal smoothing', () => {
  it('ignores a single-frame EAR glitch', () => {
    driver.calibrate().run(1000 / 30, SHUT).run(2000, OPEN);
    expect(driver.read().totalBlinks).toBe(0);
  });
});

describe('drowsiness score', () => {
  it('is zero for an alert driver', () => {
    expect(driver.calibrate().run(5000, OPEN).read().score).toBe(0);
  });

  it('rises with PERCLOS and explains itself part by part', () => {
    driver.calibrate();
    for (let i = 0; i < 30; i++) driver.run(300, SHUT).run(1200, OPEN);
    const r = driver.read();
    expect(r.score).toBeGreaterThan(15);
    expect(r.scoreParts.perclos).toBeGreaterThan(15);
    const sum = Object.values(r.scoreParts).reduce((a, b) => a + b, 0);
    expect(r.score).toBeCloseTo(Math.min(100, sum), 5);
  });
});

describe('distraction', () => {
  const TURNED = { pitch: 0, yaw: 45, roll: 0 };
  const AHEAD = { pitch: 0, yaw: 0, roll: 0 };
  const run = (ms: number, pose: typeof TURNED) => {
    for (let e = 0; e < ms; e += 1000 / 30) {
      driver.t += 1000 / 30;
      driver.analyzer.push({ ...frame(OPEN), pose }, driver.t);
    }
  };

  it('flags a sustained head turn away from the road', () => {
    driver.analyzer.beginCalibration(0);
    run(4500, AHEAD);
    run(3000, TURNED);
    expect(driver.labels()).toContain('Looking away');
  });

  it('ignores a quick mirror check', () => {
    driver.analyzer.beginCalibration(0);
    run(4500, AHEAD);
    for (let i = 0; i < 4; i++) {
      run(800, TURNED);
      run(1500, AHEAD);
    }
    expect(driver.labels()).not.toContain('Looking away');
  });
});

describe('events', () => {
  it('carry a confidence and the driver status for detections', () => {
    driver.calibrate().run(900, SHUT).run(500, OPEN);
    const microsleep = driver.analyzer.events.find((e) => e.label === 'Microsleep')!;
    expect(microsleep.confidence).toBeGreaterThan(0.5);
    expect(microsleep.confidence).toBeLessThanOrEqual(1);
    expect(microsleep.status).toBe('critical');
    expect(microsleep.kind).toBe('microsleep');
  });

  it('let a driver recover after yawns age out of the window', () => {
    driver.calibrate();
    for (let i = 0; i < 2; i++) driver.run(1500, OPEN, OPEN_MOUTH).run(1000, OPEN);
    expect(driver.state).toBe('drowsy');
    driver.run(310_000, OPEN);
    expect(driver.state).toBe('alert');
  });
});

describe('head pose', () => {
  it('reports roughly zero pitch while the driver holds the calibrated pose', () => {
    driver.calibrate();
    expect(Math.abs(driver.read().pitch)).toBeLessThan(1);
  });

  it('escalates when the head drops forward and stays down', () => {
    driver.calibrate().run(1200, OPEN, CLOSED_MOUTH, HEAD_DOWN);
    expect(driver.read(HEAD_DOWN).nods).toBe(1);
    expect(driver.labels()).toContain('Head nod');
  });

  it('ignores a brief glance down at the instruments', () => {
    // Checking mirrors and the dashboard dips the head for well under a second.
    driver.calibrate();
    for (let i = 0; i < 6; i++) driver.run(400, OPEN, CLOSED_MOUTH, HEAD_DOWN).run(600, OPEN);
    expect(driver.read().nods).toBe(0);
  });

  it('recovers pitch sign — nodding forward reads negative', () => {
    driver.calibrate();
    // Read the pose on a head-down frame; a neutral frame would reset it.
    expect(driver.read(HEAD_DOWN).pitch).toBeLessThan(0);
  });
});

describe('driver acknowledgement', () => {
  it('mutes repeat alarms for the snooze window once confirmed awake', () => {
    driver.calibrate().run(700, SHUT);
    expect(driver.state).toBe('critical');

    driver.analyzer.acknowledge(driver.t);
    expect(driver.analyzer.isAcknowledged(driver.t + 30_000)).toBe(true);
    expect(driver.labels()).toContain('Acknowledged');
  });

  it('lets the alarm return once the snooze expires', () => {
    driver.calibrate();
    driver.analyzer.acknowledge(driver.t);
    expect(driver.analyzer.isAcknowledged(driver.t + 61_000)).toBe(false);
  });
});

describe('trained model verdicts', () => {
  const verdict = (drowsy: boolean) => ({ drowsy, probability: drowsy ? 0.9 : 0.1 });
  let driver: Driver;

  beforeEach(() => {
    driver = new Driver(new DrowsinessAnalyzer());
    driver.calibrate();
  });

  const feed = (drowsy: boolean, times: number) => {
    for (let i = 0; i < times; i++) driver.analyzer.setModelVerdict(verdict(drowsy));
  };

  it('warns after a sustained run of drowsy verdicts', () => {
    feed(true, 5);
    const reading = driver.read();
    expect(reading.state).toBe('drowsy');
    expect(reading.reason).toContain('ML model');
    expect(driver.analyzer.events.at(-1)?.confidence).toBeGreaterThan(0);
  });

  it('ignores a run that is too short', () => {
    feed(true, 4);
    expect(driver.read().state).toBe('alert');
  });

  it('needs the run to be unbroken', () => {
    feed(true, 4);
    feed(false, 1);
    feed(true, 4);
    expect(driver.read().state).toBe('alert');
  });

  it('forgets the run when the model has no verdict', () => {
    feed(true, 4);
    driver.analyzer.setModelVerdict(null);
    feed(true, 1);
    expect(driver.read().state).toBe('alert');
  });

  it('cannot raise critical on its own', () => {
    feed(true, 50);
    expect(driver.read().state).toBe('drowsy');
  });

  it('cannot talk the rules out of an alarm', () => {
    feed(false, 50);
    driver.run(700, SHUT);
    expect(driver.analyzer.currentState).toBe('critical');
  });

  it('recovers after the dwell time once the model calms down', () => {
    feed(true, 5);
    driver.read();
    feed(false, 1);
    driver.run(3000, OPEN);
    expect(driver.analyzer.currentState).toBe('alert');
  });
});
