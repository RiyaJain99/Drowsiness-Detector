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
    expect(driver.analyzer.events.at(-1)?.detail).toBe('no face in frame');
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
