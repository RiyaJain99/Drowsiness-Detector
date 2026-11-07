import { describe, expect, it } from 'vitest';
import {
  FeatureWindow,
  NO_FACE,
  calibrate,
  measureForModel,
  windowFeatures,
  type FeatureConfig,
  type MlSample,
} from '../mlFeatures';
import fixture from './fixtures/feature-parity.json';

const cfg = fixture.feature_config as FeatureConfig;

function samples(series: Record<string, (number | null)[]>): MlSample[] {
  const n = (v: number | null) => v ?? NaN;
  return series.t_ms.map((t, i) => ({
    t: t!,
    ear: n(series.ear[i]),
    earL: n(series.ear_l[i]),
    earR: n(series.ear_r[i]),
    mar: n(series.mar[i]),
    pitch: n(series.pitch[i]),
    yaw: n(series.yaw[i]),
    roll: n(series.roll[i]),
  }));
}

describe('parity with ml/drowsyguard_ml/features.py', () => {
  for (const c of fixture.cases) {
    it(`reproduces every Python feature: ${c.name}`, () => {
      const all = samples(c.series);
      const cal = calibrate(all, cfg)!;
      for (const key of ['ear', 'mar', 'pitch', 'yaw', 'roll'] as const) {
        expect(cal[key]).toBeCloseTo(c.calibration[key], 9);
      }
      const last = all[all.length - 1].t;
      const window = all.filter((s) => s.t > last - cfg.window_s * 1000);
      const features = windowFeatures(window, cal, cfg)!;
      expect(features).toHaveLength(fixture.feature_names.length);
      features.forEach((value, i) => {
        expect(value, fixture.feature_names[i]).toBeCloseTo(c.features![i], 7);
      });
    });
  }

  it('reproduces the per-frame measurements', () => {
    const { points, aspect, measured } = fixture.landmarks;
    const m = measureForModel(points.map(([x, y, z]) => ({ x, y, z })), aspect);
    expect(m.ear).toBeCloseTo(measured.ear, 10);
    expect(m.earL).toBeCloseTo(measured.ear_l, 10);
    expect(m.earR).toBeCloseTo(measured.ear_r, 10);
    expect(m.mar).toBeCloseTo(measured.mar, 10);
    expect(m.pitch).toBeCloseTo(measured.pitch, 8);
    expect(m.yaw).toBeCloseTo(measured.yaw, 8);
    expect(m.roll).toBeCloseTo(measured.roll, 8);
  });
});

describe('FeatureWindow', () => {
  const open = { ear: 0.3, earL: 0.3, earR: 0.3, mar: 0.05, pitch: 0, yaw: 0, roll: 0 };

  it('thins a 30 fps stream to the training rate', () => {
    const w = new FeatureWindow(cfg);
    for (let i = 0; i < 300; i++) w.push((i * 1000) / 30, open);
    expect(w.heldSeconds).toBeGreaterThan(9.8);
    expect(w.calibration?.ear).toBeCloseTo(0.3);
  });

  it('gives no features before the minimum window', () => {
    const w = new FeatureWindow(cfg);
    for (let i = 0; i < 20 * 30; i++) w.push((i * 1000) / 30, open);
    expect(w.features()).toBeNull();
    for (let i = 20 * 30; i < 31 * 30; i++) w.push((i * 1000) / 30, open);
    expect(w.features()).not.toBeNull();
  });

  it('never holds more than the window', () => {
    const w = new FeatureWindow(cfg);
    for (let i = 0; i < 120 * 30; i++) w.push((i * 1000) / 30, open);
    expect(w.heldSeconds).toBeLessThanOrEqual(cfg.window_s);
  });

  it('refuses to predict when the face has mostly been gone', () => {
    const w = new FeatureWindow(cfg);
    for (let i = 0; i < 10 * 30; i++) w.push((i * 1000) / 30, open);
    for (let i = 10 * 30; i < 70 * 30; i++) w.push((i * 1000) / 30, NO_FACE);
    expect(w.features()).toBeNull();
  });

  it('does not calibrate on closed eyes', () => {
    const w = new FeatureWindow(cfg);
    for (let i = 0; i < 5 * 30; i++) w.push((i * 1000) / 30, { ...open, ear: 0.05 });
    expect(w.calibration).toBeNull();
  });
});
