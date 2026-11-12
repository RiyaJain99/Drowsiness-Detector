import { describe, expect, it } from 'vitest';
import { classify, evaluateModel, parseModel, type TrainedModel } from '../mlModel';

// A hand-written two-leaf stump: tests the walker, is not a trained model.
const stump = (over: Partial<TrainedModel>): TrainedModel => ({
  schema: 1,
  kind: 'random_forest',
  trained_at: '',
  threshold: 0.5,
  feature_names: ['a'],
  feature_config: {} as TrainedModel['feature_config'],
  combine: 'mean',
  strict: false,
  bias: 0,
  trees: [{ f: [0, -2, -2], t: [1, 0, 0], l: [1, -1, -1], r: [2, -1, -1], v: [0, 0.2, 0.8] }],
  ...over,
});

describe('evaluateModel', () => {
  it('branches like scikit-learn at the threshold', () => {
    expect(evaluateModel(stump({}), [1])).toBeCloseTo(0.2);
  });

  it('branches like xgboost at the threshold', () => {
    expect(evaluateModel(stump({ strict: true }), [1])).toBeCloseTo(0.8);
  });

  it('applies the logistic link with its intercept', () => {
    const model = stump({ combine: 'logistic', bias: -0.2 });
    expect(evaluateModel(model, [0])).toBeCloseTo(0.5);
  });

  it('averages trees in a forest', () => {
    const tree = stump({}).trees[0];
    const model = stump({ trees: [tree, { ...tree, v: [0, 0.6, 0.8] }] });
    expect(evaluateModel(model, [0])).toBeCloseTo(0.4);
  });
});

describe('linear models', () => {
  it('applies weights and bias through the logistic link', () => {
    const model = stump({ kind: 'logistic_regression', combine: 'linear', trees: [], weights: [2], bias: -1 });
    expect(evaluateModel(model, [0.5])).toBeCloseTo(0.5);
    expect(evaluateModel(model, [1])).toBeCloseTo(1 / (1 + Math.exp(-1)));
  });

  it('rejects a linear model without one weight per feature', () => {
    expect(parseModel({ ...stump({}), combine: 'linear', trees: [], weights: [] })).toBeNull();
  });
});

describe('classify', () => {
  it('reports confidence in the class it chose', () => {
    expect(classify(stump({}), [2])).toMatchObject({ label: 'drowsy', confidence: 0.8 });
    expect(classify(stump({}), [0])).toMatchObject({ label: 'alert', confidence: 0.8 });
  });

  it('honours the threshold chosen in training', () => {
    expect(classify(stump({ threshold: 0.9 }), [2]).label).toBe('alert');
  });
});

describe('parseModel', () => {
  it('rejects anything incomplete', () => {
    expect(parseModel(null)).toBeNull();
    expect(parseModel({ schema: 1, trees: [] })).toBeNull();
    expect(parseModel({ ...stump({}), schema: 2 })).toBeNull();
    expect(parseModel(stump({}), 5)).toBeNull();
  });
});

// Runs once training has produced a model; replays real test rows through this walker.
const exported = import.meta.glob<unknown>('/public/ml/*.json', { eager: true, import: 'default' });
const trained = exported['/public/ml/model.json'];
const check = exported['/public/ml/model-check.json'] as { rows: number[][]; expected: number[] } | undefined;

describe.runIf(trained && check)('trained model', () => {
  it('matches the probabilities from Python', () => {
    const model = parseModel(trained)!;
    check!.rows.forEach((row, i) => {
      expect(evaluateModel(model, row)).toBeCloseTo(check!.expected[i], 4);
    });
  });
});
