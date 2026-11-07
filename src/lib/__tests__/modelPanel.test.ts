import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { ModelMetrics, ModelPredictionCard } from '../../components/ModelPanel';
import type { MlState } from '../../hooks/useMonitor';
import { IDLE_READING } from '../drowsiness';
import type { LoadedModel } from '../mlModel';

const missing: MlState = { status: 'missing', loaded: null, prediction: null, warmup: 0 };
const live = (ml: MlState, running = true) =>
  renderToStaticMarkup(createElement(ModelPredictionCard, { ml, reading: IDLE_READING, running }));

describe('without a trained model', () => {
  it('says so and shows no prediction', () => {
    const html = live(missing);
    expect(html).toContain('No trained model installed');
    expect(html).not.toContain('confidence');
    expect(html).not.toContain('P(drowsy)');
  });

  it('shows no evaluation numbers', () => {
    const html = renderToStaticMarkup(createElement(ModelMetrics, { ml: missing }));
    expect(html).toContain('No model has been trained yet');
    expect(html).not.toMatch(/\d%/);
    expect(html).not.toContain('Accuracy');
  });
});

describe('with a model', () => {
  const loaded = { model: { threshold: 0.4, trees: [{}], feature_names: ['a'] }, report: null } as unknown as LoadedModel;

  it('shows the classifier output when it has one', () => {
    const html = live({
      status: 'ready',
      loaded,
      warmup: 1,
      prediction: { label: 'drowsy', probability: 0.83, confidence: 0.83, latencyMs: 0.4 },
    });
    expect(html).toContain('drowsy');
    expect(html).toContain('83%');
    expect(html).toContain('decision at 0.40');
  });

  it('does not invent metrics when the metrics file is absent', () => {
    const html = renderToStaticMarkup(createElement(ModelMetrics, { ml: { ...missing, status: 'ready', loaded } }));
    expect(html).toContain('metrics file is missing');
  });
});
