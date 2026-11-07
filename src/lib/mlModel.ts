/**
 * Evaluates the trained tree ensemble exported by `ml/drowsyguard_ml/export.py`.
 *
 * The model is data, not code: `public/ml/model.json` exists only after a real
 * training run. When it is absent there is simply no prediction — nothing here
 * ever produces a placeholder value.
 */
import type { FeatureConfig } from './mlFeatures';

interface Tree {
  /** Feature index per node; negative marks a leaf. */
  f: number[];
  t: number[];
  l: number[];
  r: number[];
  v: number[];
}

export interface TrainedModel {
  schema: number;
  kind: 'random_forest' | 'xgboost';
  trained_at: string;
  threshold: number;
  feature_names: string[];
  feature_config: FeatureConfig;
  /** 'mean' averages leaf probabilities (forest); 'logistic' sums margins (boosting). */
  combine: 'mean' | 'logistic';
  /** xgboost branches on x < t; scikit-learn on x <= t. */
  strict: boolean;
  bias: number;
  trees: Tree[];
}

export interface ModelPrediction {
  label: 'alert' | 'drowsy';
  /** Classifier's P(drowsy), 0–1. */
  probability: number;
  /** Probability of the class it chose, 0–1. */
  confidence: number;
  /** Time spent computing features and walking the trees, ms. */
  latencyMs: number;
}

const SCHEMA = 1;

function walk(tree: Tree, x: Float32Array, strict: boolean): number {
  let i = 0;
  while (tree.f[i] >= 0) {
    const value = x[tree.f[i]];
    const left = strict ? value < tree.t[i] : value <= tree.t[i];
    i = left ? tree.l[i] : tree.r[i];
  }
  return tree.v[i];
}

/** P(drowsy). Inputs go through float32 because both training libraries use it. */
export function evaluateModel(model: TrainedModel, features: readonly number[]): number {
  const x = Float32Array.from(features);
  let raw = 0;
  for (const tree of model.trees) raw += walk(tree, x, model.strict);
  return model.combine === 'mean' ? raw / model.trees.length : 1 / (1 + Math.exp(-(raw + model.bias)));
}

export function classify(model: TrainedModel, features: readonly number[], latencyMs = 0): ModelPrediction {
  const probability = evaluateModel(model, features);
  const drowsy = probability >= model.threshold;
  return {
    label: drowsy ? 'drowsy' : 'alert',
    probability,
    confidence: drowsy ? probability : 1 - probability,
    latencyMs,
  };
}

/** Rejects anything that is not a complete model, so a bad file cannot yield predictions. */
export function parseModel(raw: unknown, expectedFeatures?: number): TrainedModel | null {
  const m = raw as Partial<TrainedModel> | null;
  if (!m || m.schema !== SCHEMA || !Array.isArray(m.trees) || m.trees.length === 0) return null;
  if (!Array.isArray(m.feature_names) || !m.feature_config || typeof m.threshold !== 'number') return null;
  if (expectedFeatures !== undefined && m.feature_names.length !== expectedFeatures) return null;
  return m as TrainedModel;
}

export interface MetricBlock {
  n: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  confusion_matrix: number[][];
}

interface MeanStd {
  mean: number | null;
  std: number | null;
}

interface DataSummary {
  subjects: number;
  videos: number;
  windows: number;
}

/** Subset of `metrics.json` the dashboard shows; written by the training run. */
export interface TrainingReport {
  trained_at: string;
  datasets: string[];
  data: { train: DataSummary; val: DataSummary; test: DataSummary; all: DataSummary };
  model: { kind: string; threshold: number };
  candidates: { kind: string; val: { roc_auc: number | null; f1: number } }[];
  test: { window: MetricBlock; video: MetricBlock; rule_baseline: MetricBlock & { rule: string } };
  cross_subject: {
    folds: unknown[];
    window: Record<'accuracy' | 'precision' | 'recall' | 'f1' | 'roc_auc', MeanStd>;
  };
  feature_importance: { feature: string; importance: number }[];
}

export interface LoadedModel {
  model: TrainedModel;
  report: TrainingReport | null;
}

async function fetchJson(path: string): Promise<unknown> {
  const response = await fetch(path);
  // Vite's dev server answers unknown paths with index.html, hence the type check.
  if (!response.ok || !response.headers.get('content-type')?.includes('json')) return null;
  return response.json().catch(() => null);
}

export async function loadModel(): Promise<LoadedModel | null> {
  const model = parseModel(await fetchJson('/ml/model.json'));
  if (!model) return null;
  return { model, report: (await fetchJson('/ml/metrics.json')) as TrainingReport | null };
}
