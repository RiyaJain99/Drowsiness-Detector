import { Brain, ChartBar } from '@phosphor-icons/react';
import type { MlState } from '../hooks/useMonitor';
import { useCountUp } from '../hooks/useCountUp';
import { useInView } from '../hooks/useInView';
import type { Reading } from '../lib/drowsiness';
import type { MetricBlock, TrainingReport } from '../lib/mlModel';

const heading = 'text-body-md font-semibold uppercase tracking-wider text-outline m-0';
const KIND_NAME: Record<string, string> = { random_forest: 'Random Forest', xgboost: 'XGBoost' };

const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface-container/80 rounded-lg px-2.5 py-1.5 min-w-0">
      <div className="text-body-sm text-outline truncate">{label}</div>
      <div className="font-mono text-telemetry-val-md tabular-nums text-on-surface">{value}</div>
    </div>
  );
}

/** Ring that shows a real 0–1 value; `tick` marks the trained decision threshold. */
function Gauge({ value, tick, color, children }: { value: number; tick?: number; color: string; children: React.ReactNode }) {
  return (
    <div className="relative w-36 h-36 shrink-0 mx-auto">
      <svg viewBox="0 0 128 128" className="w-full h-full -rotate-90" aria-hidden="true">
        <circle cx="64" cy="64" r={RADIUS} fill="none" stroke="var(--color-surface-container-high)" strokeWidth="9" />
        <circle
          className="gauge-arc"
          cx="64"
          cy="64"
          r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={CIRCUMFERENCE * (1 - value)}
        />
        {tick !== undefined && (
          <line
            x1="64"
            y1="3"
            x2="64"
            y2="16"
            stroke="var(--color-on-surface)"
            strokeWidth="2"
            transform={`rotate(${tick * 360} 64 64)`}
          />
        )}
      </svg>
      <div className="absolute inset-0 grid place-content-center text-center">{children}</div>
    </div>
  );
}

/** Live verdict. Every number comes from the classifier or the analyzer; absent means absent. */
export function ModelPredictionCard({ ml, reading, running }: { ml: MlState; reading: Reading; running: boolean }) {
  const { prediction, loaded, status, warmup } = ml;
  const ready = reading.threshold !== null;
  const threshold = loaded?.model.threshold ?? 0.5;
  const drowsy = prediction?.label === 'drowsy';
  const tone = drowsy ? 'var(--color-secondary)' : 'var(--color-tertiary)';

  const waiting =
    status === 'loading'
      ? 'Loading model…'
      : status === 'missing'
        ? 'No trained model installed. Alerts are using the rule-based detector only.'
        : !running
          ? 'Start monitoring to see predictions.'
          : !ready
            ? 'Waiting for calibration.'
            : warmup < 1
              ? `Collecting behaviour for the first prediction (${Math.round(warmup * 100)}%).`
              : 'No prediction: face not visible for enough of the last minute.';

  return (
    <section className="panel p-4" aria-label="Machine learning prediction">
      <div className="flex items-center gap-1.5">
        <Brain size={14} weight="bold" className="text-tertiary" />
        <h2 className={heading}>ML prediction</h2>
        {prediction && (
          <span className="ml-auto text-telemetry-mono-sm font-mono text-outline">
            {prediction.latencyMs.toFixed(1)} ms
          </span>
        )}
      </div>

      {prediction ? (
        <div className="mt-3 flex items-center gap-4">
          <Gauge value={prediction.probability} tick={threshold} color={tone}>
            <span className="font-mono text-telemetry-val-lg uppercase" style={{ color: tone }}>
              {prediction.label}
            </span>
          </Gauge>
          <div className="min-w-0 flex-1">
            <div className="font-mono text-headline-xl tabular-nums text-on-surface">
              {(prediction.confidence * 100).toFixed(0)}%
            </div>
            <div className="text-body-sm text-outline">confidence</div>
            <div className="text-telemetry-mono-sm font-mono text-outline mt-2">
              <div>P(drowsy) {prediction.probability.toFixed(2)}</div>
              <div>decision at {threshold.toFixed(2)}</div>
            </div>
          </div>
        </div>
      ) : status === 'ready' && running && ready && warmup < 1 ? (
        <div className="mt-3 flex items-center gap-4">
          <Gauge value={warmup} color="var(--color-outline)">
            <span className="font-mono text-telemetry-val-md text-on-surface-variant">{Math.round(warmup * 100)}%</span>
          </Gauge>
          <p className="text-body-sm text-outline m-0">{waiting}</p>
        </div>
      ) : (
        <p className="text-body-sm text-outline mt-2 mb-0">{waiting}</p>
      )}

      <div className="grid grid-cols-2 gap-2 mt-3">
        <Tile label="Live EAR" value={ready ? reading.ear.toFixed(3) : '—'} />
        <Tile label="Live PERCLOS" value={ready ? `${reading.perclos.toFixed(1)}%` : '—'} />
      </div>
    </section>
  );
}

function Headline({ label, value, inView, digits, percent }: {
  label: string; value: number | null; inView: boolean; digits: number; percent?: boolean;
}) {
  const shown = useCountUp(value ?? 0, inView && value !== null);
  return (
    <div className="min-w-0">
      <div className="font-mono text-headline-xl tabular-nums" style={{ color: 'var(--state)' }}>
        {value === null ? '—' : percent ? `${(shown * 100).toFixed(digits)}%` : shown.toFixed(digits)}
      </div>
      <div className="text-body-sm text-outline">{label}</div>
    </div>
  );
}

const COMPARED = [
  ['Accuracy', 'accuracy'],
  ['Precision', 'precision'],
  ['Recall', 'recall'],
  ['F1', 'f1'],
  ['ROC-AUC', 'roc_auc'],
] as const;

/** Model against the app's existing rule, on the same windows. */
function Comparison({ model, rule, ruleName }: { model: MetricBlock; rule: MetricBlock; ruleName: string }) {
  const Bar = ({ value, color }: { value: number | null; color: string }) => (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 rounded-full bg-surface-container-high overflow-hidden">
        <div className="bar-fill h-full rounded-full" style={{ width: `${(value ?? 0) * 100}%`, background: color }} />
      </div>
      <span className="font-mono text-telemetry-mono-sm tabular-nums text-on-surface-variant w-12 text-right">
        {value === null ? '—' : value.toFixed(3)}
      </span>
    </div>
  );
  return (
    <div>
      <h3 className="text-body-sm font-semibold text-on-surface-variant mt-0 mb-1">Model vs existing rule</h3>
      <div className="flex gap-4 text-telemetry-mono-sm font-mono text-outline mb-2">
        <span className="inline-flex items-center gap-1.5"><i className="w-3 h-1.5 rounded-sm bg-tertiary" />Trained model</span>
        <span className="inline-flex items-center gap-1.5"><i className="w-3 h-1.5 rounded-sm bg-outline" />{ruleName}</span>
      </div>
      <div className="space-y-2">
        {COMPARED.map(([label, key]) => (
          <div key={key} className="grid grid-cols-[70px_1fr] gap-2 items-center">
            <span className="text-body-sm text-outline">{label}</span>
            <div className="space-y-1">
              <Bar value={model[key]} color="var(--color-tertiary)" />
              <Bar value={rule[key]} color="var(--color-outline)" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Confusion({ matrix }: { matrix: number[][] }) {
  const [[tn, fp], [fn, tp]] = matrix;
  const biggest = Math.max(tn, fp, fn, tp, 1);
  const Cell = ({ value, good, label }: { value: number; good: boolean; label: string }) => (
    <div
      className="rounded-lg px-2 py-2.5 text-center"
      style={{
        background: `color-mix(in srgb, var(${good ? '--color-tertiary' : '--color-primary-container'}) ${Math.round(8 + (value / biggest) * 42)}%, transparent)`,
      }}
    >
      <div className="font-mono text-telemetry-val-md tabular-nums text-on-surface">{value.toLocaleString()}</div>
      <div className="text-[11px] text-on-surface-variant">{label}</div>
    </div>
  );
  return (
    <div>
      <h3 className="text-body-sm font-semibold text-on-surface-variant mt-0 mb-1.5">Confusion matrix (test windows)</h3>
      <div className="grid grid-cols-2 gap-1.5">
        <Cell value={tn} good label="alert, called alert" />
        <Cell value={fp} good={false} label="alert, called drowsy" />
        <Cell value={fn} good={false} label="drowsy, called alert" />
        <Cell value={tp} good label="drowsy, called drowsy" />
      </div>
    </div>
  );
}

function Report({ report, trees, features }: { report: TrainingReport; trees: number; features: number }) {
  const { ref, inView } = useInView<HTMLDivElement>();
  const { test, cross_subject: cross, data } = report;
  const spread = (k: 'accuracy' | 'f1' | 'roc_auc') => {
    const { mean, std } = cross.window[k];
    return mean === null || std === null ? '—' : `${mean.toFixed(3)} ± ${std.toFixed(3)}`;
  };
  const strongest = report.feature_importance.slice(0, 6);
  const topImportance = strongest[0]?.importance || 1;

  return (
    <div ref={ref} data-inview={inView} className="mt-3 space-y-5">
      <div>
        <p className="text-body-sm text-outline m-0 mb-2">
          Held-out test subjects the model never saw, per window (n={test.window.n.toLocaleString()})
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <Headline label="Accuracy" value={test.window.accuracy} inView={inView} digits={1} percent />
          <Headline label="Precision" value={test.window.precision} inView={inView} digits={1} percent />
          <Headline label="Recall" value={test.window.recall} inView={inView} digits={1} percent />
          <Headline label="F1" value={test.window.f1} inView={inView} digits={1} percent />
          <Headline label="ROC-AUC" value={test.window.roc_auc} inView={inView} digits={3} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Comparison model={test.window} rule={test.rule_baseline} ruleName={test.rule_baseline.rule} />
        <Confusion matrix={test.window.confusion_matrix} />
        <div>
          <h3 className="text-body-sm font-semibold text-on-surface-variant mt-0 mb-1.5">Strongest features</h3>
          <div className="space-y-1.5">
            {strongest.map((f) => (
              <div key={f.feature}>
                <div className="flex justify-between text-telemetry-mono-sm font-mono text-on-surface-variant">
                  <span className="truncate">{f.feature}</span>
                  <span className="text-outline">{f.importance.toFixed(3)}</span>
                </div>
                <div className="h-1.5 rounded-full bg-surface-container-high overflow-hidden">
                  <div
                    className="bar-fill h-full rounded-full bg-tertiary"
                    style={{ width: `${(f.importance / topImportance) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <dl className="grid grid-cols-2 lg:grid-cols-4 gap-x-4 gap-y-2 m-0 text-body-sm pt-3 border-t border-outline-variant/20">
        {[
          ['Algorithm', `${KIND_NAME[report.model.kind] ?? report.model.kind} · ${trees} trees`],
          ['Trained', new Date(report.trained_at).toLocaleDateString()],
          ['Dataset', report.datasets.join(', ').toUpperCase()],
          ['Features', `${features} per 60 s window`],
          ['Subjects', `${data.train.subjects} train · ${data.val.subjects} val · ${data.test.subjects} test`],
          ['Windows', data.all.windows.toLocaleString()],
          ['Videos', String(data.all.videos)],
          [`Cross-subject, ${cross.folds.length} folds`, `AUC ${spread('roc_auc')} · F1 ${spread('f1')}`],
        ].map(([k, v]) => (
          <div key={k}>
            <dt className="text-outline">{k}</dt>
            <dd className="m-0 text-on-surface">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Model facts and evaluation, read from the metrics file the training run wrote. */
export function ModelMetrics({ ml }: { ml: MlState }) {
  const { loaded, status } = ml;
  return (
    <section className="panel p-4" aria-label="Machine learning model information">
      <div className="flex items-center gap-1.5">
        <ChartBar size={14} weight="bold" className="text-tertiary" />
        <h2 className={heading}>ML model</h2>
      </div>
      {loaded?.report ? (
        <Report report={loaded.report} trees={loaded.model.trees.length} features={loaded.model.feature_names.length} />
      ) : (
        <p className="text-body-sm text-outline mt-2 mb-0">
          {status === 'loading'
            ? 'Loading…'
            : status === 'ready'
              ? 'Model loaded, but its metrics file is missing, so no evaluation results can be shown.'
              : 'No model has been trained yet, so there are no results to show. See ml/README.md.'}
        </p>
      )}
    </section>
  );
}
