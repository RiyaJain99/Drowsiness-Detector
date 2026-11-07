import { ArrowRight, Brain, Cpu, LockSimple, Scan } from '@phosphor-icons/react';
import { Fragment, type CSSProperties } from 'react';
import type { MlState } from '../hooks/useMonitor';

const KIND_NAME: Record<string, string> = { random_forest: 'Random Forest', xgboost: 'XGBoost' };

/** The six EAR landmarks, in the order the formula names them. */
const LANDMARKS: [string, number, number][] = [
  ['p1', 34, 100],
  ['p2', 76, 66],
  ['p3', 124, 66],
  ['p4', 166, 100],
  ['p5', 124, 134],
  ['p6', 76, 134],
];

/**
 * The algorithm as a logo: the points sit on the lids, so when it blinks you can
 * watch the vertical distances collapse, which is exactly what EAR measures.
 */
function EyeDiagram() {
  return (
    <figure className="m-0 w-full max-w-[300px] mx-auto">
      <svg viewBox="0 0 200 200" className="w-full block" role="img" aria-label="Eye with the six landmarks used for eye aspect ratio">
        <g fill="none" stroke="var(--state)">
          <circle className="orbit" cx="100" cy="100" r="94" strokeOpacity=".35" strokeDasharray="2 8" />
          <circle className="orbit reverse" cx="100" cy="100" r="82" strokeOpacity=".25" strokeDasharray="40 18" />
        </g>
        <g className="eye-lids">
          <path
            d="M34 100 Q100 36 166 100 Q100 164 34 100 Z"
            fill="var(--state-soft)"
            stroke="var(--state)"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <circle cx="100" cy="100" r="25" fill="none" stroke="var(--state)" strokeWidth="2" />
          <circle cx="100" cy="100" r="11" fill="var(--state)" />
          <circle cx="107" cy="93" r="3.5" fill="var(--color-on-surface)" />
          <g stroke="var(--color-on-surface)" strokeOpacity=".5" strokeDasharray="3 3">
            <line x1="76" y1="66" x2="76" y2="134" />
            <line x1="124" y1="66" x2="124" y2="134" />
            <line x1="34" y1="100" x2="166" y2="100" />
          </g>
          {LANDMARKS.map(([name, x, y]) => (
            <g key={name}>
              <circle cx={x} cy={y} r="3.5" fill="var(--color-on-surface)" />
              <text
                x={x}
                y={y + (y < 100 ? -8 : y > 100 ? 15 : 0)}
                dx={y === 100 ? (x < 100 ? -9 : 9) : 0}
                dy={y === 100 ? 3 : 0}
                textAnchor="middle"
                fontSize="8"
                fontFamily="var(--font-mono)"
                fill="var(--color-on-surface-variant)"
              >
                {name}
              </text>
            </g>
          ))}
        </g>
      </svg>
      <figcaption className="text-center font-mono text-telemetry-mono-sm text-outline mt-1">
        EAR = (|p2−p6| + |p3−p5|) / 2|p1−p4|
      </figcaption>
    </figure>
  );
}

function Pipeline({ ml }: { ml: MlState }) {
  const trained = ml.loaded?.model;
  const stages = [
    ['Webcam', 'stays on device'],
    ['MediaPipe', '478 landmarks'],
    ['Features', '36 per 60 s window'],
    [trained ? (KIND_NAME[trained.kind] ?? trained.kind) : 'Classifier', trained ? `${trained.trees.length} trees` : 'not trained yet'],
    ['Warning', 'rules + model'],
  ];
  return (
    <ol className="list-none m-0 p-0 flex flex-col sm:flex-row sm:items-center gap-1.5 sm:gap-2" aria-label="Processing pipeline">
      {stages.map(([name, detail], i) => (
        <Fragment key={name}>
          <li className="rise panel px-3 py-2 min-w-0 sm:flex-[3]" style={{ '--i': 5 + i } as CSSProperties}>
            <div className="text-body-sm font-semibold text-on-surface truncate">{name}</div>
            <div className="text-telemetry-mono-sm font-mono text-outline truncate">{detail}</div>
          </li>
          {i < stages.length - 1 && (
            <span className="flow-line hidden sm:block max-w-10" style={{ '--i': i } as CSSProperties} aria-hidden="true" />
          )}
        </Fragment>
      ))}
    </ol>
  );
}

interface HeroProps {
  ml: MlState;
  loading: boolean;
  error: string | null;
  onStart: () => void;
}

export function Hero({ ml, loading, error, onStart }: HeroProps) {
  const report = ml.loaded?.report;
  const facts = [
    { icon: Scan, text: '478 landmarks, every frame' },
    { icon: LockSimple, text: 'No video leaves this device' },
    { icon: Cpu, text: 'No server' },
    {
      icon: Brain,
      text: report
        ? `Trained on ${report.data.all.subjects} real subjects`
        : ml.status === 'loading'
          ? 'Checking for a trained model'
          : 'Rule-based until a model is trained',
    },
  ];

  return (
    <section className="py-4 sm:py-8 flex flex-col gap-6" aria-labelledby="hero-title">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-10 items-center">
        <div className="lg:col-span-8 flex flex-col gap-5">
          <p className="rise m-0 font-mono text-label-caps uppercase text-tertiary" style={{ '--i': 0 } as CSSProperties}>
            Driver drowsiness detection
          </p>
          <h1
            id="hero-title"
            className="rise gradient-title m-0 font-display font-bold tracking-tight text-[40px] leading-[46px] sm:text-[60px] sm:leading-[66px]"
            style={{ '--i': 1 } as CSSProperties}
          >
            It sees the blink that lasts too long.
          </h1>
          <p className="rise m-0 max-w-[58ch] text-body-lg text-on-surface-variant" style={{ '--i': 2 } as CSSProperties}>
            DrowsyGuard reads eye closure, yawning and head movement from your webcam, then
            warns you before a tired moment becomes a crash. All of it runs in your browser.
          </p>
          <div className="rise flex flex-wrap items-center gap-3" style={{ '--i': 3 } as CSSProperties}>
            <button
              type="button"
              onClick={onStart}
              disabled={loading}
              className="sheen inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-tertiary text-on-tertiary font-bold text-body-md active:scale-[0.98] transition hover:brightness-110 disabled:opacity-60"
            >
              {loading ? 'Loading model…' : 'Start monitoring'}
              <ArrowRight size={18} weight="bold" />
            </button>
            <span className="text-body-sm text-outline">Needs camera access. Nothing is recorded.</span>
          </div>
          {error && <p role="alert" className="m-0 text-body-sm text-primary">{error}</p>}
          <ul className="rise list-none m-0 p-0 flex flex-wrap gap-x-5 gap-y-2" style={{ '--i': 4 } as CSSProperties}>
            {facts.map(({ icon: Icon, text }) => (
              <li key={text} className="inline-flex items-center gap-1.5 text-body-sm text-on-surface-variant">
                <Icon size={14} weight="bold" className="text-tertiary shrink-0" />
                {text}
              </li>
            ))}
          </ul>
        </div>
        <div className="rise lg:col-span-4" style={{ '--i': 3 } as CSSProperties}>
          <EyeDiagram />
        </div>
      </div>
      <Pipeline ml={ml} />
    </section>
  );
}
