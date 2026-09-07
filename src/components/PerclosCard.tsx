import { EyeSlash } from '@phosphor-icons/react';
import { DEFAULT_CONFIG } from '../lib/config';

const CEILING = 40;

/**
 * The hero metric. PERCLOS is what fatigue research validates against driving
 * impairment, so it gets the largest type on screen and both thresholds are
 * marked on the bar rather than left implicit in a colour change.
 */
export function PerclosCard({ perclos, ready }: { perclos: number; ready: boolean }) {
  const { drowsyPerclos, criticalPerclos } = DEFAULT_CONFIG;
  const verdict = !ready
    ? 'Awaiting baseline'
    : perclos >= criticalPerclos
      ? 'Critical'
      : perclos >= drowsyPerclos
        ? 'Above drowsy limit'
        : 'Within normal range';

  return (
    <section className="bg-surface-container-low rounded-2xl p-4 border border-outline-variant/30 relative overflow-hidden">
      <div
        className="absolute -right-8 -top-8 w-32 h-32 rounded-full blur-2xl pointer-events-none"
        style={{ background: 'var(--state-soft)' }}
      />
      <div className="flex items-start justify-between gap-3 relative">
        <div>
          <div className="flex items-center gap-1.5">
            <EyeSlash size={14} weight="bold" className="text-primary" />
            <h2 className="text-body-md font-semibold uppercase tracking-wider text-outline m-0">PERCLOS</h2>
            <span className="text-telemetry-mono-sm bg-surface-container border border-outline-variant/30 text-outline px-1.5 py-px rounded font-mono">
              last 60s
            </span>
          </div>
          <p className="text-body-sm text-outline mt-0.5 m-0">Proportion of time with eyes closed</p>
        </div>
        <span
          className="text-telemetry-mono-sm font-semibold px-2 py-1 rounded-full border shrink-0"
          style={{ color: 'var(--state)', borderColor: 'var(--state)', background: 'var(--state-soft)' }}
        >
          {verdict}
        </span>
      </div>

      <div className="mt-3 flex items-baseline gap-1.5 relative">
        <span
          className="font-mono text-display-hero tracking-tight tabular-nums"
          style={{ color: 'var(--state)' }}
        >
          {ready ? perclos.toFixed(1) : '—'}
        </span>
        <span className="font-mono text-headline-lg font-semibold" style={{ color: 'var(--state)' }}>
          %
        </span>
      </div>

      <div className="mt-2.5 relative">
        <div className="w-full h-2.5 bg-surface-container rounded-full overflow-hidden relative">
          <div
            className="h-full rounded-full transition-[width] duration-300"
            style={{ width: `${Math.min(100, (perclos / CEILING) * 100)}%`, background: 'var(--state)' }}
          />
          {/* Both thresholds marked, so the number has a scale to sit against. */}
          <span
            className="absolute inset-y-0 w-px bg-secondary"
            style={{ left: `${(drowsyPerclos / CEILING) * 100}%` }}
          />
          <span
            className="absolute inset-y-0 w-px bg-primary-container"
            style={{ left: `${(criticalPerclos / CEILING) * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-telemetry-mono-sm font-mono mt-1.5 text-outline">
          <span>0%</span>
          <span className="text-secondary">{drowsyPerclos}% drowsy</span>
          <span className="text-primary-container">{criticalPerclos}% critical</span>
        </div>
      </div>
    </section>
  );
}
