import { ClockCounterClockwise } from '@phosphor-icons/react';
import type { MonitorEvent, Severity } from '../lib/drowsiness';
import { stamp } from '../lib/format';

const DOT: Record<Severity, string> = {
  ok: 'bg-tertiary',
  warn: 'bg-secondary',
  crit: 'bg-primary-container',
};

/** System bookkeeping events read as neutral rather than a driving-state verdict. */
const SYSTEM_LABELS = new Set(['Calibrated', 'Acknowledged']);

function badge(e: MonitorEvent): { text: string; className: string } {
  if (e.severity === 'crit') return { text: 'CRITICAL', className: 'bg-primary-container/20 text-primary-container' };
  if (e.severity === 'warn') return { text: 'ADVISORY', className: 'bg-secondary/15 text-secondary' };
  if (SYSTEM_LABELS.has(e.label)) return { text: 'SYSTEM', className: 'bg-surface-container-high text-outline' };
  return { text: 'NORMAL', className: 'bg-tertiary/15 text-tertiary' };
}

export function EventLog({ events }: { events: readonly MonitorEvent[] }) {
  const newestFirst = [...events].reverse().slice(0, 40);
  const critical = events.filter((e) => e.severity === 'crit').length;

  return (
    <section className="bg-surface-container-low rounded-2xl p-3.5 border border-outline-variant/30">
      <div className="flex items-center justify-between mb-2.5">
        <h2 className="text-body-md font-semibold text-on-surface flex items-center gap-1.5 m-0">
          <ClockCounterClockwise size={14} weight="bold" className="text-outline" />
          Incident timeline
        </h2>
        <span className="text-telemetry-mono-sm text-outline font-mono">
          {events.length} recorded{critical > 0 && ` · ${critical} critical`}
        </span>
      </div>

      {newestFirst.length === 0 ? (
        <p className="text-outline text-body-sm font-mono m-0 py-2">
          Events appear here once monitoring starts.
        </p>
      ) : (
        <ol className="space-y-1.5 text-body-sm font-mono list-none m-0 p-0">
          {newestFirst.map((e, i) => {
            const { text, className } = badge(e);
            return (
              <li
                key={`${e.at}-${i}`}
                className="flex items-center justify-between gap-2 py-1.5 px-2 rounded-lg bg-surface-container/70 border border-outline-variant/20"
              >
                <span className="flex items-center gap-2 min-w-0">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${DOT[e.severity]}`} />
                  <span className="text-on-surface-variant truncate">
                    <span className="text-on-surface">{e.label}</span> — {e.detail}
                  </span>
                </span>
                <span className="flex items-center gap-2 shrink-0">
                  <span className={`px-1.5 py-0.5 rounded font-label-caps text-[9px] font-bold ${className}`}>
                    {text}
                  </span>
                  <time className="text-outline text-telemetry-mono-sm tabular-nums">{stamp(e.at)}</time>
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
