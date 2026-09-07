import { Cpu, LockSimple, Microphone, ShieldCheck, WarningOctagon } from '@phosphor-icons/react';
import type { DriverState } from '../lib/drowsiness';
import { clock } from '../lib/format';

export const STATE_COPY: Record<DriverState, string> = {
  idle: 'Camera off',
  calibrating: 'Calibrating',
  alert: 'Alert',
  drowsy: 'Drowsy',
  critical: 'Pull over',
};

interface TopBarProps {
  state: DriverState;
  reason: string;
  elapsedMs: number;
  running: boolean;
  listening: boolean;
}

export function TopBar({ state, reason, elapsedMs, running, listening }: TopBarProps) {
  const critical = state === 'critical';

  return (
    <header className="sticky top-0 z-40 bg-surface-container-lowest/90 backdrop-blur-xl border-b border-outline-variant/30">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 pt-3 pb-2.5 flex flex-col gap-2">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 min-w-0">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border"
              style={{ color: 'var(--state)', borderColor: 'var(--state)', background: 'var(--state-soft)' }}
            >
              <ShieldCheck size={18} weight="bold" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="font-display text-headline-md font-bold tracking-tight leading-none text-on-surface">
                  DROWSY<span style={{ color: 'var(--state)' }}>GUARD</span>
                </span>
                <span className="hidden sm:inline-block px-1.5 py-0.5 rounded border border-primary/30 bg-surface-container text-primary font-label-caps text-label-caps">
                  AI HUD
                </span>
              </div>
              <p className="text-body-sm text-outline flex items-center gap-1 leading-tight mt-0.5">
                <Cpu size={11} weight="fill" className="text-tertiary shrink-0" />
                On-device engine{running ? ': active' : ''}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-4 shrink-0">
            <div
              className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border text-body-sm font-semibold tracking-tight"
              style={{ color: 'var(--state)', borderColor: 'var(--state)', background: 'var(--state-soft)' }}
            >
              <span className={`w-1.5 h-1.5 rounded-full bg-current ${critical ? 'animate-pulse' : ''}`} />
              {STATE_COPY[state]}
            </div>
            <span className="font-mono text-telemetry-mono-sm text-outline tabular-nums tracking-wider hidden sm:inline">
              {clock(running ? elapsedMs : 0)}
            </span>
            {listening && (
              <div
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-tertiary/30 bg-tertiary/10 text-body-sm text-tertiary"
                title="Say “Riri” to ask your status, or to silence an alarm"
              >
                <Microphone size={12} weight="fill" className="animate-pulse" />
                <span className="hidden sm:inline">Say “Riri”</span>
              </div>
            )}
            <div className="hidden lg:flex items-center gap-1.5 bg-surface-container/60 px-2.5 py-1.5 rounded-md border border-outline-variant/30 text-body-sm text-outline">
              <LockSimple size={12} weight="fill" className="text-tertiary" />
              Video never leaves this device
            </div>
          </div>
        </div>

        {critical && (
          <div className="w-full rounded-lg bg-error-container/40 border border-primary-container/40 px-3 py-1.5 flex items-center gap-2 overflow-hidden">
            <WarningOctagon size={14} weight="bold" className="text-primary-container shrink-0 animate-pulse" />
            <span className="font-label-caps text-label-caps font-bold tracking-wider text-primary-container truncate">
              CRITICAL — {reason}
            </span>
          </div>
        )}
      </div>
    </header>
  );
}
