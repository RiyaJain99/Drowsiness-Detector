import { Eye, Lightning, MaskHappy, WarningOctagon, type Icon } from '@phosphor-icons/react';
import type { Reading } from '../lib/drowsiness';

type Tone = 'normal' | 'warn' | 'crit';

interface Cell {
  label: string;
  icon: Icon;
  value: string;
  status: string | null;
  context: string;
  contextValue: string;
  tone: Tone;
}

const TONE_TEXT: Record<Tone, string> = {
  normal: 'text-on-surface',
  warn: 'text-secondary',
  crit: 'text-primary-container',
};

const TONE_PILL: Record<Tone, string> = {
  normal: 'bg-tertiary/15 text-tertiary',
  warn: 'bg-secondary/15 text-secondary',
  crit: 'bg-primary-container/20 text-primary-container',
};

export function TelemetryGrid({ reading }: { reading: Reading }) {
  const ready = reading.threshold !== null;
  const lowBlinkRate = ready && reading.blinksPerMinute > 0 && reading.blinksPerMinute < 12;

  const cells: Cell[] = [
    {
      label: 'Eye aspect ratio',
      icon: Eye,
      value: ready ? reading.ear.toFixed(3) : '—',
      status: ready && reading.closed ? 'Below threshold' : null,
      context: 'Threshold',
      contextValue: ready ? reading.threshold!.toFixed(3) : 'calibrating',
      tone: ready && reading.closed ? 'crit' : 'normal',
    },
    {
      label: 'Blinks / min',
      icon: Lightning,
      // A falling blink rate is itself a fatigue signal, so the normal band is
      // shown rather than leaving the reader to guess whether 6 is bad.
      value: String(reading.blinksPerMinute),
      status: lowBlinkRate ? 'Below normal' : null,
      context: 'Normal',
      contextValue: '12–24',
      tone: lowBlinkRate ? 'warn' : 'normal',
    },
    {
      label: 'Microsleeps',
      icon: WarningOctagon,
      value: String(reading.microsleeps),
      status: reading.microsleeps > 0 ? 'Detected' : null,
      context: 'Closures',
      contextValue: '>0.5s',
      tone: reading.microsleeps > 0 ? 'crit' : 'normal',
    },
    {
      label: 'Yawns',
      icon: MaskHappy,
      value: String(reading.yawns),
      status: reading.yawns >= 2 ? 'Fatigue elevated' : null,
      context: 'Head nods',
      contextValue: String(reading.nods),
      tone: reading.yawns >= 2 ? 'warn' : 'normal',
    },
  ];

  return (
    <section className="grid grid-cols-2 gap-2.5">
      {cells.map((cell) => (
        <div
          key={cell.label}
          className="bg-surface-container-low p-3 rounded-xl border border-outline-variant/30 flex flex-col shadow-sm"
        >
          <div className="flex items-center justify-between gap-1">
            <span className="text-body-sm text-outline">{cell.label}</span>
            <cell.icon size={14} weight="bold" className={TONE_TEXT[cell.tone]} />
          </div>
          <div className="my-1.5 flex items-center gap-1.5 flex-wrap">
            <span className={`font-mono text-telemetry-val-lg tracking-tight tabular-nums ${TONE_TEXT[cell.tone]}`}>
              {cell.value}
            </span>
            {cell.status && (
              <span className={`px-1.5 py-0.5 rounded font-label-caps text-[10px] font-bold ${TONE_PILL[cell.tone]}`}>
                {cell.status}
              </span>
            )}
          </div>
          <div className="text-telemetry-mono-sm text-outline font-mono flex items-center justify-between pt-1 border-t border-outline-variant/20">
            <span>{cell.context}</span>
            <span className="text-on-surface-variant">{cell.contextValue}</span>
          </div>
        </div>
      ))}
    </section>
  );
}
