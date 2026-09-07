import {
  ArrowsClockwise, DownloadSimple, Microphone, MicrophoneSlash, Play, SpeakerHigh, SpeakerSlash, Stop,
} from '@phosphor-icons/react';

interface ControlDeckProps {
  running: boolean;
  loading: boolean;
  soundOn: boolean;
  voiceOn: boolean;
  voiceSupported: boolean;
  canExport: boolean;
  onToggle: () => void;
  onRecalibrate: () => void;
  onToggleSound: () => void;
  onToggleVoice: () => void;
  onExport: () => void;
}

const secondary =
  'flex items-center justify-center gap-1.5 py-2.5 px-2 rounded-xl border border-outline-variant/40 bg-surface-container ' +
  'text-on-surface-variant text-body-sm font-medium active:scale-95 transition disabled:opacity-40 disabled:active:scale-100';

export function ControlDeck({
  running, loading, soundOn, voiceOn, voiceSupported, canExport,
  onToggle, onRecalibrate, onToggleSound, onToggleVoice, onExport,
}: ControlDeckProps) {
  return (
    <nav className="bg-surface-container-low/80 backdrop-blur-xl border border-outline-variant/30 rounded-2xl px-4 pt-3 pb-3 flex flex-col gap-2.5">
      <div className="grid grid-cols-2 gap-2">
        <button onClick={onRecalibrate} disabled={!running} className={secondary} type="button">
          <ArrowsClockwise size={14} weight="bold" />
          Recalibrate
        </button>
        <button onClick={onExport} disabled={!canExport} className={secondary} type="button">
          <DownloadSimple size={14} weight="bold" />
          Export
        </button>
        <button
          onClick={onToggleSound}
          aria-pressed={soundOn}
          className={`${secondary} ${soundOn ? 'text-tertiary border-tertiary/40' : ''}`}
          type="button"
        >
          {soundOn ? <SpeakerHigh size={14} weight="bold" /> : <SpeakerSlash size={14} weight="bold" />}
          {soundOn ? 'Alarm on' : 'Muted'}
        </button>
        {voiceSupported ? (
          <button
            onClick={onToggleVoice}
            aria-pressed={voiceOn}
            className={`${secondary} ${voiceOn ? 'text-tertiary border-tertiary/40' : ''}`}
            type="button"
          >
            {voiceOn ? <Microphone size={14} weight="bold" /> : <MicrophoneSlash size={14} weight="bold" />}
            {voiceOn ? 'Voice on' : 'Voice off'}
          </button>
        ) : (
          <span className={`${secondary} opacity-40`}>
            <MicrophoneSlash size={14} weight="bold" />
            No voice
          </span>
        )}
      </div>

      <button
        onClick={onToggle}
        disabled={loading}
        className={`w-full py-3.5 px-4 rounded-xl font-bold flex items-center justify-center gap-2 text-body-sm uppercase tracking-wide active:scale-[0.98] transition disabled:opacity-60 ${
          running
            ? 'bg-surface-container border border-outline-variant/40 text-on-surface-variant'
            : 'bg-tertiary text-on-tertiary'
        }`}
        type="button"
      >
        {running ? <Stop size={18} weight="fill" /> : <Play size={18} weight="fill" />}
        {loading ? 'Loading model…' : running ? 'Stop monitoring' : 'Start monitoring'}
      </button>
    </nav>
  );
}
