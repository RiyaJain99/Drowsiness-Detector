import { Check, MapPin, Microphone, SpeakerHigh, SpeakerSlash, WarningOctagon } from '@phosphor-icons/react';

interface CriticalBannerProps {
  reason: string;
  soundOn: boolean;
  acknowledged: boolean;
  listening: boolean;
  onAcknowledge: () => void;
}

/**
 * Shown only in the critical state. The header already carries a matching red
 * strip and the beacon dot, so this banner does not add motion of its own —
 * colour, weight and size carry the urgency instead.
 */
export function CriticalBanner({ reason, soundOn, acknowledged, listening, onAcknowledge }: CriticalBannerProps) {
  return (
    <section className="rounded-2xl bg-error-container/25 border-2 border-primary-container p-3.5 flex flex-col sm:flex-row sm:items-center gap-3">
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <div className="w-10 h-10 rounded-lg bg-primary-container/20 border border-primary-container flex items-center justify-center text-primary-container shrink-0">
          <WarningOctagon size={20} weight="bold" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="font-label-caps text-label-caps font-bold tracking-wider text-primary-container">
              CRITICAL DROWSINESS
            </span>
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-primary-container text-on-primary-container font-label-caps text-[10px] font-bold">
              {soundOn && !acknowledged ? (
                <>
                  <SpeakerHigh size={10} weight="fill" /> ALARM SOUNDING
                </>
              ) : (
                <>
                  <SpeakerSlash size={10} weight="fill" /> {acknowledged ? 'SNOOZED' : 'MUTED'}
                </>
              )}
            </span>
          </div>
          <p className="text-body-sm text-on-surface-variant mt-0.5 truncate">{reason} — find somewhere to stop</p>
          {listening && (
            <p className="text-body-sm text-tertiary mt-1 flex items-center gap-1.5">
              <Microphone size={12} weight="bold" className="animate-pulse" />
              Listening — say “I’m up” to silence
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-[1fr_auto] gap-2 shrink-0">
        <button
          onClick={onAcknowledge}
          disabled={acknowledged}
          className="py-3 px-4 bg-primary text-on-primary-container rounded-xl font-bold flex items-center justify-center gap-2 text-body-sm uppercase tracking-wide active:scale-[0.98] transition disabled:opacity-50"
          type="button"
        >
          <Check size={18} weight="bold" />
          {acknowledged ? 'Awake — snoozed' : 'I am awake'}
        </button>
        <a
          href="https://www.google.com/maps/search/rest+stop+near+me"
          target="_blank"
          rel="noreferrer"
          className="py-3 px-3.5 bg-surface-container border border-outline-variant/40 rounded-xl text-on-surface-variant flex items-center justify-center gap-1.5 text-body-sm font-medium active:scale-[0.98] transition whitespace-nowrap"
        >
          <MapPin size={16} weight="bold" />
          Rest stop
        </a>
      </div>
    </section>
  );
}
