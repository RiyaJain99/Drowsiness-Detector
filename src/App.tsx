import { useEffect, useRef, useState } from 'react';
import { CameraStage } from './components/CameraStage';
import { ControlDeck } from './components/ControlDeck';
import { CriticalBanner } from './components/CriticalBanner';
import { EarTrace } from './components/EarTrace';
import { EventLog } from './components/EventLog';
import { PerclosCard } from './components/PerclosCard';
import { STATE_COPY, TopBar } from './components/TopBar';
import { TelemetryGrid } from './components/TelemetryGrid';
import { useMonitor } from './hooks/useMonitor';
import { useVoiceAssistant } from './hooks/useVoiceAssistant';
import { ALARM_REPEAT_MS, playAlarm } from './lib/alarm';
import { DEFAULT_CONFIG } from './lib/config';
import { voiceSupported } from './lib/voice';

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const monitor = useMonitor(videoRef);
  const [soundOn, setSoundOn] = useState(true);
  const [voiceOn, setVoiceOn] = useState(true);
  const { reading, events, running, loading, error } = monitor;
  const critical = reading.state === 'critical';
  const alarming = critical && !reading.acknowledged && soundOn;
  // Listens for the whole session, not just mid-alarm, so "Riri, what's my
  // PERCLOS?" works any time — not only "I'm up" to silence a critical alert.
  const listening = running && voiceOn && voiceSupported;

  useEffect(() => {
    document.body.dataset.state = reading.state;
  }, [reading.state]);

  useEffect(() => {
    // Snoozing suppresses the repeat, not the state — the display stays red.
    // Repeats on an interval rather than firing once, so the alarm keeps
    // sounding for as long as the driver hasn't acknowledged it.
    if (!alarming) return;
    playAlarm();
    const id = setInterval(playAlarm, ALARM_REPEAT_MS);
    return () => clearInterval(id);
  }, [alarming]);

  useVoiceAssistant(listening, reading, monitor.acknowledge);

  const exportSession = () => {
    const payload = {
      startedAt: new Date(Date.now() - monitor.elapsedMs).toISOString(),
      durationMs: monitor.elapsedMs,
      calibration: { baseline: reading.baseline, threshold: reading.threshold },
      config: DEFAULT_CONFIG,
      totals: {
        blinksLastMinute: reading.blinksPerMinute,
        yawns: reading.yawns,
        microsleeps: reading.microsleeps,
        headNods: reading.nods,
        finalPerclos: Number(reading.perclos.toFixed(2)),
      },
      events,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = `drowsyguard-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <TopBar
        state={reading.state}
        reason={reading.reason}
        elapsedMs={monitor.elapsedMs}
        running={running}
        listening={listening}
      />

      <main className="flex-1 mx-auto w-full max-w-7xl px-4 sm:px-6 py-5 space-y-4">
        {critical && (
          <CriticalBanner
            reason={reading.reason}
            soundOn={soundOn}
            acknowledged={reading.acknowledged}
            listening={listening}
            onAcknowledge={monitor.acknowledge}
          />
        )}

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
          {/* ── Left column: HUD viewfinder + EAR waveform ── */}
          <div className="xl:col-span-7 space-y-4">
            <CameraStage
              videoRef={videoRef}
              landmarks={monitor.landmarks}
              closed={reading.closed}
              state={reading.state}
              reason={reading.reason}
              ear={reading.ear}
              pitch={reading.pitch}
              luma={monitor.luma}
              lowLight={running && monitor.lowLight}
              running={running}
              fps={monitor.fps}
              error={error}
            />

            {running && monitor.lowLight && (
              <p className="text-body-sm text-secondary bg-secondary/10 border border-secondary/30 rounded-lg px-3 py-2 m-0">
                Light is low, so landmark tracking is less reliable. Readings may drift.
              </p>
            )}

            <EarTrace trace={monitor.trace} threshold={reading.threshold} />
          </div>

          {/* ── Right column: PERCLOS, telemetry, incident timeline ── */}
          <div className="xl:col-span-5 space-y-4">
            <PerclosCard perclos={reading.perclos} ready={reading.threshold !== null} />
            <TelemetryGrid reading={reading} />
            <EventLog events={events} />
          </div>
        </div>

        <ControlDeck
          running={running}
          loading={loading}
          soundOn={soundOn}
          voiceOn={voiceOn}
          voiceSupported={voiceSupported}
          canExport={events.length > 0}
          onToggle={running ? monitor.stop : () => void monitor.start()}
          onRecalibrate={monitor.recalibrate}
          onToggleSound={() => setSoundOn((s) => !s)}
          onToggleVoice={() => setVoiceOn((v) => !v)}
          onExport={exportSession}
        />
      </main>

      {/* One atomic status region: a screen reader hears the verdict, not every number as it ticks. */}
      <p role="status" aria-atomic="true" className="sr-only">
        {running ? `Driver state: ${STATE_COPY[reading.state]}. ${reading.reason}` : 'Monitoring is off.'}
      </p>
    </div>
  );
}
