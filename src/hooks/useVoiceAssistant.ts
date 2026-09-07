import { useEffect, useRef } from 'react';
import type { Reading } from '../lib/drowsiness';
import { createRecognition, describeReading, parseVoiceCommand, speak, voiceSupported } from '../lib/voice';

/**
 * Listens for "Riri" while `active` — asking for a status readout any time,
 * or silencing the alarm mid-critical. Scoped to the monitoring session
 * rather than always-on: this app's whole pitch is that nothing runs without
 * a reason, and an open microphone before the driver even starts would work
 * against that.
 */
export function useVoiceAssistant(active: boolean, reading: Reading, onAcknowledge: () => void): void {
  const readingRef = useRef(reading);
  readingRef.current = reading;

  useEffect(() => {
    if (!active || !voiceSupported) return;

    const recognition = createRecognition();
    if (!recognition) return;

    let stopped = false;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        // Interim results are partial guesses that keep changing as more is
        // said — matching against them fires early on a half-finished
        // sentence (e.g. "Riri" alone, before the actual question) and the
        // follow-up cancel()+speak() for the real answer can silently drop
        // audio in Chrome. Only the final transcript is acted on.
        if (!result.isFinal) continue;

        const command = parseVoiceCommand(result[0].transcript);
        if (!command) continue;

        const current = readingRef.current;
        if (command.type === 'greeting') {
          speak('Yes?');
        } else if (command.type === 'acknowledge' && current.state === 'critical' && !current.acknowledged) {
          speak("Glad you're up. Alarm silenced.");
          onAcknowledge();
        } else {
          speak(describeReading(current, command.type === 'query' ? command.metric : 'summary'));
        }
        return;
      }
    };
    // Permission denial or no speech service: stop retrying rather than
    // spamming the mic prompt every time recognition auto-restarts.
    recognition.onerror = (event) => {
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') stopped = true;
    };
    // Browsers time recognition sessions out after a period of silence; keep
    // it running for as long as the session is active.
    recognition.onend = () => {
      if (!stopped) {
        try {
          recognition.start();
        } catch {
          // Already running — a start() during an in-flight session throws.
        }
      }
    };

    try {
      recognition.start();
    } catch {
      // Ignore — most commonly a duplicate start in dev's StrictMode double-invoke.
    }

    return () => {
      stopped = true;
      recognition.onend = null;
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.stop();
    };
  }, [active, onAcknowledge]);
}
