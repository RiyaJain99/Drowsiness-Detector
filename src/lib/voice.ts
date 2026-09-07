import type { Reading } from './drowsiness';

/**
 * Voice assistant: say "Riri" any time monitoring is running to ask for a
 * status readout, or to silence the alarm mid-critical. Wraps the (still
 * vendor-prefixed) Web Speech API, which TypeScript's DOM lib does not
 * describe.
 */
export interface SpeechRecognitionResultLike {
  0: { transcript: string };
  isFinal: boolean;
}

export interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
}

export interface SpeechRecognitionErrorEventLike {
  error: string;
}

export interface SpeechRecognitionLike extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function getRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export const voiceSupported = getRecognitionCtor() !== null;

export function createRecognition(): SpeechRecognitionLike | null {
  const Ctor = getRecognitionCtor();
  return Ctor ? new Ctor() : null;
}

export function speak(text: string): void {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.02;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

// ──────────────────────────────────────────────────────── command parsing

export type QueryMetric = 'perclos' | 'yawns' | 'blinks' | 'microsleeps' | 'nods' | 'summary';

export type VoiceCommand = { type: 'acknowledge' } | { type: 'query'; metric: QueryMetric } | { type: 'greeting' };

const WAKE_WORD = /\briri\b/i;
const ADDRESS_FILLER = /\b(hey|ok|okay)\b/gi;
const ACKNOWLEDGE_PATTERN = /\b(i'?m\s+up|i\s+am\s+up|i'?m\s+awake|i\s+am\s+awake)\b/i;
const METRIC_PATTERNS: ReadonlyArray<{ pattern: RegExp; metric: QueryMetric }> = [
  { pattern: /perclos/i, metric: 'perclos' },
  { pattern: /\byawn/i, metric: 'yawns' },
  { pattern: /\bblink/i, metric: 'blinks' },
  { pattern: /microsleep/i, metric: 'microsleeps' },
  { pattern: /\b(nod|head)/i, metric: 'nods' },
];

/**
 * Everything requires the wake word — a distinctive metric keyword or an
 * exact "I'm up" phrase is still easy to say by accident in ordinary
 * conversation, and a driving-safety app should not blurt out a spoken
 * response, or silence a real alarm, because of a passenger's small talk.
 */
export function parseVoiceCommand(transcript: string): VoiceCommand | null {
  if (!WAKE_WORD.test(transcript)) return null;

  const metric = METRIC_PATTERNS.find(({ pattern }) => pattern.test(transcript));
  if (metric) return { type: 'query', metric: metric.metric };

  if (ACKNOWLEDGE_PATTERN.test(transcript)) return { type: 'acknowledge' };

  // Just the wake word (plus filler like "hey" or trailing punctuation) and
  // nothing else — respond like being addressed by name, not by dumping the
  // whole status readout.
  const remainder = transcript
    .replace(WAKE_WORD, '')
    .replace(ADDRESS_FILLER, '')
    .replace(/[^a-z0-9]/gi, '');
  if (remainder.length === 0) return { type: 'greeting' };

  // A follow-up that didn't match anything more specific still gets a summary.
  return { type: 'query', metric: 'summary' };
}

// ─────────────────────────────────────────────────────── spoken responses

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

export function describeReading(reading: Reading, metric: QueryMetric): string {
  if (reading.threshold === null) return "Still calibrating — I don't have a reading yet.";

  switch (metric) {
    case 'perclos':
      return `PERCLOS is ${reading.perclos.toFixed(0)} percent.`;
    case 'yawns':
      return reading.yawns === 0 ? "No yawns yet this session." : `${plural(reading.yawns, 'yawn')} so far.`;
    case 'blinks':
      return `${reading.blinksPerMinute} blinks per minute.`;
    case 'microsleeps':
      return reading.microsleeps === 0
        ? 'No microsleeps detected.'
        : `${plural(reading.microsleeps, 'microsleep')} detected.`;
    case 'nods':
      return reading.nods === 0 ? 'No head nods detected.' : `${plural(reading.nods, 'head nod')} detected.`;
    case 'summary':
    default: {
      const verdict =
        reading.state === 'critical'
          ? "You're at a critical drowsiness level. Find somewhere to stop."
          : reading.state === 'drowsy'
            ? "You're reading as drowsy. Consider a break."
            : "You're alert.";
      return `${verdict} PERCLOS ${reading.perclos.toFixed(0)} percent, ${plural(reading.yawns, 'yawn')}, ${plural(reading.microsleeps, 'microsleep')} this session.`;
    }
  }
}
