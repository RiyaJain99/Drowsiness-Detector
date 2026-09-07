/** A tight run of square-wave beeps, synthesised so the app ships no audio files. */
let ctx: AudioContext | null = null;

const BEEPS_PER_BURST = 3;
const BEEP_SPACING_S = 0.16;

export function playAlarm(): void {
  ctx ??= new AudioContext();
  if (ctx.state === 'suspended') void ctx.resume();

  const start = ctx.currentTime;
  for (let i = 0; i < BEEPS_PER_BURST; i++) {
    const at = start + i * BEEP_SPACING_S;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.value = 880;
    // Ramped rather than switched, so it does not click on cheap laptop speakers.
    gain.gain.setValueAtTime(0, at);
    gain.gain.linearRampToValueAtTime(0.16, at + 0.015);
    gain.gain.linearRampToValueAtTime(0, at + BEEP_SPACING_S - 0.02);
    osc.connect(gain).connect(ctx.destination);
    osc.start(at);
    osc.stop(at + BEEP_SPACING_S);
  }
}

/** Milliseconds between bursts when the alarm is left to repeat unattended. */
export const ALARM_REPEAT_MS = 1400;
