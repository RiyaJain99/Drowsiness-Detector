import { describe, expect, it } from 'vitest';
import type { Reading } from '../drowsiness';
import { describeReading, parseVoiceCommand } from '../voice';

const READING: Reading = {
  state: 'alert',
  reason: 'eyes open',
  ear: 0.3,
  mar: 0.1,
  closed: false,
  perclos: 4,
  blinksPerMinute: 14,
  yawns: 0,
  microsleeps: 0,
  nods: 0,
  pitch: 0,
  threshold: 0.23,
  baseline: 0.3,
  acknowledged: false,
  transitioned: false,
};

describe('parseVoiceCommand', () => {
  it('ignores speech without the wake word', () => {
    expect(parseVoiceCommand("I'm up")).toBeNull();
    expect(parseVoiceCommand("what's my perclos")).toBeNull();
    expect(parseVoiceCommand('good morning')).toBeNull();
  });

  it('is case-insensitive and matches inside a longer sentence', () => {
    expect(parseVoiceCommand('HEY RIRI, WHAT IS MY PERCLOS')).toEqual({ type: 'query', metric: 'perclos' });
    expect(parseVoiceCommand("hey riri i am up don't beep like that")).toEqual({ type: 'acknowledge' });
  });

  it('picks out a metric-specific query over the generic summary', () => {
    expect(parseVoiceCommand('riri what is my perclos')).toEqual({ type: 'query', metric: 'perclos' });
    expect(parseVoiceCommand('riri how many yawns')).toEqual({ type: 'query', metric: 'yawns' });
    expect(parseVoiceCommand('riri what is my blink rate')).toEqual({ type: 'query', metric: 'blinks' });
    expect(parseVoiceCommand('riri any microsleeps')).toEqual({ type: 'query', metric: 'microsleeps' });
    expect(parseVoiceCommand('riri did I nod off')).toEqual({ type: 'query', metric: 'nods' });
  });

  it('treats "I am up" as acknowledge only when no metric is mentioned', () => {
    expect(parseVoiceCommand("riri i'm up")).toEqual({ type: 'acknowledge' });
    // A metric keyword takes priority — asking a question isn't a snooze.
    expect(parseVoiceCommand("riri i'm up, what's my perclos")).toEqual({ type: 'query', metric: 'perclos' });
  });

  it('responds like being addressed by name when nothing else was said', () => {
    expect(parseVoiceCommand('riri')).toEqual({ type: 'greeting' });
    expect(parseVoiceCommand('hey riri')).toEqual({ type: 'greeting' });
    expect(parseVoiceCommand('Riri?')).toEqual({ type: 'greeting' });
  });

  it('gives a summary once asked something that is not a specific metric', () => {
    expect(parseVoiceCommand('riri how am I doing')).toEqual({ type: 'query', metric: 'summary' });
    expect(parseVoiceCommand("riri what's up")).toEqual({ type: 'query', metric: 'summary' });
  });

  it('does not match "riri" as a substring of another word', () => {
    // "cariring" contains the letters r-i-r-i, but not as a standalone word.
    expect(parseVoiceCommand('cariring down the highway')).toBeNull();
    expect(parseVoiceCommand('hey riri, cariring down the highway')).toEqual({ type: 'query', metric: 'summary' });
  });
});

describe('describeReading', () => {
  it('reports calibrating before a threshold exists', () => {
    expect(describeReading({ ...READING, threshold: null }, 'summary')).toMatch(/calibrating/i);
  });

  it('answers a metric-specific query with just that number', () => {
    expect(describeReading(READING, 'perclos')).toContain('4 percent');
    expect(describeReading({ ...READING, yawns: 1 }, 'yawns')).toBe('1 yawn so far.');
    expect(describeReading({ ...READING, yawns: 3 }, 'yawns')).toBe('3 yawns so far.');
    expect(describeReading(READING, 'yawns')).toMatch(/no yawns/i);
  });

  it('names the driving-relevant verdict in the summary', () => {
    expect(describeReading(READING, 'summary')).toMatch(/alert/i);
    expect(describeReading({ ...READING, state: 'drowsy' }, 'summary')).toMatch(/drowsy/i);
    expect(describeReading({ ...READING, state: 'critical' }, 'summary')).toMatch(/critical/i);
  });
});
