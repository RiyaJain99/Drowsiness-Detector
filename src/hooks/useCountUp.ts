import { useEffect, useState } from 'react';

/** Eases from 0 to `target` once `active`. Display only: the target is always the real value. */
export function useCountUp(target: number, active: boolean, durationMs = 1100): number {
  const [value, setValue] = useState(target);

  useEffect(() => {
    if (!active || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setValue(target);
      return;
    }
    const began = performance.now();
    let frame = 0;
    const step = (now: number) => {
      const progress = Math.min(1, (now - began) / durationMs);
      setValue(target * (1 - (1 - progress) ** 3));
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, active, durationMs]);

  return value;
}
