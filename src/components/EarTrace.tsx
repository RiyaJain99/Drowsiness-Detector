import { useEffect, useRef } from 'react';
import { DEFAULT_CONFIG } from '../lib/config';
import type { Sample } from '../lib/drowsiness';

const HEIGHT = 150;
const EAR_CEILING = 0.45;

/**
 * Rolling EAR trace with the calibrated threshold drawn in and closure periods
 * shaded. This is the one place someone can watch the algorithm decide, rather
 * than take the verdict on trust.
 */
export function EarTrace({ trace, threshold }: { trace: readonly Sample[]; threshold: number | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = canvas.clientWidth * dpr;
    const h = HEIGHT * dpr;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    if (threshold === null || trace.length === 0) return;

    const window_ = DEFAULT_CONFIG.traceWindowMs;
    const end = trace[trace.length - 1].t;
    const start = end - window_;
    const points = trace.filter((s) => s.t >= start);
    const pad = 10 * dpr;
    const x = (t: number) => ((t - start) / window_) * w;
    const y = (v: number) => h - pad - (Math.min(v, EAR_CEILING) / EAR_CEILING) * (h - pad * 2);

    // Shade every run of closed frames.
    ctx.fillStyle = 'rgba(255,84,81,.22)';
    let runStart: number | null = null;
    points.forEach((s, i) => {
      if (s.closed && runStart === null) runStart = s.t;
      if ((!s.closed || i === points.length - 1) && runStart !== null) {
        ctx.fillRect(x(runStart), 0, Math.max(2 * dpr, x(s.t) - x(runStart)), h);
        runStart = null;
      }
    });

    ctx.setLineDash([6 * dpr, 5 * dpr]);
    ctx.strokeStyle = '#ff5451';
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(0, y(threshold));
    ctx.lineTo(w, y(threshold));
    ctx.stroke();
    ctx.setLineDash([]);

    const accent = getComputedStyle(document.body).getPropertyValue('--state').trim();

    // Gradient fill under the curve, so the reading reads as a level rather
    // than just a line.
    if (points.length > 1) {
      const fill = ctx.createLinearGradient(0, pad, 0, h);
      fill.addColorStop(0, accent + '55');
      fill.addColorStop(1, accent + '00');
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.moveTo(x(points[0].t), h);
      points.forEach((s) => ctx.lineTo(x(s.t), y(s.ear)));
      ctx.lineTo(x(points[points.length - 1].t), h);
      ctx.closePath();
      ctx.fill();
    }

    ctx.strokeStyle = accent;
    ctx.lineWidth = 2 * dpr;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    points.forEach((s, i) => (i === 0 ? ctx.moveTo(x(s.t), y(s.ear)) : ctx.lineTo(x(s.t), y(s.ear))));
    ctx.stroke();

    // Live value, marked at the head of the line.
    const last = points[points.length - 1];
    ctx.fillStyle = accent;
    ctx.beginPath();
    ctx.arc(x(last.t), y(last.ear), 3 * dpr, 0, Math.PI * 2);
    ctx.fill();
  }, [trace, threshold]);

  const live = trace.length > 0 ? trace[trace.length - 1].ear : null;

  return (
    <section className="rounded-2xl border border-outline-variant/30 bg-surface-container-low px-3.5 pt-3.5 pb-2.5">
      <div className="flex items-baseline gap-2 flex-wrap mb-2.5">
        <h2 className="m-0 text-body-md font-semibold text-on-surface">Eye aspect ratio · last 20s</h2>
        <div className="flex gap-2.5 ml-auto text-telemetry-mono-sm font-mono text-outline">
          <span className="inline-flex items-center gap-1.5">
            <i className="w-3.5 h-0.5 rounded-sm" style={{ background: 'var(--state)' }} />
            EAR{live !== null ? ` ${live.toFixed(2)}` : ''}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <i className="w-3.5 border-t-2 border-dashed border-primary-container" />
            threshold
          </span>
          <span className="inline-flex items-center gap-1.5">
            <i className="w-2.5 h-2.5 rounded-sm bg-primary-container/40" />
            closed
          </span>
        </div>
      </div>
      <canvas ref={canvasRef} className="w-full block" style={{ height: HEIGHT }} />
    </section>
  );
}
