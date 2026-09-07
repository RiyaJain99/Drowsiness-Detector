import { Eye, VideoCamera, WarningOctagon } from '@phosphor-icons/react';
import { useEffect, useRef } from 'react';
import type { DriverState } from '../lib/drowsiness';
import type { Point } from '../lib/geometry';
import { EYE_CONTOURS, FACE_OVAL } from '../lib/landmarks';
import { STATE_COPY } from './TopBar';

interface CameraStageProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  landmarks: Point[] | null;
  closed: boolean;
  state: DriverState;
  reason: string;
  ear: number;
  pitch: number;
  luma: number;
  lowLight: boolean;
  running: boolean;
  fps: number;
  error: string | null;
}

const Corner = ({ className }: { className: string }) => (
  <div className={`absolute w-3 h-3 border-2 pointer-events-none ${className}`} style={{ borderColor: 'var(--state)' }} />
);

export function CameraStage({
  videoRef, landmarks, closed, state, reason, ear, pitch, luma, lowLight, running, fps, error,
}: CameraStageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !running || !video.videoWidth) return;

    const { videoWidth: w, videoHeight: h } = video;
    if (canvas.width !== w) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Mirror the feed: an unmirrored self-view reads as wrong to everyone.
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    if (!landmarks) return;
    const accent = getComputedStyle(document.body).getPropertyValue('--state').trim();
    const path = (indices: readonly number[]) => {
      ctx.beginPath();
      indices.forEach((index, n) => {
        const p = landmarks[index];
        const x = w - p.x * w; // mirrored to match the frame
        const y = p.y * h;
        n === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.closePath();
    };

    // Face outline, quiet.
    ctx.strokeStyle = accent;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = 1.5;
    path(FACE_OVAL);
    ctx.stroke();

    // Eyes, loud — this is what the verdict is built from.
    ctx.globalAlpha = 1;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    for (const contour of [EYE_CONTOURS.left, EYE_CONTOURS.right]) {
      path(contour);
      if (closed) {
        ctx.fillStyle = 'rgba(255,84,81,.3)';
        ctx.fill();
      }
      ctx.stroke();
    }
  }, [landmarks, closed, running, videoRef]);

  const tracking = landmarks !== null;

  return (
    <section
      className="relative rounded-2xl overflow-hidden bg-surface-container-lowest border-2 aspect-video shadow-[0_0_30px_-8px_var(--state-soft)]"
      style={{ borderColor: 'var(--state)' }}
    >
      <video ref={videoRef} playsInline muted className="hidden" />
      <canvas ref={canvasRef} className="w-full h-full block object-cover" />

      {running ? (
        <>
          {state === 'calibrating' && (
            <div
              className="absolute inset-x-0 h-16 pointer-events-none sweep"
              style={{ background: 'linear-gradient(to bottom, transparent, var(--state-soft), transparent)' }}
            />
          )}

          {/* HUD reticle corners frame the tracked face, quiet by default */}
          <Corner className="top-2.5 left-2.5 border-r-0 border-b-0 rounded-tl" />
          <Corner className="top-2.5 right-2.5 border-l-0 border-b-0 rounded-tr" />
          <Corner className="bottom-2.5 left-2.5 border-r-0 border-t-0 rounded-bl" />
          <Corner className="bottom-2.5 right-2.5 border-l-0 border-t-0 rounded-br" />

          <div className="absolute top-2.5 inset-x-6 flex justify-between items-center gap-2 text-telemetry-mono-sm font-mono">
            <span
              className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-black/75 backdrop-blur-md border font-semibold"
              style={{ color: 'var(--state)', borderColor: 'var(--state)' }}
            >
              {tracking ? <Eye size={11} weight="bold" /> : <WarningOctagon size={11} weight="bold" />}
              {tracking ? (closed ? 'EYES CLOSED' : 'TRACKING') : 'NO FACE'}
            </span>
            <span className="flex items-center gap-2 px-2 py-0.5 rounded bg-black/75 backdrop-blur-md border border-outline-variant/40 text-on-surface-variant tabular-nums">
              <span>{fps} FPS</span>
              <span className="text-outline">|</span>
              <span className={lowLight ? 'text-secondary font-semibold' : ''}>
                LIGHT {lowLight ? 'LOW' : `${Math.round(luma * 100)}%`}
              </span>
            </span>
          </div>

          <div className="absolute bottom-2.5 inset-x-6 flex items-end justify-between gap-2">
            <div className="min-w-0">
              <div
                className="font-mono text-2xl font-bold tracking-tight leading-none drop-shadow-[0_2px_12px_rgba(0,0,0,.9)]"
                style={{ color: 'var(--state)' }}
              >
                {STATE_COPY[state].toUpperCase()}
              </div>
              <div className="text-telemetry-mono-sm text-on-surface-variant mt-1 truncate drop-shadow-[0_1px_6px_rgba(0,0,0,.9)]">
                {reason}
              </div>
            </div>
            <div className="flex flex-col items-end gap-1 text-telemetry-mono-sm font-mono tabular-nums shrink-0">
              <span className="px-2 py-0.5 rounded bg-black/75 backdrop-blur-md border border-outline-variant/40 text-on-surface-variant">
                EAR {ear.toFixed(3)}
              </span>
              <span
                className={`px-2 py-0.5 rounded bg-black/75 backdrop-blur-md border border-outline-variant/40 ${
                  pitch < -12 ? 'text-primary-container font-semibold' : 'text-on-surface-variant'
                }`}
              >
                PITCH {pitch > 0 ? '+' : ''}
                {pitch.toFixed(0)}°{pitch < -12 ? ' NOD' : ''}
              </span>
            </div>
          </div>
        </>
      ) : (
        <div className="absolute inset-0 grid place-content-center text-center px-8 gap-2">
          {error ? (
            <WarningOctagon size={28} weight="light" className="text-primary-container mx-auto" />
          ) : (
            <VideoCamera size={28} weight="light" className="text-outline mx-auto" />
          )}
          <strong className="block text-on-surface font-semibold text-body-md">
            {error ? 'Monitoring did not start' : 'Nothing is being recorded'}
          </strong>
          <p className="text-body-sm text-outline leading-relaxed m-0">
            {error ?? 'Detection runs in your browser. No video is uploaded anywhere.'}
          </p>
        </div>
      )}
    </section>
  );
}
