import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { useCallback, useEffect, useRef, useState } from 'react';
import { DEFAULT_CONFIG } from '../lib/config';
import { DrowsinessAnalyzer, type MonitorEvent, type Reading, type Sample } from '../lib/drowsiness';
import { frameLuminance } from '../lib/frame';
import { faceMetrics, meanEyeAspectRatio, mouthAspectRatio, type Point } from '../lib/geometry';

const IDLE: Reading = {
  state: 'idle', reason: '', ear: 0, mar: 0, closed: false, perclos: 0,
  blinksPerMinute: 0, yawns: 0, microsleeps: 0, nods: 0, pitch: 0,
  threshold: null, baseline: null, acknowledged: false, transitioned: false,
};

/** Luminance is sampled every N frames — it changes slowly and costs a readback. */
const LUMA_INTERVAL = 15;

export interface Monitor {
  reading: Reading;
  events: readonly MonitorEvent[];
  landmarks: Point[] | null;
  trace: readonly Sample[];
  fps: number;
  luma: number;
  lowLight: boolean;
  elapsedMs: number;
  running: boolean;
  loading: boolean;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
  recalibrate: () => void;
  acknowledge: () => void;
}

export function useMonitor(videoRef: React.RefObject<HTMLVideoElement | null>): Monitor {
  const analyzer = useRef(new DrowsinessAnalyzer());
  const landmarker = useRef<FaceLandmarker | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const frame = useRef(0);
  const lastFrameTime = useRef(-1);
  const frameCount = useRef(0);
  const lumaTick = useRef(0);
  const fpsMark = useRef(0);
  const startedAt = useRef(0);

  const [reading, setReading] = useState<Reading>(IDLE);
  const [events, setEvents] = useState<readonly MonitorEvent[]>([]);
  const [landmarks, setLandmarks] = useState<Point[] | null>(null);
  const [trace, setTrace] = useState<readonly Sample[]>([]);
  const [fps, setFps] = useState(0);
  const [luma, setLuma] = useState(1);
  const [elapsedMs, setElapsed] = useState(0);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loop = useCallback(() => {
    frame.current = requestAnimationFrame(loop);
    const video = videoRef.current;
    const model = landmarker.current;
    if (!video || !model || video.readyState < 2) return;

    const now = performance.now();
    // MediaPipe rejects a repeated timestamp, so only run on a fresh frame.
    if (video.currentTime !== lastFrameTime.current) {
      lastFrameTime.current = video.currentTime;
      const points = model.detectForVideo(video, now).faceLandmarks?.[0] as Point[] | undefined;

      if (points) {
        const next = analyzer.current.push(
          {
            ear: meanEyeAspectRatio(points),
            mar: mouthAspectRatio(points),
            face: faceMetrics(points),
          },
          now,
        );
        setReading({ ...next, acknowledged: analyzer.current.isAcknowledged(now) });
        setLandmarks(points);
      } else {
        analyzer.current.faceLost(now);
        setLandmarks(null);
      }
      setTrace([...analyzer.current.trace]);
      setEvents([...analyzer.current.events]);
      frameCount.current += 1;

      if (++lumaTick.current % LUMA_INTERVAL === 0) setLuma(frameLuminance(video));
    }

    if (now - fpsMark.current > 1000) {
      setFps(frameCount.current);
      frameCount.current = 0;
      fpsMark.current = now;
    }
    setElapsed(Date.now() - startedAt.current);
  }, [videoRef]);

  const start = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
      landmarker.current = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: '/mediapipe/face_landmarker.task', delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
      });

      stream.current = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      const video = videoRef.current!;
      video.srcObject = stream.current;
      await video.play();

      analyzer.current.reset();
      analyzer.current.beginCalibration(performance.now());
      startedAt.current = Date.now();
      fpsMark.current = performance.now();
      setRunning(true);
      setLoading(false);
      frame.current = requestAnimationFrame(loop);
    } catch (err) {
      setLoading(false);
      const e = err as DOMException;
      setError(
        e.name === 'NotAllowedError'
          ? 'Camera access was blocked. Allow it in your browser’s site settings, then start again.'
          : e.name === 'NotFoundError'
            ? 'No camera was found on this device.'
            : `Could not start monitoring: ${e.message}`,
      );
    }
  }, [loop, videoRef]);

  const stop = useCallback(() => {
    cancelAnimationFrame(frame.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    landmarker.current?.close();
    landmarker.current = null;
    lastFrameTime.current = -1;
    analyzer.current.reset();
    setRunning(false);
    setReading(IDLE);
    setLandmarks(null);
    setTrace([]);
    setFps(0);
    setLuma(1);
  }, []);

  const recalibrate = useCallback(() => {
    if (running) analyzer.current.beginCalibration(performance.now());
  }, [running]);

  const acknowledge = useCallback(() => {
    const now = performance.now();
    analyzer.current.acknowledge(now);
    setEvents([...analyzer.current.events]);
    setReading((r) => ({ ...r, acknowledged: true }));
  }, []);

  useEffect(() => stop, [stop]);

  return {
    reading, events, landmarks, trace, fps, luma,
    lowLight: luma < DEFAULT_CONFIG.lowLightLuma,
    elapsedMs, running, loading, error,
    start, stop, recalibrate, acknowledge,
  };
}
