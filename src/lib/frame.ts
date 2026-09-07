/**
 * Mean luminance of the video frame, 0–1.
 *
 * Landmark tracking degrades badly in the dark and the driver deserves to be
 * told that rather than silently trusting a bad signal. Sampled on a tiny
 * offscreen canvas so the cost does not scale with camera resolution.
 */
const SAMPLE = 32;
let scratch: HTMLCanvasElement | null = null;

export function frameLuminance(video: HTMLVideoElement): number {
  scratch ??= document.createElement('canvas');
  scratch.width = scratch.height = SAMPLE;
  const ctx = scratch.getContext('2d', { willReadFrequently: true });
  if (!ctx || !video.videoWidth) return 1;

  ctx.drawImage(video, 0, 0, SAMPLE, SAMPLE);
  const { data } = ctx.getImageData(0, 0, SAMPLE, SAMPLE);

  let total = 0;
  for (let i = 0; i < data.length; i += 4) {
    // Rec. 601 luma — weights the channels by perceived brightness.
    total += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return total / (data.length / 4) / 255;
}
