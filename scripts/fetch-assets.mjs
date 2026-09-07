/**
 * Copies the MediaPipe WASM runtime out of node_modules and fetches the face
 * landmarker model into public/, so the deployed app has no runtime CDN
 * dependency. Idempotent — safe to run on every dev and build.
 */
import { createWriteStream } from 'node:fs';
import { cp, mkdir, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WASM_SRC = resolve(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const WASM_DEST = resolve(root, 'public/mediapipe/wasm');
const MODEL_DEST = resolve(root, 'public/mediapipe/face_landmarker.task');
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const exists = async (path) => {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
};

await mkdir(dirname(MODEL_DEST), { recursive: true });

if (await exists(WASM_DEST)) {
  console.log('wasm runtime already vendored');
} else {
  await cp(WASM_SRC, WASM_DEST, { recursive: true });
  console.log('vendored wasm runtime -> public/mediapipe/wasm');
}

if (await exists(MODEL_DEST)) {
  console.log('face landmarker model already present');
} else {
  console.log('downloading face landmarker model (~3.7 MB)…');
  const response = await fetch(MODEL_URL).catch((cause) => {
    throw new Error(`Could not reach ${MODEL_URL} — check your network or proxy.`, { cause });
  });
  if (!response.ok) {
    throw new Error(
      `Model download failed with HTTP ${response.status}.\n` +
        `Download it manually from:\n  ${MODEL_URL}\n` +
        `and save it to:\n  public/mediapipe/face_landmarker.task`,
    );
  }
  await pipeline(Readable.fromWeb(response.body), createWriteStream(MODEL_DEST));
  console.log('saved -> public/mediapipe/face_landmarker.task');
}
