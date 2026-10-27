import SherpaOnnx from '../NativeSherpaOnnx';

/**
 * Kokoro v1.0 on ONNX Runtime — Smriti GP-2026-072 (Route B, DL-129).
 *
 * sherpa-onnx is built without TTS here, so espeak-ng (GPL-3.0) is not in the
 * native libraries. Text-to-phoneme is the app's job; this surface only runs
 * the model: one session per resident engine, one run per planned unit.
 * Android only. See SherpaOnnxKokoroOrtHelper.kt.
 */

export type KokoroRunResult = {
  /** The unit's waveform: little-endian float32 samples, base64-encoded. */
  base64: string;
  /** Number of float32 samples in `base64`. */
  samples: number;
  /** Native time spent in session.run, ms. */
  runMs: number;
};

/** Create a session on the model at `modelPath` (CPU, `numThreads` intra-op threads). */
export function createKokoroOrtSession(modelPath: string, numThreads: number): Promise<string> {
  return SherpaOnnx.createKokoroSession(modelPath, numThreads);
}

/** Run one unit: token ids including the two 0 pads, a 256-float style row, and speed. */
export function runKokoro(
  handle: string,
  tokenIds: number[],
  style: number[],
  speed: number
): Promise<KokoroRunResult> {
  return SherpaOnnx.runKokoro(handle, tokenIds, style, speed);
}

/** Release a session. Safe to call twice. */
export function releaseKokoroOrtSession(handle: string): Promise<void> {
  return SherpaOnnx.releaseKokoroSession(handle);
}
