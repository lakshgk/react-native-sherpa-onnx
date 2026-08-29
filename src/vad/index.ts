import { Buffer } from 'buffer';
import { DeviceEventEmitter } from 'react-native';
import SherpaOnnx from '../NativeSherpaOnnx';

/**
 * Voice Activity Detection (Silero) — Smriti DL-104 Direction A.
 *
 * Replaces the previous throwing placeholder. The native detector consumes a
 * tee of the PCM live stream (see `createPcmLiveStream` in ../audio), so
 * segments are only emitted while a PCM live stream is running. Android only;
 * iOS implementation pending.
 *
 * Segmentation contract (upstream sherpa-onnx VoiceActivityDetector):
 *   - A naturally-closed segment has its trailing silence TRIMMED by
 *     `minSilenceDuration`, and carries roughly 64 ms of leading pre-roll.
 *   - A segment emitted by the stop-time flush (`viaFlush: true`) is NOT
 *     trailing-trimmed, so its tail differs in character from a natural close.
 *   - `flush()` is a no-op when no segment is open, so audio that never
 *     crossed the speech threshold is discarded rather than emitted.
 *
 * ⚠ `maxSpeechDuration` does not cut segments — see `maxSegmentMs` on
 * `VadInitOptions` and the note in ../NativeSherpaOnnx.ts.
 */

/** Decode base64-encoded Int16 PCM to float array in [-1, 1]. Mirrors the
 *  decoder in ../audio/index.ts — same wire form on both events. */
function base64PcmToFloatArray(base64: string): Float32Array {
  const bytes = Buffer.from(base64, 'base64');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const len = bytes.byteLength / 2;
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    out[i] = view.getInt16(i * 2, true) / 32768;
  }
  return out;
}

export type VadInitOptions = {
  /** Absolute path to silero_vad.onnx (k2-fsa v4 export). */
  modelPath: string;
  /** Speech probability gate. Default 0.5. */
  threshold?: number;
  /** Seconds of sub-threshold audio that closes a segment. Default 0.25. */
  minSilenceDuration?: number;
  /** Seconds of speech required before a segment opens. Default 0.25. */
  minSpeechDuration?: number;
  /**
   * Soft pressure valve, NOT a cut (see module note). Default 5.0.
   */
  maxSpeechDuration?: number;
  /** Must be 512 at 16 kHz — other values are fatal natively. Default 512. */
  windowSize?: number;
  /** Must be 16000 — other values are fatal natively. Default 16000. */
  sampleRate?: number;
  /**
   * App-side hard cap in ms. Once a segment has been open this long it is
   * force-flushed and tagged `forcedByCap`. 0 disables the cap, which allows
   * unbounded segment growth on pauseless speech — only appropriate when
   * deliberately observing that behaviour.
   */
  maxSegmentMs?: number;
  debug?: boolean;
};

export type VadSpeechSegment = {
  /** Segment audio at 16 kHz, floats in [-1, 1]. */
  samples: Float32Array;
  /** Absolute sample offset since the last internal reset (i.e. since start). */
  startSample: number;
  /** Segment length in samples (=== samples.length). */
  sampleCount: number;
  /** True when produced by the stop-time flush: NOT trailing-trimmed. */
  viaFlush: boolean;
  /** True when cut by the app-side `maxSegmentMs` cap rather than by silence. */
  forcedByCap: boolean;
  /** How far behind the live audio edge this segment closed. */
  latencyMs: number;
};

export type VadHandle = {
  start: () => Promise<void>;
  stop: () => Promise<void>;
  release: () => Promise<void>;
  /** Subscribe to completed speech segments. Returns an unsubscribe function. */
  onSpeechSegment: (
    callback: (segment: VadSpeechSegment) => void
  ) => () => void;
};

/**
 * Initialize the Silero VAD and return a handle. Mirrors the
 * createWakeWordDetector / createPcmLiveStream handle pattern.
 */
export async function createVad(options: VadInitOptions): Promise<VadHandle> {
  await SherpaOnnx.initializeVad({
    modelPath: options.modelPath,
    threshold: options.threshold,
    minSilenceDuration: options.minSilenceDuration,
    minSpeechDuration: options.minSpeechDuration,
    maxSpeechDuration: options.maxSpeechDuration,
    windowSize: options.windowSize ?? 512,
    sampleRate: options.sampleRate ?? 16000,
    maxSegmentMs: options.maxSegmentMs,
    debug: options.debug,
  });

  return {
    start: () => SherpaOnnx.startVadDetection(),
    stop: () => SherpaOnnx.stopVadDetection(),
    release: () => SherpaOnnx.unloadVad(),
    onSpeechSegment: (callback: (segment: VadSpeechSegment) => void) => {
      const sub = DeviceEventEmitter.addListener(
        'vadSpeechSegment',
        (event: {
          base64Pcm?: string;
          startSample?: number;
          sampleCount?: number;
          viaFlush?: boolean;
          forcedByCap?: boolean;
          latencyMs?: number;
        }) => {
          const base64 = event?.base64Pcm ?? '';
          if (!base64) return;
          const samples = base64PcmToFloatArray(base64);
          callback({
            samples,
            startSample:
              typeof event?.startSample === 'number' ? event.startSample : 0,
            sampleCount:
              typeof event?.sampleCount === 'number'
                ? event.sampleCount
                : samples.length,
            viaFlush: event?.viaFlush === true,
            forcedByCap: event?.forcedByCap === true,
            latencyMs:
              typeof event?.latencyMs === 'number' ? event.latencyMs : 0,
          });
        }
      );
      return () => sub.remove();
    },
  };
}
