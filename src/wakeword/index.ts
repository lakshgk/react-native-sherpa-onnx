import { DeviceEventEmitter } from 'react-native';
import SherpaOnnx from '../NativeSherpaOnnx';

/**
 * Wake-word (openWakeWord) JS API — Smriti C2 / GP-2026-015 Prompt 1.
 *
 * The native detector consumes a tee of the PCM live stream (see
 * `createPcmLiveStream` in ../audio): detections are only emitted while a PCM
 * live stream is running. Android only; iOS implementation pending.
 *
 * Detection contract (DL-058 / DL-065): the native layer emits one
 * `wakeWordDetection` event per 80 ms inference frame scoring at or above the
 * signed threshold — threshold-only gating, raw and UNDEDUPLICATED. Consumers
 * own dedup, comparing against the event's native `timestamp` (not JS arrival
 * time). See OpenWakeWordHelper.kt for the spec provenance header.
 */

export type WakeWordInitOptions = {
  /** Absolute path to openWakeWord's shared melspectrogram.onnx. */
  melspectrogramPath: string;
  /** Absolute path to openWakeWord's shared embedding_model.onnx. */
  embeddingPath: string;
  /** Absolute path to the custom-trained classifier head (.onnx). */
  classifierPath: string;
  /**
   * The signed wake_word_model detection threshold (REQUIRED — no default;
   * DL-058: threshold-only, no frame-count parameter exists).
   */
  threshold: number;
};

export type WakeWordDetectionEvent = {
  /**
   * Always null at this layer: the two-stage voice_command shape (UL-014 v2)
   * with the VERB stage unfilled — verb capture happens downstream.
   */
  verb: null;
  /** Classifier confidence for the detected frame, threshold..1. */
  confidence: number;
  /** NATIVE epoch-millis at inference completion — dedup against this. */
  timestamp: number;
};

export type WakeWordDetectorHandle = {
  start: () => Promise<void>;
  stop: () => Promise<void>;
  release: () => Promise<void>;
  /**
   * Subscribe to raw, undeduplicated detection events. Returns an
   * unsubscribe function.
   */
  onDetection: (callback: (event: WakeWordDetectionEvent) => void) => () => void;
};

/**
 * Initialize the openWakeWord detector from three model files and return a
 * handle. Mirrors the createPcmLiveStream handle pattern (../audio/index.ts).
 */
export async function createWakeWordDetector(
  options: WakeWordInitOptions
): Promise<WakeWordDetectorHandle> {
  await SherpaOnnx.initializeWakeWord({
    melspectrogramPath: options.melspectrogramPath,
    embeddingPath: options.embeddingPath,
    classifierPath: options.classifierPath,
    threshold: options.threshold,
  });

  return {
    start: () => SherpaOnnx.startWakeWordDetection(),
    stop: () => SherpaOnnx.stopWakeWordDetection(),
    release: () => SherpaOnnx.unloadWakeWord(),
    onDetection: (callback: (event: WakeWordDetectionEvent) => void) => {
      const sub = DeviceEventEmitter.addListener(
        'wakeWordDetection',
        (event: { verb?: null; confidence?: number; timestamp?: number }) => {
          callback({
            verb: null,
            confidence:
              typeof event?.confidence === 'number' ? event.confidence : 0,
            timestamp:
              typeof event?.timestamp === 'number' ? event.timestamp : Date.now(),
          });
        }
      );
      return () => sub.remove();
    },
  };
}
