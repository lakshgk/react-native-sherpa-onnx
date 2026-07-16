import { DeviceEventEmitter } from 'react-native';
import SherpaOnnx from '../NativeSherpaOnnx';

/**
 * Wake-word (openWakeWord) JS API — N8 spike surface.
 *
 * The native detector consumes a tee of the PCM live stream (see
 * `createPcmLiveStream` in ../audio): scores are only emitted while a PCM live
 * stream is running. Android only; iOS implementation pending.
 */

export type WakeWordInitOptions = {
  /** Absolute path to openWakeWord's shared melspectrogram.onnx. */
  melspectrogramPath: string;
  /** Absolute path to openWakeWord's shared embedding_model.onnx. */
  embeddingPath: string;
  /** Absolute path to the custom-trained classifier head (.onnx). */
  classifierPath: string;
  /** Placeholder threshold for the emitted `detected` flag (default 0.5). */
  threshold?: number;
};

export type WakeWordScoreEvent = {
  /** Raw classifier confidence for the most recent audio window, 0..1. */
  score: number;
  /** Epoch millis at inference completion. */
  timestamp: number;
  /** score >= threshold (placeholder gate — no consecutive-frames requirement). */
  detected: boolean;
};

export type WakeWordDetectorHandle = {
  start: () => Promise<void>;
  stop: () => Promise<void>;
  release: () => Promise<void>;
  /** Subscribe to per-window scores. Returns an unsubscribe function. */
  onScore: (callback: (event: WakeWordScoreEvent) => void) => () => void;
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
    onScore: (callback: (event: WakeWordScoreEvent) => void) => {
      const sub = DeviceEventEmitter.addListener(
        'wakeWordScore',
        (event: { score?: number; timestamp?: number; detected?: boolean }) => {
          callback({
            score: typeof event?.score === 'number' ? event.score : 0,
            timestamp:
              typeof event?.timestamp === 'number' ? event.timestamp : Date.now(),
            detected: event?.detected === true,
          });
        }
      );
      return () => sub.remove();
    },
  };
}
