/**
 * Types for open-vocabulary keyword spotting (KWS) via sherpa-onnx KeywordSpotter.
 */

export interface KwsInitOptions {
  /**
   * Absolute path to the KWS model directory (zipformer transducer:
   * encoder/decoder/joiner .onnx + tokens.txt), e.g. an extracted
   * sherpa-onnx-kws-zipformer-gigaspeech-3.3M package.
   */
  modelDir: string;
  /**
   * Absolute path to the keywords file in sherpa-onnx tokenized keywords format
   * (one keyword per line, space-separated model tokens, optional :boost /
   * #threshold / @ORIGINAL suffixes). See
   * https://k2-fsa.github.io/sherpa/onnx/kws/index.html
   */
  keywordsFile: string;
  /** Boost score applied to keyword paths during search (default 1.5). */
  keywordsScore?: number;
  /** Detection threshold, 0–1 (default 0.25). Higher = fewer false positives. */
  keywordsThreshold?: number;
  /** Beam width for the search (default 4). */
  maxActivePaths?: number;
  /** Trailing blank frames required to commit a detection (default 2). */
  numTrailingBlanks?: number;
  numThreads?: number;
  provider?: string;
  debug?: boolean;
}

export interface KwsDetection {
  /** Detected keyword label ('' when nothing detected in this chunk). */
  keyword: string;
  tokens: string[];
  timestamps: number[];
  /** True when keyword is non-empty. The native stream auto-resets after a detection. */
  detected: boolean;
}

export interface KeywordSpotterEngine {
  /**
   * Feed a chunk of 16 kHz mono float PCM ([-1, 1]); decodes and returns the
   * current spotting result. Call repeatedly with live audio.
   */
  processChunk(samples: number[], sampleRate: number): Promise<KwsDetection>;
  /** Reset stream state manually (native auto-resets after each detection). */
  reset(): Promise<void>;
  /** Release the stream and the spotter instance. */
  release(): Promise<void>;
}
