import SherpaOnnx from '../NativeSherpaOnnx';
import type {
  KeywordSpotterEngine,
  KwsDetection,
  KwsInitOptions,
} from './types';

export type {
  KeywordSpotterEngine,
  KwsDetection,
  KwsInitOptions,
} from './types';

let kwsInstanceCounter = 0;
let kwsStreamCounter = 0;

function normalizeDetection(raw: {
  keyword?: string;
  tokens?: string[] | unknown;
  timestamps?: number[] | unknown;
  detected?: boolean;
}): KwsDetection {
  const keyword = typeof raw.keyword === 'string' ? raw.keyword : '';
  return {
    keyword,
    tokens: Array.isArray(raw.tokens) ? (raw.tokens as string[]) : [],
    timestamps: Array.isArray(raw.timestamps)
      ? (raw.timestamps as number[])
      : [],
    detected: raw.detected === true || keyword.length > 0,
  };
}

/**
 * Create an open-vocabulary KeywordSpotter (sherpa-onnx KWS engine) with one
 * live stream attached. Android only for now; iOS native implementation pending.
 *
 * ```ts
 * const kws = await createKeywordSpotter({
 *   modelDir: '/data/user/0/.../files/models/kws',
 *   keywordsFile: '/data/user/0/.../files/models/kws/keywords.txt',
 * });
 * const result = await kws.processChunk(pcmFloats, 16000);
 * if (result.detected) console.log('keyword:', result.keyword);
 * ```
 */
export async function createKeywordSpotter(
  options: KwsInitOptions
): Promise<KeywordSpotterEngine> {
  const instanceId = `kws_${++kwsInstanceCounter}_${Date.now()}`;
  const streamId = `kws_stream_${++kwsStreamCounter}_${Date.now()}`;

  const initResult = await SherpaOnnx.initializeKwsWithOptions(instanceId, {
    modelDir: options.modelDir,
    keywordsFile: options.keywordsFile,
    keywordsScore: options.keywordsScore,
    keywordsThreshold: options.keywordsThreshold,
    maxActivePaths: options.maxActivePaths,
    numTrailingBlanks: options.numTrailingBlanks,
    numThreads: options.numThreads,
    provider: options.provider,
    debug: options.debug,
  });
  if (!initResult || initResult.success !== true) {
    throw new Error(
      `KWS init failed: ${initResult?.error ?? 'unknown native error'}`
    );
  }

  try {
    await SherpaOnnx.createKwsStream(instanceId, streamId, undefined);
  } catch (e) {
    await SherpaOnnx.unloadKws(instanceId).catch(() => {});
    throw e;
  }

  let released = false;

  return {
    async processChunk(
      samples: number[],
      sampleRate: number
    ): Promise<KwsDetection> {
      const raw = await SherpaOnnx.processKwsAudioChunk(
        streamId,
        samples,
        sampleRate
      );
      return normalizeDetection(raw ?? {});
    },
    async reset(): Promise<void> {
      await SherpaOnnx.resetKwsStream(streamId);
    },
    async release(): Promise<void> {
      if (released) return;
      released = true;
      await SherpaOnnx.releaseKwsStream(streamId).catch(() => {});
      await SherpaOnnx.unloadKws(instanceId).catch(() => {});
    },
  };
}
