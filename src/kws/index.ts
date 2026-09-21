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
  // `let`, not `const`: renewStream() swaps in a fresh stream on the same
  // spotter, and processChunk / reset / release must follow it.
  let streamId = `kws_stream_${++kwsStreamCounter}_${Date.now()}`;

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
    modelType: options.modelType,
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
    async renewStream(): Promise<void> {
      if (released) {
        throw new Error('KWS renewStream after release');
      }
      // Create -> swap -> release (Smriti GP-2026-021 item 5, ruled 2026-09-21).
      // Creating first means a failed create leaves the engine on its old,
      // still-working stream instead of on none. `undefined` keywords fall back
      // to the spotter's own keywordsFile, so the keyword set is unchanged.
      const oldId = streamId;
      const newId = `kws_stream_${++kwsStreamCounter}_${Date.now()}`;
      await SherpaOnnx.createKwsStream(instanceId, newId, undefined);
      streamId = newId;
      // A failed release leaks the old stream until unloadKws, which releases
      // every stream on the instance; not worth failing the renew over.
      await SherpaOnnx.releaseKwsStream(oldId).catch(() => {});
    },
    async release(): Promise<void> {
      if (released) return;
      released = true;
      await SherpaOnnx.releaseKwsStream(streamId).catch(() => {});
      await SherpaOnnx.unloadKws(instanceId).catch(() => {});
    },
  };
}
