/**
 * KeywordSpotterEngine.renewStream() (Smriti GP-2026-021 item 5, ruled 2026-09-21):
 * create -> swap -> release, and a failed create leaves the old stream in use.
 *
 * The native module is faked. Its stream bookkeeping mirrors SherpaOnnxKwsHelper.kt
 * at b41daf5, so "the old stream still works" is tested against the same rules
 * the real helper enforces rather than asserted:
 *   - decoding an unknown stream rejects "Stream not found"      (Kt:249-253)
 *   - creating a stream id that already exists rejects            (Kt:203-206)
 *   - releasing an unknown stream resolves, no error              (Kt:307-310)
 * This covers wrapper plumbing only. Whether a renewed stream decodes like a
 * fresh one is a native question, answered on device by the carry-over probe.
 */
// `mock`-prefixed: jest hoists jest.mock() above these declarations and only
// lets the factory close over names with that prefix.
const mockState = {
  calls: [] as string[],
  live: new Set<string>(),
  failNextCreate: false,
  failNextRelease: false,
};
const calls = mockState.calls;
const live = mockState.live;

jest.mock('../NativeSherpaOnnx', () => ({
  __esModule: true,
  default: {
    initializeKwsWithOptions: jest.fn(async () => ({ success: true })),
    createKwsStream: jest.fn(async (instanceId: string, streamId: string, keywords: unknown) => {
      mockState.calls.push(`create:${instanceId}:${streamId}:${String(keywords)}`);
      if (mockState.failNextCreate) {
        mockState.failNextCreate = false;
        throw new Error('Create KWS stream failed: injected');
      }
      if (mockState.live.has(streamId)) throw new Error(`Stream already exists: ${streamId}`);
      mockState.live.add(streamId);
    }),
    processKwsAudioChunk: jest.fn(async (streamId: string) => {
      mockState.calls.push(`process:${streamId}`);
      if (!mockState.live.has(streamId)) throw new Error(`Stream not found: ${streamId}`);
      return { keyword: '', tokens: [], timestamps: [] };
    }),
    resetKwsStream: jest.fn(async (streamId: string) => {
      mockState.calls.push(`reset:${streamId}`);
      if (!mockState.live.has(streamId)) throw new Error(`Stream not found: ${streamId}`);
    }),
    releaseKwsStream: jest.fn(async (streamId: string) => {
      mockState.calls.push(`release:${streamId}`);
      if (mockState.failNextRelease) {
        mockState.failNextRelease = false;
        throw new Error('releaseKwsStream failed: injected');
      }
      mockState.live.delete(streamId);
    }),
    unloadKws: jest.fn(async (instanceId: string) => {
      mockState.calls.push(`unload:${instanceId}`);
    }),
  },
}));

import { createKeywordSpotter } from '../kws';

const opts = { modelDir: '/m', keywordsFile: '/m/kw.txt' };
const idOf = (entry: string | undefined): string => (entry ?? '').split(':')[2] ?? '';

beforeEach(() => {
  calls.length = 0;
  live.clear();
  mockState.failNextCreate = false;
  mockState.failNextRelease = false;
});

describe('renewStream', () => {
  it('creates the new stream before releasing the old one', async () => {
    const kws = await createKeywordSpotter(opts);
    const oldId = idOf(calls[0]);
    calls.length = 0;
    await kws.renewStream();
    expect(calls).toHaveLength(2);
    expect(calls[0]).toMatch(/^create:/);
    expect(calls[1]).toBe(`release:${oldId}`);
  });

  it('mints a new id on the same instance and passes no keywords', async () => {
    const kws = await createKeywordSpotter(opts);
    const [, initInstance, oldId] = (calls[0] ?? '').split(':');
    calls.length = 0;
    await kws.renewStream();
    const [, instance, newId, keywords] = (calls[0] ?? '').split(':');
    expect(instance).toBe(initInstance);
    expect(newId).not.toBe(oldId);
    expect(keywords).toBe('undefined');
  });

  it('routes processChunk, reset and release to the new stream afterwards', async () => {
    const kws = await createKeywordSpotter(opts);
    const oldId = idOf(calls[0]);
    calls.length = 0;
    await kws.renewStream();
    const newId = idOf(calls[0]);
    calls.length = 0;
    await kws.processChunk([0, 0], 16000);
    await kws.reset();
    await kws.release();
    expect(calls).toEqual([`process:${newId}`, `reset:${newId}`, `release:${newId}`, expect.stringMatching(/^unload:/)]);
    expect(live.has(oldId)).toBe(false);
  });

  it('on a failed create, keeps the old stream in use, unreleased and working', async () => {
    const kws = await createKeywordSpotter(opts);
    const oldId = idOf(calls[0]);
    mockState.failNextCreate = true;
    calls.length = 0;
    await expect(kws.renewStream()).rejects.toThrow('injected');
    expect(calls.filter((c) => c.startsWith('release:'))).toEqual([]);
    expect(live.has(oldId)).toBe(true);
    await expect(kws.processChunk([0, 0], 16000)).resolves.toMatchObject({ detected: false });
    expect(calls[calls.length - 1]).toBe(`process:${oldId}`);
  });

  it('still succeeds, on the new stream, when releasing the old one fails', async () => {
    const kws = await createKeywordSpotter(opts);
    mockState.failNextRelease = true;
    calls.length = 0;
    await expect(kws.renewStream()).resolves.toBeUndefined();
    const newId = idOf(calls[0]);
    await kws.processChunk([0, 0], 16000);
    expect(calls[calls.length - 1]).toBe(`process:${newId}`);
  });

  it('rejects after release() without creating a stream', async () => {
    const kws = await createKeywordSpotter(opts);
    await kws.release();
    calls.length = 0;
    await expect(kws.renewStream()).rejects.toThrow('after release');
    expect(calls).toEqual([]);
  });
});
