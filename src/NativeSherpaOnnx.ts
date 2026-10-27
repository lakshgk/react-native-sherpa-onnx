import { TurboModuleRegistry, type TurboModule } from 'react-native';

/** Unified shape for all acceleration backends (QNN, NNAPI, XNNPACK, Core ML). */
export type AccelerationSupport = {
  providerCompiled: boolean;
  hasAccelerator: boolean;
  canInit: boolean;
};

export interface Spec extends TurboModule {
  /**
   * Test method to verify sherpa-onnx native library is loaded.
   */
  testSherpaInit(): Promise<string>;

  // ==================== STT Methods ====================

  /**
   * Initialize Speech-to-Text (STT) with model directory.
   * Expects an absolute path (use resolveModelPath first for asset/file paths).
   * @param instanceId - Unique ID for this engine instance (from createSTT)
   * @param modelDir - Absolute path to model directory
   * @param preferInt8 - Optional: true = prefer int8 models, false = prefer regular models, undefined = try int8 first (default)
   * @param modelType - Optional: explicit model type ('transducer', 'nemo_transducer', 'paraformer', 'nemo_ctc', 'wenet_ctc', 'sense_voice', 'zipformer_ctc', 'whisper', 'funasr_nano', 'qwen3_asr', 'fire_red_asr', 'moonshine', 'moonshine_v2', 'dolphin', 'canary', 'omnilingual', 'medasr', 'telespeech_ctc', 'auto'), undefined = auto (default)
   * @param debug - Optional: enable debug logging in native layer and sherpa-onnx (default: false)
   * @param hotwordsFile - Optional: path to hotwords file (OfflineRecognizerConfig)
   * @param hotwordsScore - Optional: hotwords score (default in Kotlin 1.5)
   * @param numThreads - Optional: number of threads for inference (default in Kotlin: 1)
   * @param provider - Optional: provider string e.g. 'cpu' (stored in config only)
   * @param ruleFsts - Optional: path(s) to rule FSTs for ITN (comma-separated)
   * @param ruleFars - Optional: path(s) to rule FARs for ITN (comma-separated)
   * @param dither - Optional: dither for feature extraction. **Android:** applied. **iOS:** ignored (native API does not expose it)
   * @param modelOptions - Optional: model-specific options (whisper, senseVoice, canary, funasrNano, qwen3Asr). Only the block for the loaded model type is applied.
   * @param modelingUnit - Optional: 'cjkchar' | 'bpe' | 'cjkchar+bpe' for hotwords tokenization (OfflineModelConfig.modelingUnit)
   * @param bpeVocab - Optional: path to BPE vocab file (OfflineModelConfig.bpeVocab), used when modelingUnit is bpe or cjkchar+bpe
   * @returns Object with success boolean, array of detected models (each with type and modelDir), and optional error when success is false.
   */
  initializeStt(
    instanceId: string,
    modelDir: string,
    preferInt8?: boolean,
    modelType?: string,
    debug?: boolean,
    hotwordsFile?: string,
    hotwordsScore?: number,
    numThreads?: number,
    provider?: string,
    ruleFsts?: string,
    ruleFars?: string,
    dither?: number,
    modelOptions?: Object,
    modelingUnit?: string,
    bpeVocab?: string
  ): Promise<{
    success: boolean;
    /** Present when success is false (native structured failure). */
    error?: string;
    detectedModels: Array<{ type: string; modelDir: string }>;
    modelType?: string;
    decodingMethod?: string;
  }>;

  /**
   * Detect STT model type and structure without initializing the recognizer.
   * Uses the same native file-based detection as initializeStt. Useful to show model-specific
   * options before init or to query the type for a given path.
   * @param modelDir - Absolute path to model directory (use resolveModelPath first for asset/file paths)
   * @param preferInt8 - Optional: true = prefer int8, false = prefer regular, undefined = try int8 first
   * @param modelType - Optional: explicit type or 'auto' (default)
   * @returns Object with success, detectedModels (array of { type, modelDir }), modelType (primary detected type), and optionally isHardwareSpecificUnsupported (true when the model is for unsupported hardware e.g. RK35xx, Ascend)
   */
  detectSttModel(
    modelDir: string,
    preferInt8?: boolean,
    modelType?: string
  ): Promise<{
    success: boolean;
    /** Present when success is false (or native included a message). */
    error?: string;
    /** True when detection failed because the model targets unsupported hardware (RK35xx, Ascend, CANN). Use to show a specific message or block init. */
    isHardwareSpecificUnsupported?: boolean;
    detectedModels: Array<{ type: string; modelDir: string }>;
    modelType?: string;
  }>;

  /**
   * Transcribe an audio file. Returns full recognition result (text, tokens, timestamps, lang, emotion, event, durations).
   */
  transcribeFile(
    instanceId: string,
    filePath: string
  ): Promise<{
    text: string;
    tokens: string[];
    timestamps: number[];
    lang: string;
    emotion: string;
    event: string;
    durations: number[];
  }>;

  /**
   * Transcribe from float PCM samples (e.g. from microphone). Same return type as transcribeFile.
   */
  transcribeSamples(
    instanceId: string,
    samples: number[],
    sampleRate: number
  ): Promise<{
    text: string;
    tokens: string[];
    timestamps: number[];
    lang: string;
    emotion: string;
    event: string;
    durations: number[];
  }>;

  /**
   * Update recognizer config at runtime (decodingMethod, maxActivePaths, hotwordsFile, hotwordsScore, blankPenalty, ruleFsts, ruleFars).
   */
  setSttConfig(instanceId: string, options: Object): Promise<void>;

  /**
   * Release STT resources.
   */
  unloadStt(instanceId: string): Promise<void>;

  // ==================== Online (streaming) STT Methods ====================

  /**
   * Initialize OnlineRecognizer for streaming STT (single options object to avoid iOS TurboModule marshalling crash with many args).
   * @param instanceId - Unique ID for this engine instance (from createStreamingSTT)
   * @param options - All init options (modelDir, modelType, enableEndpoint, decodingMethod, maxActivePaths, and optional endpoint/rule params).
   *   `options.dither`: **Android** only; **iOS** ignores it (native `FeatureConfig` has no dither field).
   * @returns `{ success: true }` on success, or `{ success: false, error?: string }` on structured native failure.
   */
  initializeOnlineSttWithOptions(
    instanceId: string,
    options: {
      modelDir: string;
      modelType: string;
      enableEndpoint?: boolean;
      decodingMethod?: string;
      maxActivePaths?: number;
      hotwordsFile?: string;
      hotwordsScore?: number;
      numThreads?: number;
      provider?: string;
      ruleFsts?: string;
      ruleFars?: string;
      /** Feature dither. Android: applied. iOS: ignored. */
      dither?: number;
      blankPenalty?: number;
      debug?: boolean;
      rule1MustContainNonSilence?: boolean;
      rule1MinTrailingSilence?: number;
      rule1MinUtteranceLength?: number;
      rule2MustContainNonSilence?: boolean;
      rule2MinTrailingSilence?: number;
      rule2MinUtteranceLength?: number;
      rule3MustContainNonSilence?: boolean;
      rule3MinTrailingSilence?: number;
      rule3MinUtteranceLength?: number;
    }
  ): Promise<{ success: boolean; error?: string }>;

  /** Create a new stream for the given OnlineRecognizer instance. */
  createSttStream(
    instanceId: string,
    streamId: string,
    hotwords?: string
  ): Promise<void>;

  /** Feed PCM samples to a streaming STT stream. */
  acceptSttWaveform(
    streamId: string,
    samples: number[],
    sampleRate: number
  ): Promise<void>;

  /** Signal end of input for a streaming STT stream. */
  sttStreamInputFinished(streamId: string): Promise<void>;

  /** Run decoding on the stream (call when isSttStreamReady is true). */
  decodeSttStream(streamId: string): Promise<void>;

  /** True if the stream has enough audio to decode. */
  isSttStreamReady(streamId: string): Promise<boolean>;

  /** Get current partial or final result (call after decodeSttStream). */
  getSttStreamResult(streamId: string): Promise<{
    text: string;
    tokens: string[];
    timestamps: number[];
  }>;

  /** True if endpoint (end of utterance) was detected. */
  isSttStreamEndpoint(streamId: string): Promise<boolean>;

  /** Reset stream state for reuse. */
  resetSttStream(streamId: string): Promise<void>;

  /** Release stream and remove from native state. */
  releaseSttStream(streamId: string): Promise<void>;

  /** Release OnlineRecognizer and all its streams. */
  unloadOnlineStt(instanceId: string): Promise<void>;

  /**
   * Convenience: feed audio, decode while ready, return result and endpoint status in one call.
   */
  processSttAudioChunk(
    streamId: string,
    samples: number[],
    sampleRate: number
  ): Promise<{
    text: string;
    tokens: string[];
    timestamps: number[];
    isEndpoint: boolean;
  }>;

  // ==================== Keyword Spotting (KWS) Methods ====================

  /**
   * Initialize a sherpa-onnx open-vocabulary KeywordSpotter (single options object,
   * mirroring initializeOnlineSttWithOptions).
   * Android only for now; iOS implementation pending.
   * @param instanceId - Unique ID for this spotter instance
   * @param options - modelDir (zipformer-transducer KWS model: encoder/decoder/joiner .onnx
   *   + tokens.txt), keywordsFile (sherpa-onnx tokenized keywords format), and optional
   *   tuning params (keywordsScore, keywordsThreshold, maxActivePaths, numTrailingBlanks,
   *   numThreads, provider, debug).
   */
  initializeKwsWithOptions(
    instanceId: string,
    options: {
      modelDir: string;
      keywordsFile: string;
      keywordsScore?: number;
      keywordsThreshold?: number;
      maxActivePaths?: number;
      numTrailingBlanks?: number;
      numThreads?: number;
      provider?: string;
      debug?: boolean;
      /** Online transducer variant. KWS zipformer packages are zipformer2 exports; defaults to 'zipformer2'. */
      modelType?: string;
    }
  ): Promise<{ success: boolean; error?: string }>;

  /** Create a new stream for the given KeywordSpotter instance. Optional per-stream keywords override (tokenized format). */
  createKwsStream(
    instanceId: string,
    streamId: string,
    keywords?: string
  ): Promise<void>;

  /**
   * Convenience: feed audio, decode while ready, return spotting result. `detected` is true
   * when `keyword` is non-empty; the stream is reset natively after a detection so it is
   * immediately re-armed.
   */
  processKwsAudioChunk(
    streamId: string,
    samples: number[],
    sampleRate: number
  ): Promise<{
    keyword: string;
    tokens: string[];
    timestamps: number[];
    detected: boolean;
  }>;

  /** Reset KWS stream state. */
  resetKwsStream(streamId: string): Promise<void>;

  /** Release KWS stream and remove from native state. */
  releaseKwsStream(streamId: string): Promise<void>;

  /** Release KeywordSpotter and all its streams. */
  unloadKws(instanceId: string): Promise<void>;

  /**
   * Start native PCM live capture. Microphone audio is captured and resampled to the requested
   * sampleRate; chunks are emitted via the "pcmLiveStreamData" event (base64 Int16 PCM).
   * App must have RECORD_AUDIO (Android) and NSMicrophoneUsageDescription (iOS) and grant permission before calling.
   */
  startPcmLiveStream(options: {
    sampleRate: number;
    channelCount?: number;
    bufferSizeFrames?: number;
  }): Promise<void>;

  /** Stop native PCM live capture. */
  stopPcmLiveStream(): Promise<void>;

  // ==================== Wake Word (openWakeWord) Methods ====================

  /**
   * Initialize the openWakeWord three-stage detector (melspectrogram → embedding →
   * classifier), running on the ONNX Runtime bundled with this SDK. All paths are
   * absolute filesystem paths (a leading file:// is tolerated and stripped).
   * `threshold` is the signed wake_word_model detection threshold and is
   * REQUIRED — the native layer supplies no default (DL-058: threshold-only
   * gating; there is no frame-count parameter). Android only; iOS pending.
   */
  initializeWakeWord(options: {
    melspectrogramPath: string;
    embeddingPath: string;
    classifierPath: string;
    threshold: number;
  }): Promise<{ success: boolean }>;

  /**
   * Start wake-word detection, fed by a native tee of the PCM live stream
   * (single mic client; only produces events while a PCM live stream is
   * running). Emits a "wakeWordDetection" event ({ verb: null, confidence,
   * timestamp }) for EVERY 80 ms inference frame scoring ≥ threshold — raw
   * and undeduplicated; consumers own dedup against the native `timestamp`
   * (DL-065). Requires initializeWakeWord first.
   */
  startWakeWordDetection(): Promise<void>;

  /** Stop wake-word detection (detector stays initialized). */
  stopWakeWordDetection(): Promise<void>;

  /** Release the wake-word ONNX sessions and streaming state. */
  unloadWakeWord(): Promise<void>;

  // ==================== VAD (Silero) Methods ====================

  /**
   * Initialize the Silero VAD over the sherpa-onnx native VoiceActivityDetector.
   * `modelPath` is an absolute filesystem path to silero_vad.onnx (a leading
   * file:// is tolerated and stripped); the k2-fsa v4 export is expected.
   *
   * ⚠ `sampleRate` MUST be 16000 and `windowSize` MUST be 512 — the native
   * layer calls exit(-1) (process death, no exception) on other values. Both
   * are validated in Kotlin before the native object is constructed.
   *
   * ⚠ `maxSpeechDuration` is NOT a segment cut. Upstream it is a soft
   * pressure valve: past it, minSilenceDuration drops to 0.1s and threshold
   * rises to 0.90 to hunt for a break, but a speaker who never pauses never
   * closes a segment and the internal buffer grows without bound. Use
   * `maxSegmentMs` for a real cap.
   *
   * @param maxSegmentMs App-side hard cap in ms; the segment is force-flushed
   *   (and tagged forcedByCap) once it has been open this long. 0 disables
   *   the cap — uncapped is only appropriate when deliberately observing the
   *   unbounded-growth behaviour. Android only; iOS pending.
   */
  initializeVad(options: {
    modelPath: string;
    threshold?: number;
    minSilenceDuration?: number;
    minSpeechDuration?: number;
    maxSpeechDuration?: number;
    windowSize?: number;
    sampleRate?: number;
    maxSegmentMs?: number;
    debug?: boolean;
  }): Promise<{ success: boolean }>;

  /**
   * Start VAD segmentation, fed by a native tee of the PCM live stream (the
   * same single mic client the wake-word detector shares; only produces
   * events while a PCM live stream is running). Emits a "vadSpeechSegment"
   * event per completed speech segment: { base64Pcm, startSample,
   * sampleCount, viaFlush, forcedByCap, latencyMs }. Requires initializeVad.
   */
  startVadDetection(): Promise<void>;

  /**
   * Stop VAD segmentation (VAD stays initialized). Flushes any in-progress
   * segment first — a flushed segment is emitted with viaFlush: true and,
   * unlike a naturally-closed one, retains its trailing silence.
   */
  stopVadDetection(): Promise<void>;

  /** Release the native VAD. */
  unloadVad(): Promise<void>;

  // ==================== Alignment / Subtitle Methods ====================

  /**
   * Run wav2vec2 CTC forced alignment on an audio file and transcript.
   * @param modelPath - Absolute path to wav2vec2 ONNX model file
   * @param audioPath - Absolute path to input audio file (WAV recommended)
   * @param text - Transcript to align
   * @param vocabJson - JSON map of token -> id (stringified to reduce bridge overhead)
   */
  runCTCForcedAlignment(
    modelPath: string,
    audioPath: string,
    text: string,
    vocabJson: string
  ): Promise<{
    words: Array<{ text: string; start: number; end: number }>;
    chars: Array<{ text: string; start: number; end: number }>;
  }>;

  detectAlignmentModel(
    modelDir: string,
    modelType?: string
  ): Promise<{
    success: boolean;
    error?: string;
    detectedModels: Array<{ type: string; modelDir: string }>;
    modelType?: string;
    paths?: {
      model?: string;
    };
  }>;

  // ==================== Kokoro on ONNX Runtime (Smriti GP-2026-072) ====================
  // Route B (DL-129): sherpa-onnx is built without TTS, so espeak-ng is not in the binaries;
  // Kokoro v1.0 runs as a raw ONNX Runtime session (SherpaOnnxKokoroOrtHelper.kt).

  /** Create a Kokoro ONNX Runtime session (CPU, `numThreads` intra-op threads); resolves a handle. */
  createKokoroSession(modelPath: string, numThreads: number): Promise<string>;

  /**
   * Run one unit: token ids (with the 0 pads), the 256-float style row, and `speed`.
   * Resolves the waveform as little-endian float32, base64-encoded, with its sample count.
   */
  runKokoro(
    handle: string,
    tokenIds: number[],
    style: number[],
    speed: number
  ): Promise<{ base64: string; samples: number; runMs: number }>;

  /** Release a Kokoro session (idempotent). */
  releaseKokoroSession(handle: string): Promise<void>;

  // ==================== Speech Enhancement Methods ====================

  detectEnhancementModel(
    modelDir: string,
    modelType?: string
  ): Promise<{
    success: boolean;
    error?: string;
    detectedModels: Array<{ type: string; modelDir: string }>;
    modelType?: string;
  }>;

  initializeEnhancement(
    instanceId: string,
    modelDir: string,
    modelType?: string,
    numThreads?: number,
    provider?: string,
    debug?: boolean
  ): Promise<{
    success: boolean;
    error?: string;
    detectedModels: Array<{ type: string; modelDir: string }>;
    modelType?: string;
    sampleRate?: number;
  }>;

  enhanceFile(
    instanceId: string,
    inputPath: string,
    outputPath?: string
  ): Promise<{ samples: number[]; sampleRate: number }>;

  enhanceSamples(
    instanceId: string,
    samples: number[],
    sampleRate: number
  ): Promise<{ samples: number[]; sampleRate: number }>;

  getEnhancementSampleRate(instanceId: string): Promise<number>;

  unloadEnhancement(instanceId: string): Promise<void>;

  initializeOnlineEnhancement(
    instanceId: string,
    modelDir: string,
    modelType?: string,
    numThreads?: number,
    provider?: string,
    debug?: boolean
  ): Promise<{
    success: boolean;
    error?: string;
    sampleRate?: number;
    frameShiftInSamples?: number;
  }>;

  feedEnhancementSamples(
    instanceId: string,
    samples: number[],
    sampleRate: number
  ): Promise<{ samples: number[]; sampleRate: number }>;

  flushOnlineEnhancement(
    instanceId: string
  ): Promise<{ samples: number[]; sampleRate: number }>;

  resetOnlineEnhancement(instanceId: string): Promise<void>;

  unloadOnlineEnhancement(instanceId: string): Promise<void>;

  // ==================== Helper - Assets ====================

  /**
   * Resolve model path based on configuration.
   * Handles asset paths, file system paths, and auto-detection.
   * Returns an absolute path that can be used by native code.
   *
   * @param config - Object with 'type' ('asset' | 'file' | 'auto') and 'path' (string)
   */
  resolveModelPath(config: { type: string; path: string }): Promise<string>;

  /**
   * List all model folders in the assets/models directory.
   * Scans the platform-specific model directory and returns folder names.
   *
   * @returns Array of model info objects found in assets/models/ (Android) or bundle models/ (iOS)
   *
   * @example
   * ```typescript
   * const folders = await listAssetModels();
   * // Returns: [{ folder: 'sherpa-onnx-streaming-zipformer-en-2023-06-26', hint: 'stt' }, { folder: 'sherpa-onnx-matcha-icefall-en_US-ljspeech', hint: 'tts' }]
   *
   * // Then use with resolveModelPath and initialize:
   * for (const model of folders) {
   *   const path = await resolveModelPath({ type: 'asset', path: `models/${model.folder}` });
   *   const result = await initializeStt(path);
   *   if (result.success) {
   *     console.log(`Found models in ${model.folder}:`, result.detectedModels);
   *   }
   * }
   * ```
   */
  listAssetModels(): Promise<
    Array<{ folder: string; hint: 'stt' | 'tts' | 'unknown' }>
  >;

  /**
   * List model folders under a specific filesystem path.
   * When recursive is true, returns relative folder paths under the base path.
   */
  listModelsAtPath(
    path: string,
    recursive: boolean
  ): Promise<Array<{ folder: string; hint: 'stt' | 'tts' | 'unknown' }>>;

  /**
   * **Play Asset Delivery (PAD):** Returns the filesystem path to the models directory
   * of an Android asset pack, or null if the pack is not available (e.g. not installed).
   * Use this to list and load models that are delivered via PAD instead of bundled app assets.
   */
  getAssetPackPath(packName: string): Promise<string | null>;

  /**
   * Read the contents of a text file from the bundled assets (Android) or main bundle (iOS).
   * @param assetPath The relative path to the asset file (e.g., 'model_licenses/asr-models-license-status.csv')
   * @returns Resolves with the string content of the file or rejects if the file cannot be read.
   */
  readAssetFileAsUtf8(assetPath: string): Promise<string>;

  // ==================== Helper - Extraction ====================

  /**
   * Extract a .tar.bz2 archive to a target folder.
   * Returns { success, path } or { success, reason }.
   *
   * **Android:** When `showNotificationsEnabled` is true (default), a system notification shows
   * extraction progress. Optional `notificationTitle` / `notificationText` customize the copy.
   * **iOS:** Notification parameters are accepted but have no effect (no extraction progress notification).
   */
  extractTarBz2(
    sourcePath: string,
    targetPath: string,
    force: boolean,
    showNotificationsEnabled?: boolean,
    notificationTitle?: string,
    notificationText?: string
  ): Promise<{
    success: boolean;
    path?: string;
    sha256?: string;
    reason?: string;
  }>;

  /**
   * Cancel any in-progress tar.bz2 extraction.
   */
  cancelExtractTarBz2(): Promise<void>;

  /**
   * Extract a .tar.zst (or .zst) archive to a target folder.
   * Returns { success, path } or { success, reason }.
   *
   * **Android:** Same notification behavior as `extractTarBz2`. **iOS:** No effect.
   */
  extractTarZst(
    sourcePath: string,
    targetPath: string,
    force: boolean,
    showNotificationsEnabled?: boolean,
    notificationTitle?: string,
    notificationText?: string
  ): Promise<{
    success: boolean;
    path?: string;
    sha256?: string;
    reason?: string;
  }>;

  /**
   * Cancel any in-progress tar.zst extraction.
   */
  cancelExtractTarZst(): Promise<void>;

  /**
   * Cancel extraction for a specific source archive path (per-operation cancel for parallel extractions).
   */
  cancelExtractBySourcePath(sourcePath: string): Promise<void>;

  /**
   * List asset paths of .tar.zst and .tar.bz2 archives in a PAD pack when stored as APK_ASSETS.
   * Android only; returns [] when pack is not available or not APK_ASSETS. Used by getBundledArchives.
   */
  listBundledArchiveAssetPaths(packName: string): Promise<string[]>;

  /**
   * Extract a .tar.zst archive from Android assets (AssetManager) to a target folder. Android only.
   * Streams from asset; no copy of the archive to disk. Used when PAD pack is APK_ASSETS.
   * Notification options match `extractTarZst` (Android only).
   */
  extractTarZstFromAsset(
    assetPath: string,
    targetPath: string,
    force: boolean,
    showNotificationsEnabled?: boolean,
    notificationTitle?: string,
    notificationText?: string
  ): Promise<{
    success: boolean;
    path?: string;
    sha256?: string;
    reason?: string;
  }>;

  /**
   * Extract a .tar.bz2 archive from Android assets (AssetManager) to a target folder. Android only.
   * Streams from asset; no copy of the archive to disk. Used when PAD pack is APK_ASSETS.
   * Notification options match `extractTarBz2` (Android only).
   */
  extractTarBz2FromAsset(
    assetPath: string,
    targetPath: string,
    force: boolean,
    showNotificationsEnabled?: boolean,
    notificationTitle?: string,
    notificationText?: string
  ): Promise<{
    success: boolean;
    path?: string;
    sha256?: string;
    reason?: string;
  }>;

  /**
   * Compute SHA-256 of a file and return the hex digest.
   */
  computeFileSha256(filePath: string): Promise<string>;

  // ==================== Helper - Audio conversion ====================

  /**
   * Convert arbitrary audio file to requested format (e.g. "mp3", "flac", "wav").
   * Requires FFmpeg prebuilts when called on Android.
   * For MP3 (libshine), outputSampleRateHz can be 32000, 44100, or 48000; 0 or omitted = 44100.
   * WAV output is always 16 kHz mono (sherpa-onnx). Resolves when conversion succeeds, rejects with an error message on failure.
   */
  convertAudioToFormat(
    inputPath: string,
    outputPath: string,
    format: string,
    outputSampleRateHz?: number
  ): Promise<void>;

  /**
   * Convert any supported audio file to WAV 16 kHz mono 16-bit PCM.
   * Requires FFmpeg prebuilts when called on Android.
   */
  convertAudioToWav16k(inputPath: string, outputPath: string): Promise<void>;

  /**
   * Decode an audio file to mono float samples in [-1, 1] and the effective sample rate.
   * Supports the same inputs as convertAudioToFormat (file paths and Android content:// URIs).
   * On Android, non-WAV formats require FFmpeg prebuilts; WAV may use a fast path via WaveReader.
   * @param targetSampleRateHz - If > 0, resample to this rate; if 0 or omitted, keep the decoded stream rate.
   */
  decodeAudioFileToFloatSamples(
    inputPath: string,
    targetSampleRateHz?: number
  ): Promise<{ samples: number[]; sampleRate: number }>;

  // ==================== Execution Provider Methods ====================

  /**
   * Return the list of available ONNX Runtime execution providers (e.g. "CPU", "NNAPI", "QNN", "XNNPACK").
   * Requires the ORT Java bridge (libonnxruntime4j_jni.so + OrtEnvironment class) from the onnxruntime AAR.
   */
  getAvailableProviders(): Promise<string[]>;

  // ==================== Acceleration support (unified format) ====================

  /**
   * Unified acceleration support: providerCompiled (ORT EP built in), hasAccelerator (NPU/ANE present), canInit (session with EP works).
   * All get*Support methods return this shape. Optional modelBase64: if omitted, SDK uses embedded test model for canInit.
   */
  getQnnSupport(modelBase64?: string): Promise<AccelerationSupport>;
  /** Device SoC model string (e.g. SM8850 on Android 12+). Null if not available. isSupported: true when SoC is SM8xxx (supported for QNN). */
  getDeviceQnnSoc(): Promise<{ soc: string | null; isSupported: boolean }>;
  getNnapiSupport(modelBase64?: string): Promise<AccelerationSupport>;
  getXnnpackSupport(modelBase64?: string): Promise<AccelerationSupport>;
  getCoreMlSupport(modelBase64?: string): Promise<AccelerationSupport>;
}

export default TurboModuleRegistry.getEnforcing<Spec>('SherpaOnnx');
