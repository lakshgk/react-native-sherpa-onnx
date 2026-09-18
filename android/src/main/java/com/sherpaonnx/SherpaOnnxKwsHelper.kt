package com.sherpaonnx

import android.content.Context
import android.net.Uri
import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.WritableMap
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.KeywordSpotter
import com.k2fsa.sherpa.onnx.KeywordSpotterConfig
import com.k2fsa.sherpa.onnx.KeywordSpotterResult
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit

/**
 * Helper for open-vocabulary keyword spotting using sherpa-onnx KeywordSpotter + OnlineStream.
 * Mirrors SherpaOnnxOnlineSttHelper: manages spotter instances and streams; resolves model
 * paths by scanning the model directory (KWS models are zipformer transducers:
 * encoder/decoder/joiner .onnx + tokens.txt).
 *
 * The keywords file must be in sherpa-onnx tokenized keywords format (one keyword per line,
 * space-separated model tokens, optional :boost / #threshold / @ORIGINAL suffixes) — see
 * https://k2-fsa.github.io/sherpa/onnx/kws/index.html. Raw-text keyword phrases are NOT
 * tokenized natively here.
 */
internal class SherpaOnnxKwsHelper(
  private val context: Context,
  private val logTag: String
) {

  private data class KwsInstance(
    val spotter: KeywordSpotter,
    val config: KeywordSpotterConfig,
    val streams: MutableMap<String, OnlineStream> = mutableMapOf()
  )

  private val instances = ConcurrentHashMap<String, KwsInstance>()

  /**
   * Every KWS operation runs on this one thread (Smriti GP-2026-021 Prompt 2b, §20).
   *
   * Why: the React Native module methods of this package share one native-modules
   * thread, and SherpaOnnxTtsHelper.writeTtsPcmChunk blocks that thread for a whole
   * ~200 ms AudioTrack slice (WRITE_BLOCKING). A KWS call issued during TTS playback
   * therefore waited behind a write: measured on a Pixel 10 Pro XL, processKwsAudioChunk
   * 204 ms median while speaking vs 18 ms paused, and even resetKwsStream (no decode)
   * 64 ms vs 3 ms. Moving KWS off that thread removes the wait; a SINGLE thread keeps
   * every stream's calls in submission order and means release/unload can never run
   * concurrently with an in-flight decode on the same native stream.
   */
  private val kwsExecutor: ExecutorService =
    Executors.newSingleThreadExecutor { r -> Thread(r, "SherpaOnnxKws") }

  private fun onKwsThread(promise: Promise, what: String, block: () -> Unit) {
    try {
      kwsExecutor.execute(block)
    } catch (e: RejectedExecutionException) {
      promise.reject("STATE_ERROR", "$what after KWS shutdown", e)
    }
  }
  private val streamToInstance = ConcurrentHashMap<String, String>()

  private fun getInstance(instanceId: String): KwsInstance? = instances[instanceId]

  private fun getStream(streamId: String): Pair<KwsInstance, OnlineStream>? {
    val instanceId = streamToInstance[streamId] ?: return null
    val inst = instances[instanceId] ?: return null
    val stream = inst.streams[streamId] ?: return null
    return inst to stream
  }

  private fun resolveContentUriToFile(path: String, cacheFilePrefix: String): String {
    if (!path.startsWith("content://")) return path
    val uri = Uri.parse(path)
    val cacheFile = File(context.cacheDir, "${cacheFilePrefix}_${System.nanoTime()}")
    context.contentResolver.openInputStream(uri)?.use { input ->
      cacheFile.outputStream().use { output -> input.copyTo(output) }
    } ?: throw IllegalStateException("File is not readable (content URI could not be opened): $path")
    return cacheFile.absolutePath
  }

  /**
   * Scan model directory for zipformer-transducer KWS files.
   * Returns encoder, decoder, joiner, tokens absolute paths.
   */
  private fun scanKwsModelPaths(modelDir: String): Map<String, String> {
    val dir = File(modelDir)
    if (!dir.exists() || !dir.isDirectory) {
      throw IllegalArgumentException("KWS model directory does not exist or is not a directory: $modelDir")
    }
    val files = dir.listFiles()?.filter { it.isFile }.orEmpty()

    fun firstFile(prefix: String, suffix: String = ".onnx"): String =
      files.firstOrNull { it.name.startsWith(prefix) && it.name.endsWith(suffix) }?.absolutePath.orEmpty()

    val paths = mapOf(
      "encoder" to firstFile("encoder"),
      "decoder" to firstFile("decoder"),
      "joiner" to firstFile("joiner"),
      "tokens" to (files.firstOrNull { it.name == "tokens.txt" }?.absolutePath ?: "")
    )
    if (paths.getValue("encoder").isEmpty() || paths.getValue("decoder").isEmpty() ||
      paths.getValue("joiner").isEmpty() || paths.getValue("tokens").isEmpty()
    ) {
      throw IllegalArgumentException(
        "KWS model requires encoder, decoder, joiner .onnx files and tokens.txt in $modelDir"
      )
    }
    return paths
  }

  fun initializeKws(
    instanceId: String,
    modelDir: String,
    keywordsFile: String,
    keywordsScore: Double?,
    keywordsThreshold: Double?,
    maxActivePaths: Double?,
    numTrailingBlanks: Double?,
    numThreads: Double?,
    provider: String?,
    debug: Boolean?,
    modelType: String?,
    promise: Promise
  ) = onKwsThread(promise, "initializeKws") {
    initializeKwsNow(
      instanceId, modelDir, keywordsFile, keywordsScore, keywordsThreshold, maxActivePaths,
      numTrailingBlanks, numThreads, provider, debug, modelType, promise
    )
  }

  private fun initializeKwsNow(
    instanceId: String,
    modelDir: String,
    keywordsFile: String,
    keywordsScore: Double?,
    keywordsThreshold: Double?,
    maxActivePaths: Double?,
    numTrailingBlanks: Double?,
    numThreads: Double?,
    provider: String?,
    debug: Boolean?,
    modelType: String?,
    promise: Promise
  ) {
    try {
      val paths = scanKwsModelPaths(modelDir)
      val resolvedKeywordsFile = resolveContentUriToFile(keywordsFile.trim(), "kws_keywords")
      if (!File(resolvedKeywordsFile).isFile) {
        throw IllegalArgumentException("Keywords file does not exist: $resolvedKeywordsFile")
      }
      val config = KeywordSpotterConfig(
        featConfig = FeatureConfig(sampleRate = 16000, featureDim = 80),
        modelConfig = OnlineModelConfig(
          transducer = OnlineTransducerModelConfig(
            encoder = paths.getValue("encoder"),
            decoder = paths.getValue("decoder"),
            joiner = paths.getValue("joiner")
          ),
          tokens = paths.getValue("tokens"),
          numThreads = numThreads?.toInt() ?: 1,
          debug = debug ?: false,
          provider = provider ?: "cpu",
          // KWS zipformer models (kws-zipformer-gigaspeech/wenetspeech) are
          // zipformer2 exports. "zipformer" selects the zipformer1 loader,
          // which fatally exits on missing 'attention_dims' metadata —
          // observed on-device 2026-07-10 (GP-2026-012 S1a first run).
          modelType = modelType ?: "zipformer2"
        ),
        maxActivePaths = maxActivePaths?.toInt() ?: 4,
        keywordsFile = resolvedKeywordsFile,
        keywordsScore = keywordsScore?.toFloat() ?: 1.5f,
        keywordsThreshold = keywordsThreshold?.toFloat() ?: 0.25f,
        numTrailingBlanks = numTrailingBlanks?.toInt() ?: 2
      )
      val spotter = KeywordSpotter(assetManager = null, config = config)
      instances[instanceId] = KwsInstance(spotter = spotter, config = config)
      promise.resolve(Arguments.createMap().apply { putBoolean("success", true) })
    } catch (e: Exception) {
      Log.e(logTag, "initializeKws failed: ${e.message}", e)
      promise.reject("INIT_ERROR", "KWS init failed: ${e.message}", e)
    }
  }

  fun createKwsStream(instanceId: String, streamId: String, keywords: String?, promise: Promise) =
    onKwsThread(promise, "createKwsStream") { createKwsStreamNow(instanceId, streamId, keywords, promise) }

  private fun createKwsStreamNow(instanceId: String, streamId: String, keywords: String?, promise: Promise) {
    try {
      val inst = getInstance(instanceId)
        ?: run {
          promise.reject("STREAM_ERROR", "KWS instance not found: $instanceId")
          return
        }
      if (inst.streams.containsKey(streamId)) {
        promise.reject("STREAM_ERROR", "Stream already exists: $streamId")
        return
      }
      val stream = inst.spotter.createStream(keywords = keywords?.trim().orEmpty())
      inst.streams[streamId] = stream
      streamToInstance[streamId] = instanceId
      promise.resolve(null)
    } catch (e: Exception) {
      Log.e(logTag, "createKwsStream failed: ${e.message}", e)
      promise.reject("STREAM_ERROR", "Create KWS stream failed: ${e.message}", e)
    }
  }

  private fun readableArrayToFloatArray(arr: ReadableArray): FloatArray =
    FloatArray(arr.size()) { i -> arr.getDouble(i).toFloat() }

  private fun resultToWritableMap(result: KeywordSpotterResult): WritableMap {
    val map = Arguments.createMap()
    map.putString("keyword", result.keyword)
    val tokensArray = Arguments.createArray()
    for (t in result.tokens) tokensArray.pushString(t)
    map.putArray("tokens", tokensArray)
    val timestampsArray = Arguments.createArray()
    for (t in result.timestamps) timestampsArray.pushDouble(t.toDouble())
    map.putArray("timestamps", timestampsArray)
    map.putBoolean("detected", result.keyword.isNotEmpty())
    return map
  }

  /**
   * Convenience: accept waveform, decode while ready, return result. After a detection the
   * stream is reset (sherpa-onnx KWS convention — see the upstream Android KWS demo) so the
   * spotter is immediately re-armed for the next keyword.
   */
  fun processKwsAudioChunk(
    streamId: String,
    samples: ReadableArray,
    sampleRate: Int,
    promise: Promise
  ) {
    // Copy on the calling thread: the ReadableArray is not handed across threads.
    val floatSamples = try {
      readableArrayToFloatArray(samples)
    } catch (e: Exception) {
      promise.reject("STREAM_ERROR", "processKwsAudioChunk failed: ${e.message}", e)
      return
    }
    onKwsThread(promise, "processKwsAudioChunk") {
      processKwsAudioChunkNow(streamId, floatSamples, sampleRate, promise)
    }
  }

  private fun processKwsAudioChunkNow(
    streamId: String,
    floatSamples: FloatArray,
    sampleRate: Int,
    promise: Promise
  ) {
    try {
      val (inst, stream) = getStream(streamId)
        ?: run {
          promise.reject("STREAM_ERROR", "Stream not found: $streamId")
          return
        }
      stream.acceptWaveform(floatSamples, sampleRate)
      while (inst.spotter.isReady(stream)) {
        inst.spotter.decode(stream)
      }
      val result = inst.spotter.getResult(stream)
      if (result.keyword.isNotEmpty()) {
        inst.spotter.reset(stream)
      }
      promise.resolve(resultToWritableMap(result))
    } catch (e: Exception) {
      Log.e(logTag, "processKwsAudioChunk failed: ${e.message}", e)
      promise.reject("STREAM_ERROR", "processKwsAudioChunk failed: ${e.message}", e)
    }
  }

  fun resetKwsStream(streamId: String, promise: Promise) =
    onKwsThread(promise, "resetKwsStream") { resetKwsStreamNow(streamId, promise) }

  private fun resetKwsStreamNow(streamId: String, promise: Promise) {
    try {
      val (inst, stream) = getStream(streamId)
        ?: run {
          promise.reject("STREAM_ERROR", "Stream not found: $streamId")
          return
        }
      inst.spotter.reset(stream)
      promise.resolve(null)
    } catch (e: Exception) {
      Log.e(logTag, "resetKwsStream failed: ${e.message}", e)
      promise.reject("STREAM_ERROR", "resetKwsStream failed: ${e.message}", e)
    }
  }

  fun releaseKwsStream(streamId: String, promise: Promise) =
    onKwsThread(promise, "releaseKwsStream") { releaseKwsStreamNow(streamId, promise) }

  private fun releaseKwsStreamNow(streamId: String, promise: Promise) {
    try {
      val instanceId = streamToInstance.remove(streamId) ?: run {
        promise.resolve(null)
        return
      }
      val inst = instances[instanceId] ?: run {
        promise.resolve(null)
        return
      }
      inst.streams.remove(streamId)?.release()
      promise.resolve(null)
    } catch (e: Exception) {
      Log.e(logTag, "releaseKwsStream failed: ${e.message}", e)
      promise.reject("STREAM_ERROR", "releaseKwsStream failed: ${e.message}", e)
    }
  }

  fun unloadKws(instanceId: String, promise: Promise) =
    onKwsThread(promise, "unloadKws") { unloadKwsNow(instanceId, promise) }

  private fun unloadKwsNow(instanceId: String, promise: Promise) {
    try {
      val inst = instances.remove(instanceId) ?: run {
        promise.resolve(null)
        return
      }
      val streamIds = inst.streams.keys.toList()
      inst.streams.values.forEach { it.release() }
      inst.streams.clear()
      streamIds.forEach { streamToInstance.remove(it) }
      inst.spotter.release()
      promise.resolve(null)
    } catch (e: Exception) {
      Log.e(logTag, "unloadKws failed: ${e.message}", e)
      promise.reject("RELEASE_ERROR", "unloadKws failed: ${e.message}", e)
    }
  }

  /** Call from Module.onCatalystInstanceDestroy to release all resources. */
  fun shutdown() {
    // Release on the KWS thread (after any queued decode), then stop the thread.
    try {
      kwsExecutor.submit(Runnable { shutdownNow() }).get(3, TimeUnit.SECONDS)
    } catch (e: Exception) {
      Log.w(logTag, "shutdown: KWS release did not complete on the KWS thread: ${e.message}")
    }
    kwsExecutor.shutdown()
  }

  private fun shutdownNow() {
    instances.keys.toList().forEach { instanceId ->
      try {
        val inst = instances.remove(instanceId) ?: return@forEach
        val streamIds = inst.streams.keys.toList()
        inst.streams.values.forEach { it.release() }
        inst.streams.clear()
        streamIds.forEach { streamToInstance.remove(it) }
        inst.spotter.release()
      } catch (e: Exception) {
        Log.w(logTag, "shutdown: failed to release KWS instance $instanceId: ${e.message}")
      }
    }
    streamToInstance.clear()
  }
}
