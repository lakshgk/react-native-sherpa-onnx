package com.sherpaonnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import java.io.File
import java.util.ArrayDeque
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread
import kotlin.random.Random

/**
 * N8 SPIKE (Smriti C2, GP-2026-015 pre-Generate) — openWakeWord three-stage
 * pipeline (melspectrogram → embedding → classifier) on the ONNX Runtime Java
 * API already bundled by this SDK's prebuilt pipeline (build.gradle:
 * extractOnnxruntimeClasses). Fed by a native-side tee of SherpaOnnxPcmCapture —
 * no second mic client.
 *
 * PORTING SOURCE (Rule 16 — observed behavior, cited):
 * Re-MENTIA/openwakeword-android-kt @ 397333e, Apache-2.0:
 *   - streaming state machine: wakeword/.../lib/audio/AudioProcessor.kt:52-176
 *   - melspec stage + mel transform (x/10 + 2): .../lib/ml/MelSpectrogram.kt:29-76
 *   - embedding stage (input "input_1", [1,76,32,1] → [n,96]): .../lib/ml/EmbeddingModel.kt:27-52
 *   - classifier inference ([1,16,96] → [1,1] score): .../lib/ml/OnnxModelRunner.kt:45-62
 *
 * DELIBERATE DEVIATIONS from the porting source (each observation-backed):
 *   1. INPUT SCALING — samples are fed as int16-range floats (short.toFloat(),
 *      NO /32768 normalization). Re-MENTIA's AudioRecorder.kt:77-78 divides by
 *      32768, but the reference stack that validated candidate-3 rev v7
 *      (dscripka openWakeWord Python, smriti spikes/s0_wakeword s0_validate.py,
 *      S0 PASS 2026-07-16) feeds raw int16-valued floats. The S0 corpus is the
 *      observation; Re-MENTIA's own scaling is the outlier.
 *   2. SESSION LIFETIME — OrtSessions are created once at initialize() and
 *      reused. Re-MENTIA recreates melspec/embedding sessions on every call
 *      (MelSpectrogram.kt:37, EmbeddingModel.kt:34) — behaviorally identical
 *      numerics, prohibitive per-80ms-frame cost on-device.
 *   3. MODEL LOADING — from absolute file paths (this SDK's convention, cf.
 *      SherpaOnnxSttHelper modelPath handling) instead of AssetManager.
 *
 * SPIKE SCOPE: placeholder threshold (default 0.5) for a detected flag only;
 * NO minimum-consecutive-frames gate (DQ-C2-04's signed values are
 * Generate-phase). Raw {score, timestamp} emitted per processed chunk.
 */
internal class OpenWakeWordHelper(
  private val logTag: String,
  private val onScore: (score: Float, timestamp: Long, detected: Boolean) -> Unit
) {

  companion object {
    // Constants ported from AudioProcessor.kt:20-29 (Re-MENTIA @ 397333e).
    private const val N_PREPARED_SAMPLES = 1280
    private const val SAMPLE_RATE = 16000
    private const val MEL_SPECTROGRAM_MAX_LEN = 10 * 97
    private const val FEATURE_BUFFER_MAX_LEN = 120
    private const val WINDOW_SIZE = 76
    private const val STEP_SIZE = 8
    private const val MEL_SPEC_FRAMES = 32
    private const val CLASSIFIER_FEATURE_FRAMES = 16

    /** Bounded inference queue: capture thread never blocks; backlog drops oldest. */
    private const val QUEUE_CAPACITY = 8
  }

  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

  private var melspecSession: OrtSession? = null
  private var embeddingSession: OrtSession? = null
  private var classifierSession: OrtSession? = null
  private var threshold: Float = 0.5f

  // Streaming state (AudioProcessor.kt:34-38).
  private var featureBuffer: Array<FloatArray>? = null
  private val rawDataBuffer = ArrayDeque<Float>(SAMPLE_RATE * 10)
  private var rawDataRemainder = floatArrayOf()
  private var melSpectrogramBuffer: Array<FloatArray> =
    Array(WINDOW_SIZE) { FloatArray(MEL_SPEC_FRAMES) { 1.0f } }
  private var accumulatedSamples = 0

  private val queue = LinkedBlockingQueue<FloatArray>(QUEUE_CAPACITY)
  @Volatile
  private var running = false
  private var workerThread: Thread? = null
  private var droppedChunks = 0

  val isInitialized: Boolean
    get() = classifierSession != null

  fun initialize(
    melspectrogramPath: String,
    embeddingPath: String,
    classifierPath: String,
    thresholdOverride: Double?,
    promise: Promise
  ) {
    try {
      unloadInternal()
      melspecSession = createSessionFromFile(melspectrogramPath)
      embeddingSession = createSessionFromFile(embeddingPath)
      classifierSession = createSessionFromFile(classifierPath)
      threshold = thresholdOverride?.toFloat() ?: 0.5f
      resetStreamingState()
      // Prime the feature buffer with random-noise embeddings
      // (AudioProcessor.kt:40-44; mirrors dscripka's AudioFeatures init).
      val randomData = FloatArray(SAMPLE_RATE * 4) { Random.nextFloat() * 2000f - 1000f }
      featureBuffer = batchEmbeddings(randomData, WINDOW_SIZE, STEP_SIZE)
      Log.i(logTag, "OWW_INIT ok threshold=$threshold primedFrames=${featureBuffer?.size}")
      promise.resolve(Arguments.createMap().apply { putBoolean("success", true) })
    } catch (e: Exception) {
      unloadInternal()
      Log.e(logTag, "initializeWakeWord failed: ${e.message}", e)
      promise.reject("OWW_INIT_ERROR", "Wake word init failed: ${e.message}", e)
    }
  }

  private fun createSessionFromFile(path: String): OrtSession {
    val resolved = path.removePrefix("file://")
    val file = File(resolved)
    if (!file.isFile) {
      throw IllegalArgumentException("Wake word model file does not exist: $resolved")
    }
    return env.createSession(file.readBytes())
  }

  fun start(promise: Promise) {
    try {
      if (!isInitialized) {
        promise.reject("OWW_STATE_ERROR", "Wake word detector not initialized")
        return
      }
      if (running) {
        promise.resolve(null)
        return
      }
      running = true
      droppedChunks = 0
      workerThread = thread(name = "OpenWakeWordHelper") {
        while (running) {
          val chunk = try {
            queue.take()
          } catch (_: InterruptedException) {
            break
          }
          if (!running) break
          try {
            val score = processChunk(chunk)
            val detected = score >= threshold
            if (detected) {
              Log.i(logTag, "OWW_DETECTION score=${"%.5f".format(score)}")
            }
            onScore(score, System.currentTimeMillis(), detected)
          } catch (e: Exception) {
            Log.e(logTag, "OWW worker inference error: ${e.message}", e)
          }
        }
      }
      promise.resolve(null)
    } catch (e: Exception) {
      running = false
      promise.reject("OWW_STATE_ERROR", "Wake word start failed: ${e.message}", e)
    }
  }

  fun stop(promise: Promise?) {
    running = false
    workerThread?.interrupt()
    workerThread?.join(2000)
    workerThread = null
    queue.clear()
    promise?.resolve(null)
  }

  fun unload(promise: Promise) {
    try {
      stop(null)
      unloadInternal()
      promise.resolve(null)
    } catch (e: Exception) {
      promise.reject("OWW_RELEASE_ERROR", "Wake word unload failed: ${e.message}", e)
    }
  }

  /** Call from Module.onCatalystInstanceDestroy (same pattern as SherpaOnnxKwsHelper.shutdown). */
  fun shutdown() {
    try {
      stop(null)
      unloadInternal()
    } catch (e: Exception) {
      Log.w(logTag, "OWW shutdown: ${e.message}")
    }
  }

  private fun unloadInternal() {
    melspecSession?.close(); melspecSession = null
    embeddingSession?.close(); embeddingSession = null
    classifierSession?.close(); classifierSession = null
    resetStreamingState()
    featureBuffer = null
  }

  private fun resetStreamingState() {
    rawDataBuffer.clear()
    rawDataRemainder = floatArrayOf()
    melSpectrogramBuffer = Array(WINDOW_SIZE) { FloatArray(MEL_SPEC_FRAMES) { 1.0f } }
    accumulatedSamples = 0
  }

  /**
   * Tee entry point — called on SherpaOnnxPcmCapture's capture thread with the
   * post-resample Int16 chunk. Must never block: bounded queue, drop-oldest.
   * Deviation 1 applies here: int16 values pass through as floats unscaled.
   */
  fun acceptPcmChunk(samples: ShortArray, sampleRate: Int) {
    if (!running) return
    if (sampleRate != SAMPLE_RATE) {
      // The capture layer resamples to the requested rate (16k for Parakeet);
      // anything else means the stream was started for a different consumer.
      Log.w(logTag, "OWW tee ignoring chunk at $sampleRate Hz (expected $SAMPLE_RATE)")
      return
    }
    val floats = FloatArray(samples.size) { i -> samples[i].toFloat() }
    if (!queue.offer(floats)) {
      queue.poll()
      queue.offer(floats)
      droppedChunks++
      if (droppedChunks % 25 == 1) {
        Log.w(logTag, "OWW inference backlog: dropped $droppedChunks chunks so far")
      }
    }
  }

  // ─── Streaming state machine (worker thread only) ─────────────────────────

  /** AudioProcessor.predictWakeWord (AudioProcessor.kt:52-58). */
  private fun processChunk(audioBuffer: FloatArray): Float {
    streamingFeatures(audioBuffer)
    val features = getFeatures(CLASSIFIER_FEATURE_FRAMES, -1)
    return predictClassifier(features)
  }

  /**
   * AudioProcessor.streamingFeatures (AudioProcessor.kt:60-126), ported
   * verbatim including the per-call accumulatedSamples reset. NOTE: with this
   * SDK's capture chunking (~0.1 s ≈ 1600 samples @16k) every call takes the
   * ≥N_PREPARED_SAMPLES path with remainder carry, so the source's sub-1280
   * else-branch (whose samples only survive as mel left-context) never
   * activates in practice here.
   */
  private fun streamingFeatures(audioBuffer: FloatArray): Int {
    var processedSamples = 0
    accumulatedSamples = 0
    var buffer = audioBuffer

    if (rawDataRemainder.isNotEmpty()) {
      buffer = rawDataRemainder + audioBuffer
      rawDataRemainder = floatArrayOf()
    }

    if (accumulatedSamples + buffer.size >= N_PREPARED_SAMPLES) {
      val remainder = (accumulatedSamples + buffer.size) % N_PREPARED_SAMPLES
      if (remainder != 0) {
        val evenChunks = buffer.copyOfRange(0, buffer.size - remainder)
        bufferRawData(evenChunks)
        accumulatedSamples += evenChunks.size
        rawDataRemainder = buffer.copyOfRange(buffer.size - remainder, buffer.size)
      } else {
        bufferRawData(buffer)
        accumulatedSamples += buffer.size
        rawDataRemainder = floatArrayOf()
      }
    } else {
      accumulatedSamples += buffer.size
      bufferRawData(buffer)
    }

    if (accumulatedSamples >= N_PREPARED_SAMPLES && accumulatedSamples % N_PREPARED_SAMPLES == 0) {
      streamingMelSpectrogram(accumulatedSamples)

      val x = Array(1) { Array(WINDOW_SIZE) { Array(MEL_SPEC_FRAMES) { FloatArray(1) } } }

      for (i in (accumulatedSamples / N_PREPARED_SAMPLES) - 1 downTo 0) {
        val ndx = if (i == 0) melSpectrogramBuffer.size else melSpectrogramBuffer.size - STEP_SIZE * i
        val start = maxOf(0, ndx - WINDOW_SIZE)

        for ((k, j) in (start until ndx).withIndex()) {
          for (w in 0 until MEL_SPEC_FRAMES) {
            x[0][k][w][0] = melSpectrogramBuffer[j][w]
          }
        }

        if (x[0].size == WINDOW_SIZE) {
          val newFeatures = generateEmbeddings(x)
          featureBuffer = featureBuffer?.let { existing -> existing + newFeatures } ?: newFeatures
        }
      }

      processedSamples = accumulatedSamples
      accumulatedSamples = 0
    }

    featureBuffer?.let { buffer2 ->
      if (buffer2.size > FEATURE_BUFFER_MAX_LEN) {
        featureBuffer = buffer2.takeLast(FEATURE_BUFFER_MAX_LEN).toTypedArray()
      }
    }

    return if (processedSamples != 0) processedSamples else accumulatedSamples
  }

  /** AudioProcessor.bufferRawData (AudioProcessor.kt:128-136). */
  private fun bufferRawData(data: FloatArray) {
    while (rawDataBuffer.size + data.size > SAMPLE_RATE * 10) {
      rawDataBuffer.poll()
    }
    data.forEach { rawDataBuffer.offer(it) }
  }

  /** AudioProcessor.streamingMelSpectrogram (AudioProcessor.kt:138-157). */
  private fun streamingMelSpectrogram(nSamples: Int) {
    require(rawDataBuffer.size >= 400) {
      "The number of input frames must be at least 400 samples @ 16kHz (25 ms)!"
    }
    val bufferList = rawDataBuffer.toList()
    val tempArray = bufferList.takeLast(nSamples + 480).toFloatArray()

    val newMelSpectrogram = computeMelSpectrogram(tempArray)
    melSpectrogramBuffer = melSpectrogramBuffer + newMelSpectrogram

    if (melSpectrogramBuffer.size > MEL_SPECTROGRAM_MAX_LEN) {
      melSpectrogramBuffer = melSpectrogramBuffer.takeLast(MEL_SPECTROGRAM_MAX_LEN).toTypedArray()
    }
  }

  /** AudioProcessor.getFeatures (AudioProcessor.kt:159-176). */
  private fun getFeatures(nFeatureFrames: Int, startNdx: Int): Array<Array<FloatArray>> {
    val buffer = featureBuffer ?: return arrayOf(arrayOf(floatArrayOf()))
    val startIndex = if (startNdx != -1) startNdx else maxOf(0, buffer.size - nFeatureFrames)
    val endIndex = if (startNdx != -1) minOf(startNdx + nFeatureFrames, buffer.size) else buffer.size
    val features = buffer.sliceArray(startIndex until endIndex)
    return arrayOf(features)
  }

  /** AudioProcessor.getEmbeddings (AudioProcessor.kt:178-199) — init-time priming only. */
  private fun batchEmbeddings(audioData: FloatArray, windowSize: Int, stepSize: Int): Array<FloatArray> {
    val spec = computeMelSpectrogram(audioData)
    val windows = mutableListOf<Array<FloatArray>>()
    for (i in 0..spec.size - windowSize step stepSize) {
      val window = spec.sliceArray(i until i + windowSize)
      if (window.size == windowSize) {
        windows.add(window)
      }
    }
    val batch = Array(windows.size) { i ->
      Array(windowSize) { j ->
        Array(spec[0].size) { k ->
          FloatArray(1) { windows[i][j][k] }
        }
      }
    }
    return generateEmbeddings(batch)
  }

  // ─── ONNX stages (sessions hoisted — deviation 2) ──────────────────────────

  /** MelSpectrogram.computeMelSpectrogram + squeeze + x/10+2 transform (MelSpectrogram.kt:29-76). */
  private fun computeMelSpectrogram(audioSamples: FloatArray): Array<FloatArray> {
    val session = melspecSession ?: throw IllegalStateException("melspec session not loaded")
    java.nio.FloatBuffer.wrap(audioSamples).let { floatBuffer ->
      OnnxTensor.createTensor(
        env,
        floatBuffer,
        longArrayOf(1L, audioSamples.size.toLong())
      ).use { inputTensor ->
        session.run(mapOf(session.inputNames.first() to inputTensor)).use { results ->
          @Suppress("UNCHECKED_CAST")
          val outputTensor = results[0].value as Array<Array<Array<FloatArray>>>
          val squeezed = Array(outputTensor[0][0].size) { i ->
            FloatArray(outputTensor[0][0][0].size) { j -> outputTensor[0][0][i][j] }
          }
          return Array(squeezed.size) { i ->
            FloatArray(squeezed[i].size) { j -> squeezed[i][j] / 10.0f + 2.0f }
          }
        }
      }
    }
  }

  /** EmbeddingModel.generateEmbeddings (EmbeddingModel.kt:27-52). */
  private fun generateEmbeddings(input: Array<Array<Array<FloatArray>>>): Array<FloatArray> {
    val session = embeddingSession ?: throw IllegalStateException("embedding session not loaded")
    OnnxTensor.createTensor(env, input).use { inputTensor ->
      session.run(mapOf(session.inputNames.first() to inputTensor)).use { results ->
        @Suppress("UNCHECKED_CAST")
        val rawOutput = results[0].value as Array<Array<Array<FloatArray>>>
        // Reshape (n, 1, 1, 96) → (n, 96).
        return Array(rawOutput.size) { i -> rawOutput[i][0][0].copyOf() }
      }
    }
  }

  /** OnnxModelRunner.predictWakeWord (OnnxModelRunner.kt:45-62). */
  private fun predictClassifier(inputArray: Array<Array<FloatArray>>): Float {
    val session = classifierSession ?: throw IllegalStateException("classifier session not loaded")
    OnnxTensor.createTensor(env, inputArray).use { inputTensor ->
      session.run(mapOf(session.inputNames.first() to inputTensor)).use { outputs ->
        @Suppress("UNCHECKED_CAST")
        val result = outputs[0].value as Array<FloatArray>
        return result[0][0]
      }
    }
  }
}
