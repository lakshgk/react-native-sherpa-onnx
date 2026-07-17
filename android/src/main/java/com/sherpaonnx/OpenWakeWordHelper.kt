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
 * ─────────────────────────────────────────────────────────────────
 * SPEC PROVENANCE
 * UL Registry version:  smriti_prefix@2.0.0-draft (pending sign-off),
 *                       wake_word_model@draft (UL number TBD at sign-off)
 * Bounded Context:      VoiceCommand (v2.0 pending merge — GP-2026-015 Prompt 4)
 * Invariant IDs:        INV-OWW-001, INV-OWW-002, INV-OWW-003 (DL-058),
 *                       INV-OWW-007
 * Generation Plan:      GP-2026-015 (Prompt 1)
 * Confidence Zone Map:  syntactic(0%) / pattern(15%) / domain-semantic(85%)
 * Generated:            2026-07-17
 * Intent Engineer:      Smriti Architect-in-the-Loop
 * ─────────────────────────────────────────────────────────────────
 * CONFIDENCE ZONES
 *   PATTERN     → lifecycle promise plumbing (initialize/start/stop/unload/
 *                 shutdown result+reject conventions; follows
 *                 SherpaOnnxKwsHelper.kt)
 *   SEMANTIC    → companion constants (pipeline geometry);
 *                 acceptPcmChunk + bounded queue + worker re-chunking
 *                 (INV-OWW-002, INV-OWW-003 cadence);
 *                 streaming state machine (streamingFeatures et al.);
 *                 ONNX stage runners (mel transform, reshapes, score);
 *                 threshold gate + detection emission (INV-OWW-003);
 *                 start() full reset + re-prime (INV-OWW-007)
 *                 ⚠ REQUIRES DOMAIN EXPERT REVIEW
 *                 ⚠ LAYER 4 TEST ID: none yet — see LAYER 4 COVERAGE
 * ─────────────────────────────────────────────────────────────────
 * LAYER 4 COVERAGE (domain-semantic regions only)
 *   NONE AT THIS STAGE (GP-2026-015 Prompt 1 has no L4 tests by design).
 *   Planned Prompt 4: oww_inv_001_l4 (load failure non-fatal),
 *   oww_inv_002_l4 (tee never blocks Parakeet), oww_inv_003_l4
 *   (threshold-only gate, DL-058), oww_inv_007_l4 (stateless across
 *   sessions) — all against FakeOpenWakeWordDetector in the smriti repo.
 *   Interim covering evidence for this native code is the Rule 15 N8
 *   device record (DL-058; tag n8-spike-2026-07-17) plus the Prompt 1
 *   confirmation run recorded in smriti FEATURE_NOTES.
 * ─────────────────────────────────────────────────────────────────
 * OPEN DOMAIN QUESTIONS
 *   (none) — DQ-GP015-P1-01 (dedup placement) RESOLVED: DL-065, JS layer.
 * ─────────────────────────────────────────────────────────────────
 *
 * Acoustic wake-word detector for the "Smriti" trigger (smriti_prefix as an
 * ACOUSTIC EVENT, UL-015 v2): runs the openWakeWord three-stage pipeline
 * (melspectrogram → embedding → classifier) on the ONNX Runtime Java API
 * already bundled by this SDK's prebuilt pipeline (build.gradle:
 * extractOnnxruntimeClasses), fed by a native-side tee of SherpaOnnxPcmCapture
 * — no second mic client (INV-OWW-002 by construction). A detection is
 * emitted for EVERY 1280-sample (80 ms) inference frame whose classifier
 * score is at or above the signed wake_word_model threshold — threshold-only
 * gating per DL-058 / INV-OWW-003: NO minimum-consecutive-frames requirement,
 * NO debounce, NO hysteresis, NO cross-frame state of any kind at this layer.
 * Confusable/coincidence absorption belongs to INV-VC-009's wake+verb
 * conjunction (JS layers, GP-2026-015 Prompts 2/3); dedup of consecutive
 * ≥threshold frames belongs to wakeWordDetector.ts (DL-065). Do not
 * reintroduce either here.
 *
 * PORTING SOURCE (Rule 16 — observed behavior, cited; all line refs verified
 * against a fresh clone at the pinned commit, 2026-07-17):
 * Re-MENTIA/openwakeword-android-kt @ 397333e, Apache-2.0
 * (wakeword/src/main/kotlin/com/rementia/openwakeword/lib/):
 *   - constants: audio/AudioProcessor.kt:20-29 — except
 *     CLASSIFIER_FEATURE_FRAMES, hoisted from the literal 16 at
 *     AudioProcessor.kt:54 (source has no named constant for it)
 *   - streaming state fields: audio/AudioProcessor.kt:34-38
 *   - feature-buffer priming: audio/AudioProcessor.kt:40-44
 *   - predictWakeWord: audio/AudioProcessor.kt:52-58 → processChunk
 *   - streamingFeatures: audio/AudioProcessor.kt:60-126 (ported verbatim,
 *     incl. the per-call accumulatedSamples reset at :62)
 *   - bufferRawData: audio/AudioProcessor.kt:128-136
 *   - streamingMelSpectrogram: audio/AudioProcessor.kt:138-157
 *   - getFeatures: audio/AudioProcessor.kt:159-176
 *   - getEmbeddings: audio/AudioProcessor.kt:178-199 → batchEmbeddings
 *   - melspec stage + squeeze + x/10+2 transform: ml/MelSpectrogram.kt:29-76
 *   - embedding stage ("input_1", [n,76,32,1] → [n,96]): ml/EmbeddingModel.kt:27-52
 *   - classifier inference ([1,16,96] → [1,1] score): ml/OnnxModelRunner.kt:45-62
 *
 * DELIBERATE DEVIATIONS from the porting source (each observation- or
 * invariant-backed):
 *   D1. INPUT SCALING — samples are fed as int16-range floats
 *       (short.toFloat(), NO /32768 normalization). Re-MENTIA's
 *       AudioRecorder.kt:77-78 divides by 32768, but the reference stack that
 *       validated candidate 3 rev v7 (dscripka openWakeWord Python, smriti
 *       spikes/s0_wakeword s0_validate.py, S0 PASS 2026-07-16) feeds raw
 *       int16-valued floats. The S0 corpus is the observation; Re-MENTIA's
 *       own scaling is the outlier.
 *   D2. SESSION LIFETIME — OrtSessions are created once at initialize() and
 *       reused. Re-MENTIA recreates melspec/embedding sessions on every call
 *       (MelSpectrogram.kt:36-37, EmbeddingModel.kt:32-34) — behaviorally
 *       identical numerics, prohibitive per-80ms-frame cost on-device.
 *   D3. MODEL LOADING — from absolute file paths (this SDK's convention, cf.
 *       SherpaOnnxSttHelper modelPath handling) instead of AssetManager.
 *   D4. RESET + RE-PRIME ON start() — INV-OWW-007 (stateless across
 *       sessions): every start() fully resets the streaming state (raw/mel/
 *       remainder/carry buffers) and re-primes the feature buffer via the
 *       same AudioProcessor.kt:40-44 priming path used at init. The source
 *       has no session concept (its AudioProcessor lives for the app run),
 *       so this deviation cites the invariant, not a source observation.
 *   CADENCE. One inference per exact 1280-sample block (true 12.5 Hz),
 *       re-chunking tee'd capture chunks in the worker — ported from fork
 *       commit 2698349 (the DL-058-designated reference implementation;
 *       verified 12.47 Hz observed on-device). Calibration numbers are only
 *       valid at this cadence.
 */
internal class OpenWakeWordHelper(
  private val logTag: String,
  /**
   * Invoked on the worker thread for every inference frame whose score is at
   * or above the signed threshold (INV-OWW-003: threshold-only). Raw and
   * undeduplicated by design — DL-065 places dedup at the JS boundary.
   * timestampMs is the NATIVE epoch-millis at inference completion; JS-side
   * dedup must compare against this, not JS arrival time (DL-065).
   */
  private val onDetection: (confidence: Float, timestampMs: Long) -> Unit
) {

  companion object {
    // Constants ported from AudioProcessor.kt:20-29 (Re-MENTIA @ 397333e),
    // except CLASSIFIER_FEATURE_FRAMES — hoisted from the literal at
    // AudioProcessor.kt:54.
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

    /** OWW_CADENCE heartbeat interval (256 blocks ≈ 20.5 s at 12.5 Hz). */
    private const val CADENCE_LOG_INTERVAL_BLOCKS = 256L
  }

  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

  private var melspecSession: OrtSession? = null
  private var embeddingSession: OrtSession? = null
  private var classifierSession: OrtSession? = null

  /**
   * The signed wake_word_model detection threshold (DL-058: threshold-only —
   * there is no frame-count attribute to read). Required at initialize();
   * this module never supplies a default, so the signed value cannot be
   * silently bypassed.
   */
  private var threshold: Float = Float.NaN

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

  /**
   * Worker-thread-only carry buffer for re-chunking tee'd audio into exact
   * N_PREPARED_SAMPLES blocks — one prediction per block, true 12.5 Hz
   * (fork 2698349, the DL-058 reference cadence). N8 finding (2026-07-17):
   * predicting once per ~120 ms capture chunk (~8.3 Hz) destroys the
   * reference frame behavior; the reference pipeline (dscripka
   * Model.predict) consumes exact 1280-sample chunks.
   */
  private var pendingSamples = floatArrayOf()

  val isInitialized: Boolean
    get() = classifierSession != null

  /**
   * INV-OWW-001 contract: any failure here rejects the promise AFTER a full
   * unloadInternal(), and isInitialized derives from live session handles —
   * a failed initialize can never leave a false "loaded" state. Catching the
   * rejection (warn-log, proceed without wake-word detection for the
   * session) is the JS boundary's job (GP-2026-015 Prompt 2).
   */
  fun initialize(
    melspectrogramPath: String,
    embeddingPath: String,
    classifierPath: String,
    threshold: Double,
    promise: Promise
  ) {
    try {
      unloadInternal()
      melspecSession = createSessionFromFile(melspectrogramPath)
      embeddingSession = createSessionFromFile(embeddingPath)
      classifierSession = createSessionFromFile(classifierPath)
      this.threshold = threshold.toFloat()
      resetStreamingState()
      primeFeatureBuffer()
      Log.i(logTag, "OWW_INIT ok threshold=${this.threshold} primedFrames=${featureBuffer?.size}")
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

  /**
   * Prime the feature buffer with random-noise embeddings
   * (AudioProcessor.kt:40-44; mirrors dscripka's AudioFeatures init).
   * Called at initialize() and again on every start() (deviation D4 /
   * INV-OWW-007) so no embedding state survives from a prior session.
   */
  private fun primeFeatureBuffer() {
    val randomData = FloatArray(SAMPLE_RATE * 4) { Random.nextFloat() * 2000f - 1000f }
    featureBuffer = batchEmbeddings(randomData, WINDOW_SIZE, STEP_SIZE)
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
      // Deviation D4 (INV-OWW-007): full per-session reset — streaming
      // buffers, carry buffer, queue, and a fresh feature-buffer prime.
      resetStreamingState()
      pendingSamples = floatArrayOf()
      queue.clear()
      primeFeatureBuffer()
      running = true
      droppedChunks = 0
      workerThread = thread(name = "OpenWakeWordHelper") {
        var blockCount = 0L
        val workerStartMs = System.currentTimeMillis()
        while (running) {
          val chunk = try {
            queue.take()
          } catch (_: InterruptedException) {
            break
          }
          if (!running) break
          try {
            // Re-chunk to exact 1280-sample blocks; one prediction per block
            // (true 12.5 Hz — see pendingSamples doc above).
            val buf = if (pendingSamples.isEmpty()) chunk else pendingSamples + chunk
            var off = 0
            while (off + N_PREPARED_SAMPLES <= buf.size) {
              val block = buf.copyOfRange(off, off + N_PREPARED_SAMPLES)
              off += N_PREPARED_SAMPLES
              val score = processChunk(block)
              blockCount++
              if (blockCount % CADENCE_LOG_INTERVAL_BLOCKS == 0L) {
                Log.i(
                  logTag,
                  "OWW_CADENCE blocks=$blockCount elapsedMs=${System.currentTimeMillis() - workerStartMs}"
                )
              }
              // Dense frame-shape instrumentation for calibration evidence:
              // every non-floor score, adb-visible (logcat-only channel; the
              // JS bridge carries detections only).
              if (score >= 0.1f) {
                Log.i(logTag, "OWW_FRAME score=${"%.5f".format(score)}")
              }
              // INV-OWW-003 (DL-058): the ENTIRE wake gate. A single frame at
              // or above the signed threshold fires; nothing else is consulted.
              if (score >= threshold) {
                Log.i(logTag, "OWW_DETECTION score=${"%.5f".format(score)}")
                onDetection(score, System.currentTimeMillis())
              }
            }
            pendingSamples = buf.copyOfRange(off, buf.size)
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
    pendingSamples = floatArrayOf()
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
    threshold = Float.NaN
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
   * post-resample Int16 chunk. INV-OWW-002: must never block — bounded queue,
   * drop-oldest; the only work on the capture thread is this copy + offer.
   * Deviation D1 applies here: int16 values pass through as floats unscaled.
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
   * verbatim including the per-call accumulatedSamples reset (:62). NOTE:
   * with the worker's exact-1280 re-chunking every call takes the
   * ≥N_PREPARED_SAMPLES path with zero remainder, so the source's sub-1280
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

  /** AudioProcessor.getEmbeddings (AudioProcessor.kt:178-199) — priming only (init + D4 start). */
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

  // ─── ONNX stages (sessions hoisted — deviation D2) ─────────────────────────

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
