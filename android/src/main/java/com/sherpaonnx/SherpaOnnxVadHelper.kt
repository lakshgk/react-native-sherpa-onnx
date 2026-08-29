package com.sherpaonnx

import android.util.Base64
import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread

/**
 * ─────────────────────────────────────────────────────────────────
 * ENABLING WORK — NOT a generated domain module, NO invariants.
 * Smriti DL-104 Direction A (signed 2026-08-29): exposes the Silero VAD
 * already present in the sherpa-onnx AAR so that the Rule 15 / §N8 device
 * spike can characterise its behaviour BEFORE GP-2026-026's invariants are
 * written. Follows OpenWakeWordHelper.kt (GP-2026-015 Prompt 1) for
 * lifecycle, tee, queue and emission conventions.
 *
 * Substrate verified 2026-08-29 against AAR com.xdcobra.sherpa:sherpa-onnx:
 * 1.12.34-2 — classes.jar carries Vad / VadModelConfig / SileroVadModelConfig
 * / SpeechSegment, and libsherpa-onnx-jni.so (arm64-v8a) exports all twelve
 * VAD JNI entry points. SpeechSegment ctor descriptor is (I[F)V with fields
 * start / samples, matching upstream kotlin-api/Vad.kt @ v1.12.34.
 * ─────────────────────────────────────────────────────────────────
 * HAZARDS observed in upstream source @ v1.12.34 — each is defended below.
 *
 *  1. NOTHING IN THE VAD IS THREAD-SAFE. voice-activity-detector.cc,
 *     circular-buffer.cc and silero-vad-model.cc carry no synchronisation,
 *     and Reset()/Flush() are declared const while mutating impl state — so
 *     `const` carries no thread-safety signal here. EVERY vad.* call in this
 *     file is confined to the single worker thread. (The official
 *     SherpaOnnxVadAsr example calls vad.reset() from the UI thread while its
 *     recording thread polls; that example is racy — deliberately not copied.)
 *
 *  2. BAD CONFIG CALLS exit(-1) — PROCESS DEATH, no exception, no stack.
 *     Reachable via wrong sampleRate, both model fields empty, or a v5 model
 *     with windowSize != 512. Every such value is validated in initialize()
 *     BEFORE the Vad is constructed, because once it is called there is
 *     nothing left to catch.
 *
 *  3. METHODS CALLED AFTER release() SEGFAULT. Only acceptWaveform / reset /
 *     compute carry ValidatePointer in JNI; empty / pop / front / clear /
 *     flush / isSpeechDetected do not. The worker is joined before release,
 *     and `vad` is nulled under the lifecycle flag so no path can reach a
 *     freed pointer.
 *
 *  4. maxSpeechDuration IS NOT A CUT. It is a soft pressure valve: exceeding
 *     it merely drops minSilenceDuration to 0.1s and raises threshold to 0.90
 *     to hunt for a break. A speaker who never pauses never ends a segment,
 *     and the 60s ring buffer DOUBLES on overflow rather than capping
 *     ("Overflow! ... No data loss!" in logcat is the canary), so a long
 *     monologue grows an unbounded FloatArray that then crosses JNI in one
 *     front() call. The hard cap is therefore APP-SIDE: see maxSegmentMs.
 *
 *  5. compute() MUST NOT be interleaved with acceptWaveform() — both advance
 *     the same LSTM state. compute() is not called anywhere in this file.
 *
 *  6. SCALE DIFFERS FROM THE WAKE-WORD TEE. openWakeWord consumes raw
 *     int16-scale floats; Silero expects [-1, 1]. This file divides by 32768.
 * ─────────────────────────────────────────────────────────────────
 */
class SherpaOnnxVadHelper(
  private val logTag: String,
  /**
   * Invoked from the worker thread for each completed speech segment.
   * base64Pcm is little-endian Int16 (same wire convention as
   * pcmLiveStreamData); startSample is the VAD's absolute sample offset since
   * the last reset(); viaFlush marks a segment produced by Flush() rather
   * than by a natural silence close — those two differ in tail character
   * (see emitSegment).
   */
  private val onSegment: (
    base64Pcm: String,
    startSample: Long,
    sampleCount: Int,
    viaFlush: Boolean,
    forcedByCap: Boolean,
    latencyMs: Long
  ) -> Unit
) {
  companion object {
    /** Silero @16kHz is trained on 512/1024/1536; the k2-fsa v4 export
     *  requires exactly 512 (v5 hard-exits otherwise). */
    private const val REQUIRED_WINDOW_SIZE = 512
    private const val REQUIRED_SAMPLE_RATE = 16_000
    /** Bounded tee queue — same backpressure stance as the wake-word helper:
     *  drop oldest rather than block the capture thread. */
    private const val QUEUE_CAPACITY = 64
  }

  private var vad: Vad? = null
  private val queue = LinkedBlockingQueue<FloatArray>(QUEUE_CAPACITY)
  private var workerThread: Thread? = null

  @Volatile private var running = false
  @Volatile private var initialized = false

  /** App-side hard cap (hazard 4). 0 disables it — the spike sets it high
   *  deliberately to observe the uncapped pathology. */
  private var maxSegmentMs = 0L

  private var droppedChunks = 0
  /** Total samples handed to the VAD since start(); the clock all latency
   *  and cap arithmetic is measured against. Worker thread only. */
  private var fedSamples = 0L
  /** fedSamples at which the currently-open segment was first observed, or
   *  -1 when no segment is open. Worker thread only. */
  private var openSegmentAtSample = -1L

  val isInitialized: Boolean get() = initialized

  /**
   * Construct the VAD. Every value that can reach an exit(-1) is validated
   * here first (hazard 2) — a rejected promise is recoverable, a killed
   * process is not.
   */
  fun initialize(
    modelPath: String,
    threshold: Double,
    minSilenceDuration: Double,
    minSpeechDuration: Double,
    maxSpeechDuration: Double,
    windowSize: Int,
    sampleRate: Int,
    maxSegmentMs: Double,
    debug: Boolean,
    promise: Promise
  ) {
    try {
      unloadInternal()

      val resolved = modelPath.removePrefix("file://")
      val file = File(resolved)
      if (!file.isFile) {
        throw IllegalArgumentException("VAD model file does not exist: $resolved")
      }
      if (sampleRate != REQUIRED_SAMPLE_RATE) {
        throw IllegalArgumentException(
          "sampleRate must be $REQUIRED_SAMPLE_RATE (native VAD calls exit(-1) otherwise), got $sampleRate"
        )
      }
      if (windowSize != REQUIRED_WINDOW_SIZE) {
        throw IllegalArgumentException(
          "windowSize must be $REQUIRED_WINDOW_SIZE for silero at 16kHz (v5 calls exit(-1) otherwise), got $windowSize"
        )
      }
      if (threshold <= 0.0 || threshold >= 1.0) {
        throw IllegalArgumentException("threshold must be in (0,1), got $threshold")
      }
      if (minSilenceDuration <= 0.0 || minSpeechDuration <= 0.0 || maxSpeechDuration <= 0.0) {
        throw IllegalArgumentException("durations must be > 0")
      }

      val config = VadModelConfig(
        sileroVadModelConfig = SileroVadModelConfig(
          model = resolved,
          threshold = threshold.toFloat(),
          minSilenceDuration = minSilenceDuration.toFloat(),
          minSpeechDuration = minSpeechDuration.toFloat(),
          windowSize = windowSize,
          maxSpeechDuration = maxSpeechDuration.toFloat()
        ),
        sampleRate = sampleRate,
        numThreads = 1,
        provider = "cpu",
        debug = debug
      )
      // newFromFile (not newFromAsset): the asset path skips config.Validate()
      // upstream, so the file path is both correct for our download/asset
      // layout and the only one that validates.
      vad = Vad(assetManager = null, config = config)
      this.maxSegmentMs = maxSegmentMs.toLong()
      initialized = true

      Log.i(
        logTag,
        "VAD_INIT ok threshold=$threshold minSil=$minSilenceDuration minSpeech=$minSpeechDuration " +
          "maxSpeech=$maxSpeechDuration window=$windowSize cap=${this.maxSegmentMs}ms model=$resolved"
      )
      promise.resolve(Arguments.createMap().apply { putBoolean("success", true) })
    } catch (e: Exception) {
      unloadInternal()
      Log.e(logTag, "initializeVad failed: ${e.message}", e)
      promise.reject("VAD_INIT_ERROR", "VAD init failed: ${e.message}", e)
    }
  }

  fun start(promise: Promise) {
    try {
      if (!initialized) {
        promise.reject("VAD_STATE_ERROR", "VAD not initialized")
        return
      }
      if (running) {
        promise.resolve(null)
        return
      }
      queue.clear()
      droppedChunks = 0
      fedSamples = 0L
      openSegmentAtSample = -1L
      running = true

      workerThread = thread(name = "SherpaOnnxVadHelper") {
        // reset() on the worker, never the caller's thread (hazard 1).
        try {
          vad?.reset()
        } catch (e: Exception) {
          Log.e(logTag, "VAD reset at start failed: ${e.message}", e)
        }
        while (running) {
          val chunk = try {
            queue.poll(100, java.util.concurrent.TimeUnit.MILLISECONDS)
          } catch (e: InterruptedException) {
            null
          } ?: continue
          processChunk(chunk)
        }
      }
      Log.i(logTag, "VAD_START")
      promise.resolve(null)
    } catch (e: Exception) {
      Log.e(logTag, "startVadDetection failed: ${e.message}", e)
      promise.reject("VAD_START_ERROR", e.message ?: "startVadDetection failed", e)
    }
  }

  /**
   * Stop the loop, then flush the open segment (if any) on the worker before
   * it exits. Flush() emits the in-progress segment WITHOUT the trailing-
   * silence trim a natural close applies — the segment is tagged viaFlush so
   * a consumer can tell the two apart.
   */
  fun stop(promise: Promise?) {
    try {
      if (!running) {
        promise?.resolve(null)
        return
      }
      running = false
      val t = workerThread
      workerThread = null
      // Join before touching the VAD from this thread: the worker owns it.
      t?.join(2_000)

      // The worker has exited; this thread is now the sole owner, so the
      // flush + drain below is still single-threaded (hazard 1 preserved).
      try {
        vad?.let { v ->
          v.flush()
          drainSegments(v, viaFlush = true, forcedByCap = false)
        }
      } catch (e: Exception) {
        Log.e(logTag, "VAD flush at stop failed: ${e.message}", e)
      }
      Log.i(logTag, "VAD_STOP fedSamples=$fedSamples dropped=$droppedChunks")
      promise?.resolve(null)
    } catch (e: Exception) {
      Log.e(logTag, "stopVadDetection failed: ${e.message}", e)
      promise?.reject("VAD_STOP_ERROR", e.message ?: "stopVadDetection failed", e)
    }
  }

  fun unload(promise: Promise) {
    try {
      unloadInternal()
      promise.resolve(null)
    } catch (e: Exception) {
      promise.reject("VAD_UNLOAD_ERROR", e.message ?: "unloadVad failed", e)
    }
  }

  fun shutdown() {
    unloadInternal()
  }

  private fun unloadInternal() {
    running = false
    val t = workerThread
    workerThread = null
    try {
      t?.join(2_000)
    } catch (e: InterruptedException) {
      Log.w(logTag, "VAD worker join interrupted")
    }
    queue.clear()
    // Release only after the worker is provably gone — post-release calls
    // segfault rather than throwing (hazard 3).
    try {
      vad?.release()
    } catch (e: Exception) {
      Log.w(logTag, "VAD release failed: ${e.message}")
    }
    vad = null
    initialized = false
    openSegmentAtSample = -1L
  }

  /**
   * PCM tee entry point. Must not block the capture thread — on backlog the
   * oldest chunk is dropped, matching the wake-word helper's stance.
   *
   * NOTE the scale conversion (hazard 6): Silero wants [-1, 1], so this
   * divides by 32768 where the wake-word tee keeps int16 scale.
   */
  fun acceptPcmChunk(samples: ShortArray, sampleRate: Int) {
    if (!running) return
    if (sampleRate != REQUIRED_SAMPLE_RATE) {
      Log.w(logTag, "VAD tee ignoring chunk at $sampleRate Hz (expected $REQUIRED_SAMPLE_RATE)")
      return
    }
    val floats = FloatArray(samples.size) { i -> samples[i] / 32768.0f }
    if (!queue.offer(floats)) {
      queue.poll()
      queue.offer(floats)
      droppedChunks++
      if (droppedChunks % 25 == 1) {
        Log.w(logTag, "VAD_BACKLOG dropped=$droppedChunks")
      }
    }
  }

  // ─── Worker thread only ───────────────────────────────────────────────

  private fun processChunk(chunk: FloatArray) {
    val v = vad ?: return
    try {
      // AcceptWaveform buffers internally and slices its own windows, so an
      // arbitrary chunk length is accepted; only the inner IsSpeech() needs
      // exactly windowSize, which AcceptWaveform always satisfies.
      v.acceptWaveform(chunk)
      fedSamples += chunk.size

      val speaking = v.isSpeechDetected()
      if (speaking && openSegmentAtSample < 0) {
        openSegmentAtSample = fedSamples
        Log.i(logTag, "VAD_STATE open at=$fedSamples")
      }

      drainSegments(v, viaFlush = false, forcedByCap = false)

      // App-side hard cap (hazard 4): the library will not end this segment
      // on its own if the speaker never pauses.
      if (maxSegmentMs > 0 && openSegmentAtSample >= 0 && v.isSpeechDetected()) {
        val openMs = (fedSamples - openSegmentAtSample) * 1000L / REQUIRED_SAMPLE_RATE
        if (openMs >= maxSegmentMs) {
          Log.w(logTag, "VAD_CAP_FIRED openMs=$openMs cap=$maxSegmentMs — forcing flush")
          v.flush()
          drainSegments(v, viaFlush = true, forcedByCap = true)
        }
      }
    } catch (e: Exception) {
      Log.e(logTag, "VAD processChunk failed: ${e.message}", e)
    }
  }

  /** front()/pop() drain. Caller must be the sole owner of [v]. */
  private fun drainSegments(v: Vad, viaFlush: Boolean, forcedByCap: Boolean) {
    while (!v.empty()) {
      val segment = v.front()
      val start = segment.start.toLong()
      val samples = segment.samples
      // How far behind the live edge this segment closed — the number the
      // spike reads to characterise speech-end latency and the trailing trim.
      val latencyMs = (fedSamples - (start + samples.size)) * 1000L / REQUIRED_SAMPLE_RATE

      Log.i(
        logTag,
        "VAD_SEG start=$start n=${samples.size} durMs=${samples.size * 1000L / REQUIRED_SAMPLE_RATE} " +
          "fed=$fedSamples latencyMs=$latencyMs viaFlush=$viaFlush cap=$forcedByCap"
      )

      onSegment(encodeInt16Base64(samples), start, samples.size, viaFlush, forcedByCap, latencyMs)
      v.pop()
    }
    if (!v.isSpeechDetected()) {
      openSegmentAtSample = -1L
    }
  }

  /** Float [-1,1] → little-endian Int16 → base64 (pcmLiveStreamData wire form). */
  private fun encodeInt16Base64(samples: FloatArray): String {
    val buf = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    for (s in samples) {
      val clamped = when {
        s > 1.0f -> 1.0f
        s < -1.0f -> -1.0f
        else -> s
      }
      buf.putShort((clamped * 32767.0f).toInt().toShort())
    }
    return Base64.encodeToString(buf.array(), Base64.NO_WRAP)
  }
}
