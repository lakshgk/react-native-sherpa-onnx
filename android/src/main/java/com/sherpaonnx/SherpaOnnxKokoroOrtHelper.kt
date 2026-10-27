package com.sherpaonnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Base64
import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableArray
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * ─────────────────────────────────────────────────────────────────
 * SPEC PROVENANCE
 * UL Registry version:  tts_adapter@2.0.0 (UL-033)
 * Bounded Context:      SpeechIO
 * Invariant IDs:        INV-RB-009 (SIP-2026-14, drafted: 4 intra-op
 *                       threads, CPU EP), INV-RB-015 (the session is the
 *                       resident engine; release is explicit)
 * Generation Plan:      GP-2026-072 (Route B, DL-129; Step A §4.1 option
 *                       (b), SA-14)
 * Confidence Zone Map:  syntactic(20%) / pattern(70%) / domain-semantic(10%)
 * Generated:            2026-10-27
 * Intent Engineer:      Smriti Architect-in-the-Loop
 * ─────────────────────────────────────────────────────────────────
 * CONFIDENCE ZONES
 *   PATTERN     → session lifecycle and the promise plumbing (follows
 *                 OpenWakeWordHelper.kt: ai.onnxruntime on the runtime the
 *                 APK already ships, OrtEnvironment.getEnvironment(),
 *                 session.run(mapOf(name to tensor)))
 *   SEMANTIC    → createSession's options (INV-RB-009: 4 intra-op threads,
 *                 the CPU execution provider sherpa used)
 *                 ⚠ REQUIRES DOMAIN EXPERT REVIEW
 *                 ⚠ LAYER 4 TEST ID: inv_rb_009_l4 (the JS side passes the
 *                   pinned values; this file is device-gated, G1/G2/G6)
 * ─────────────────────────────────────────────────────────────────
 * LAYER 4 COVERAGE (domain-semantic regions only)
 *   None in Node: Kotlin is device-gated (G1, G2, G6). The JS engine that
 *   drives it is L4-tested against its Rule 16 fake.
 * ─────────────────────────────────────────────────────────────────
 * OPEN DOMAIN QUESTIONS
 *   (none)
 * ─────────────────────────────────────────────────────────────────
 *
 * Kokoro v1.0 run directly on ONNX Runtime, replacing sherpa-onnx's TTS
 * path (and with it espeak-ng; DL-129). One session per resident engine;
 * one run per planned unit: the caller passes the unit's token ids, its
 * style row and the speed, and gets the unit's whole waveform back as
 * little-endian float32, base64-encoded, so it crosses the bridge as one
 * string rather than a 100,000-element array. Runs never overlap on a
 * session: they go through one single-thread executor, and the JS side
 * orders every create / run / release through its mutex (INV-RB-015).
 * The model's inputs are `input_ids` (int64 [1, n]), `style` (float32
 * [1, 256]) and `speed` (float32 [1]); its output is `waveform` (float32),
 * as read from the onnx-community fp32 export on 2026-10-27
 * (`spikes/gp2026-072-route-b/mac-half-record.md`).
 */
class SherpaOnnxKokoroOrtHelper(private val logTag: String) {

  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  private val sessions = ConcurrentHashMap<String, OrtSession>()
  private val nextId = AtomicInteger(1)
  /** Runs off the module thread, one at a time (a run takes up to seconds). */
  private val executor: ExecutorService = Executors.newSingleThreadExecutor { r ->
    Thread(r, "kokoro-ort").apply { isDaemon = true }
  }

  fun createKokoroSession(modelPath: String, numThreads: Double, promise: Promise) {
    executor.execute {
      try {
        val resolved = modelPath.removePrefix("file://")
        if (!File(resolved).isFile) {
          throw IllegalArgumentException("Kokoro model file does not exist: $resolved")
        }
        val options = OrtSession.SessionOptions().apply {
          setIntraOpNumThreads(numThreads.toInt())
          setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        // By path, not by bytes: a 325 MB model read into a byte array first
        // would hold it twice while the session is built.
        val session = env.createSession(resolved, options)
        val id = "kokoro_ort_${nextId.getAndIncrement()}"
        sessions[id] = session
        Log.i(logTag, "KOKORO_ORT create id=$id threads=${numThreads.toInt()}")
        promise.resolve(id)
      } catch (e: Exception) {
        Log.e(logTag, "createKokoroSession failed: ${e.message}", e)
        promise.reject("KOKORO_ORT_CREATE_ERROR", "Kokoro session create failed: ${e.message}", e)
      }
    }
  }

  fun runKokoro(handle: String, tokenIds: ReadableArray, style: ReadableArray, speed: Double, promise: Promise) {
    executor.execute {
      val session = sessions[handle]
      if (session == null) {
        promise.reject("KOKORO_ORT_NO_SESSION", "Kokoro session not found: $handle")
        return@execute
      }
      var ids: OnnxTensor? = null
      var sty: OnnxTensor? = null
      var spd: OnnxTensor? = null
      try {
        val n = tokenIds.size()
        val idBuf = LongBuffer.allocate(n)
        for (i in 0 until n) idBuf.put(tokenIds.getDouble(i).toLong())
        idBuf.flip()
        val styleBuf = FloatBuffer.allocate(style.size())
        for (i in 0 until style.size()) styleBuf.put(style.getDouble(i).toFloat())
        styleBuf.flip()
        ids = OnnxTensor.createTensor(env, idBuf, longArrayOf(1, n.toLong()))
        sty = OnnxTensor.createTensor(env, styleBuf, longArrayOf(1, style.size().toLong()))
        spd = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(speed.toFloat())), longArrayOf(1))
        val t0 = System.nanoTime()
        session.run(mapOf("input_ids" to ids, "style" to sty, "speed" to spd)).use { result ->
          val out = result.get(0) as OnnxTensor
          val fb = out.floatBuffer
          val samples = fb.remaining()
          val bytes = ByteBuffer.allocate(samples * 4).order(ByteOrder.LITTLE_ENDIAN)
          bytes.asFloatBuffer().put(fb)
          val ms = (System.nanoTime() - t0) / 1_000_000
          Log.i(logTag, "KOKORO_ORT run id=$handle tokens=$n samples=$samples ms=$ms")
          promise.resolve(Arguments.createMap().apply {
            putString("base64", Base64.encodeToString(bytes.array(), Base64.NO_WRAP))
            putInt("samples", samples)
            putDouble("runMs", ms.toDouble())
          })
        }
      } catch (e: Exception) {
        Log.e(logTag, "runKokoro failed: ${e.message}", e)
        promise.reject("KOKORO_ORT_RUN_ERROR", "Kokoro run failed: ${e.message}", e)
      } finally {
        ids?.close()
        sty?.close()
        spd?.close()
      }
    }
  }

  fun releaseKokoroSession(handle: String, promise: Promise) {
    executor.execute {
      try {
        sessions.remove(handle)?.close()
        Log.i(logTag, "KOKORO_ORT release id=$handle")
        promise.resolve(null)
      } catch (e: Exception) {
        promise.reject("KOKORO_ORT_RELEASE_ERROR", "Kokoro session release failed: ${e.message}", e)
      }
    }
  }

  fun shutdown() {
    sessions.values.forEach { runCatching { it.close() } }
    sessions.clear()
    executor.shutdownNow()
  }
}
