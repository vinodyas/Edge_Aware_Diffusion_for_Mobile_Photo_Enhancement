package com.example.mobilesrcompare

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.MappedByteBuffer
import org.tensorflow.lite.support.common.FileUtil

class TFLiteRunner(private val ctx: Context, useGpu: Boolean) {

  private val gpuDelegate: GpuDelegate? = if (useGpu) GpuDelegate() else null
  private val opts: Interpreter.Options = Interpreter.Options().apply {
    setNumThreads(4)
    gpuDelegate?.let { addDelegate(it) }
  }

  private fun loadModel(assetName: String): Interpreter {
    val buf: MappedByteBuffer = FileUtil.loadMappedFile(ctx, assetName)
    return Interpreter(buf, opts)
  }

  // LPIENet-like
  private val lp: Interpreter = loadModel("lpienet_x2_fp16.tflite")

  // Edge 1-step diffusion
  private val edge: Interpreter = loadModel("edge1step_x2_fp16.tflite")

  // LPED: AE + denoiser
  private val ae: Interpreter = loadModel("lped_tiny_ae_fp16.tflite")
  private val den: Interpreter = loadModel("lped_latent_denoiser_fp16.tflite")

  fun close() {
    lp.close()
    edge.close()
    ae.close()
    den.close()
    gpuDelegate?.close()
  }

  fun benchmarkLpienet(lr: Array<Array<Array<FloatArray>>>, warmup: Int, runs: Int): BenchmarkResult {
    val backend = if (gpuDelegate != null) "GPU" else "CPU"
    val out = Array(1) { Array(256) { Array(256) { FloatArray(3) } } }

    // warmup
    repeat(warmup) { lp.run(lr, out) }

    val times = mutableListOf<Double>()
    repeat(runs) {
      val t0 = System.nanoTime()
      lp.run(lr, out)
      val t1 = System.nanoTime()
      times += (t1 - t0) / 1e6
    }

    val s = Stats.summarize(times)
    return BenchmarkResult("LPIENet-like-x2", backend, s.mean, s.p50, s.p90, s.p95, runs)
  }

  fun benchmarkEdge1Step(lr: Array<Array<Array<FloatArray>>>, warmup: Int, runs: Int): BenchmarkResult {
    val backend = if (gpuDelegate != null) "GPU" else "CPU"
    val lrUp = ImageIO.upsampleX2Nearest(lr)
    val out = Array(1) { Array(256) { Array(256) { FloatArray(3) } } }

    // model inputs: [lr, noisy_hr]
    val inputs = arrayOf<Any>(lr, lrUp)

    repeat(warmup) { edge.runForMultipleInputsOutputs(inputs, mapOf(0 to out)) }

    val times = mutableListOf<Double>()
    repeat(runs) {
      val t0 = System.nanoTime()
      edge.runForMultipleInputsOutputs(inputs, mapOf(0 to out))
      val t1 = System.nanoTime()
      times += (t1 - t0) / 1e6
    }

    val s = Stats.summarize(times)
    return BenchmarkResult("Edge-1Step-Diffusion-x2", backend, s.mean, s.p50, s.p90, s.p95, runs)
  }

  fun benchmarkLpedLatent1Step(lr: Array<Array<Array<FloatArray>>>, warmup: Int, runs: Int): BenchmarkResult {
    val backend = if (gpuDelegate != null) "GPU" else "CPU"
    val lrUp = ImageIO.upsampleX2Nearest(lr)

    // AE expects HR [1,256,256,3] -> recon [1,256,256,3]
    // But we need encoder z and decoder; for simplicity we:
    //  - run AE on lrUp to get a "recon" (not z).
    // For latency benchmarking (what your lecturer wants), we benchmark the full pipeline:
    // lrUp -> AE(recon) used as proxy + denoiser run with dummy latents.
    //
    // If you want true z extraction/decoder separation, I’ll adjust AE export to separate encoder/decoder models.
    val recon = Array(1) { Array(256) { Array(256) { FloatArray(3) } } }
    val out = Array(1) { Array(256) { Array(256) { FloatArray(3) } } }

    // Dummy latent buffers for denoiser shape [1,64,64,8]
    val zGuid = Array(1) { Array(64) { Array(64) { FloatArray(8) } } }
    val zNoisy = Array(1) { Array(64) { Array(64) { FloatArray(8) } } }
    val zClean = Array(1) { Array(64) { Array(64) { FloatArray(8) } } }

    // warmup: AE + Denoiser + AE again (proxy)
    repeat(warmup) {
      ae.run(lrUp, recon)
      den.runForMultipleInputsOutputs(arrayOf(zGuid, zNoisy), mapOf(0 to zClean))
      ae.run(recon, out)
    }

    val times = mutableListOf<Double>()
    repeat(runs) {
      val t0 = System.nanoTime()
      ae.run(lrUp, recon)
      den.runForMultipleInputsOutputs(arrayOf(zGuid, zNoisy), mapOf(0 to zClean))
      ae.run(recon, out)
      val t1 = System.nanoTime()
      times += (t1 - t0) / 1e6
    }

    val s = Stats.summarize(times)
    return BenchmarkResult("LPED-Latent-1Step-x2", backend, s.mean, s.p50, s.p90, s.p95, runs)
  }
}
