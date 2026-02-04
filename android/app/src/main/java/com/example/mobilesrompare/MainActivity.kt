package com.example.mobilesrcompare

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

  private lateinit var tv: TextView

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    tv = findViewById(R.id.tvStatus)
    val btnCpu = findViewById<Button>(R.id.btnCpu)
    val btnGpu = findViewById<Button>(R.id.btnGpu)

    btnCpu.setOnClickListener { runBench(useGpu = false) }
    btnGpu.setOnClickListener { runBench(useGpu = true) }
  }

  private fun runBench(useGpu: Boolean) {
    tv.text = "Running… (${if (useGpu) "GPU" else "CPU"})"

    Thread {
      val runner = TFLiteRunner(this, useGpu)

      // Defaults: warmup 10, runs 100
      val warmup = 10
      val runs = 100

      // Use a fixed synthetic LR input for stable benchmarking (avoid IO noise)
      val lr = ImageIO.makeRandomLR(128, 128)

      val results = mutableListOf<BenchmarkResult>()

      // 1) LPIENet-like (single model)
      results += runner.benchmarkLpienet(lr, warmup, runs)

      // 2) Edge 1-step diffusion (single model, uses lr_up as "noisy")
      results += runner.benchmarkEdge1Step(lr, warmup, runs)

      // 3) LPED latent 1-step (2 models: AE + Denoiser)
      results += runner.benchmarkLpedLatent1Step(lr, warmup, runs)

      runner.close()

      val csvPath = CsvLogger.writeResults(this, results, useGpu)
      runOnUiThread {
        tv.text = "Done (${if (useGpu) "GPU" else "CPU"})\nSaved: $csvPath\n\n" +
            results.joinToString("\n") { it.toPretty() }
      }
    }.start()
  }
}
