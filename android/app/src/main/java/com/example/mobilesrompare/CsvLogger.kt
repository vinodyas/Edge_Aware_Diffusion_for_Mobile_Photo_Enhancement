package com.example.mobilesrcompare

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

object CsvLogger {
  fun writeResults(ctx: Context, results: List<BenchmarkResult>, useGpu: Boolean): String {
    val dir = ctx.getExternalFilesDir(null) ?: ctx.filesDir
    val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val file = File(dir, "bench_${if (useGpu) "gpu" else "cpu"}_$ts.csv")

    file.bufferedWriter().use { w ->
      w.write("model,backend,mean_ms,p50_ms,p90_ms,p95_ms,runs\n")
      for (r in results) {
        w.write("${r.modelName},${r.backend},${r.meanMs},${r.p50Ms},${r.p90Ms},${r.p95Ms},${r.runs}\n")
      }
    }
    return file.absolutePath
  }
}
