package com.example.mobilesrcompare

import kotlin.math.roundToLong

data class BenchmarkResult(
  val modelName: String,
  val backend: String,
  val meanMs: Double,
  val p50Ms: Double,
  val p90Ms: Double,
  val p95Ms: Double,
  val runs: Int
) {
  fun toPretty(): String =
    "$modelName [$backend] mean=${meanMs.fmt()} p50=${p50Ms.fmt()} p90=${p90Ms.fmt()} p95=${p95Ms.fmt()} (n=$runs)"
}

private fun Double.fmt(): String = String.format("%.2fms", this)

object Stats {
  fun summarize(timesMs: List<Double>): Quad {
    val sorted = timesMs.sorted()
    fun pct(p: Double): Double {
      val idx = ((p * (sorted.size - 1)).roundToLong()).toInt().coerceIn(0, sorted.size - 1)
      return sorted[idx]
    }
    val mean = timesMs.average()
    return Quad(mean, pct(0.50), pct(0.90), pct(0.95))
  }

  data class Quad(val mean: Double, val p50: Double, val p90: Double, val p95: Double)
}
