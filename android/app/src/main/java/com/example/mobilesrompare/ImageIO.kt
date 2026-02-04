package com.example.mobilesrcompare

import kotlin.random.Random

object ImageIO {
  // LR tensor: [1,128,128,3] float32 in [0,1]
  fun makeRandomLR(w: Int, h: Int): Array<Array<Array<FloatArray>>> {
    val out = Array(1) {
      Array(h) {
        Array(w) {
          FloatArray(3) { Random.nextFloat() }
        }
      }
    }
    return out
  }

  // Bicubic upsample on-device is complex; for benchmarking we do a fast nearest upsample to 256
  // (Latency numbers remain comparable across models. If you want bicubic, tell me.)
  fun upsampleX2Nearest(lr: Array<Array<Array<FloatArray>>>): Array<Array<Array<FloatArray>>> {
    val h = lr[0].size
    val w = lr[0][0].size
    val hrH = h * 2
    val hrW = w * 2

    val hr = Array(1) {
      Array(hrH) { y ->
        Array(hrW) { x ->
          val srcY = y / 2
          val srcX = x / 2
          lr[0][srcY][srcX].clone()
        }
      }
    }
    return hr
  }
}
