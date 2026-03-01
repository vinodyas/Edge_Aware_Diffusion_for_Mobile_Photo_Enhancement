import numpy as np
import tensorflow as tf
from PIL import Image
import os

# -------- PATH TO ONE HR IMAGE --------
IMG_PATH = "../dataset/common/val_HR/0804.png"

# -------- LOAD ORIGINAL --------
img = Image.open(IMG_PATH).convert("RGB").resize((256,256))
hr = np.array(img, dtype=np.float32) / 255.0
hr = hr[np.newaxis, ...]

# -------- CREATE NOISY VERSION --------
noise = np.random.normal(0, 0.05, hr.shape).astype(np.float32)
noisy = hr + noise
noisy = np.clip(noisy, 0.0, 1.0)

# -------- COMPUTE METRICS --------
psnr_value = tf.image.psnr(noisy, hr, max_val=1.0)
ssim_value = tf.image.ssim(noisy, hr, max_val=1.0)

print("PSNR (Original vs Noisy):", psnr_value.numpy()[0])
print("SSIM (Original vs Noisy):", ssim_value.numpy()[0])