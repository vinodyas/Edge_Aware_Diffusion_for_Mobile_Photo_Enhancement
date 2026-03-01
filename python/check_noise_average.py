import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Path to validation HR images
VAL_HR_DIR = "../dataset/common/val_HR"

image_list = sorted(os.listdir(VAL_HR_DIR))[:50]

psnr_list = []
ssim_list = []

for img_name in image_list:

    img_path = os.path.join(VAL_HR_DIR, img_name)

    # Load original image
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    hr = np.array(img, dtype=np.float32) / 255.0
    hr = hr[np.newaxis, ...]

    # Create noisy version
    noise = np.random.normal(0, 0.05, hr.shape).astype(np.float32)
    noisy = hr + noise
    noisy = np.clip(noisy, 0.0, 1.0)

    # Compute PSNR and SSIM
    psnr = tf.image.psnr(noisy, hr, max_val=1.0)
    ssim = tf.image.ssim(noisy, hr, max_val=1.0)

    psnr_list.append(psnr.numpy()[0])
    ssim_list.append(ssim.numpy()[0])

# Compute average
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)

print("Average PSNR (Original vs Noisy):", avg_psnr)
print("Average SSIM (Original vs Noisy):", avg_ssim)