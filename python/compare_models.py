import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# Paths
# --------------------------------------------------
VAL_HR_DIR = "../dataset/common/val_HR"   # real validation images
IMAGE_LIST = sorted(os.listdir(VAL_HR_DIR))[:50]  # use 10 images

# --------------------------------------------------
# Image loaders
# --------------------------------------------------
def load_lr_image(path, size=(128, 128)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

def load_hr_image(path, size=(256, 256)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

def make_edge(lr):
    lr_up = tf.image.resize(lr, (256, 256), method="bicubic")
    gray = tf.image.rgb_to_grayscale(lr_up)
    edges = tf.image.sobel_edges(gray)
    mag = tf.sqrt(edges[..., 0]**2 + edges[..., 1]**2 + 1e-6)
    return mag.numpy()

# --------------------------------------------------
# Measure latency (MODEL-AWARE)
# --------------------------------------------------
def measure_latency(model, runs=10):
    times = []

    for img_name in IMAGE_LIST:
        img_path = os.path.join(VAL_HR_DIR, img_name)

        # ---- LPIENet-like (LR input) ----
        if model.name == "lpienet_like_x2":
            inputs = load_lr_image(img_path)

        # ---- Edge-aware diffusion (3 inputs) ----
        elif model.name == "edge_1step_diffusion_x2":
            lr = load_lr_image(img_path)
            edge = make_edge(lr)
            noisy = np.random.normal(0, 0.05, (1, 256, 256, 3)).astype(np.float32)
            inputs = [lr, edge, noisy]

        # ---- Tiny Autoencoder (HR input) ----
        elif model.name == "tiny_ae":
            inputs = load_hr_image(img_path)

        else:
            continue

        # warm-up
        model(inputs)

        start = time.time()
        for _ in range(runs):
            model(inputs)
        end = time.time()

        times.append((end - start) * 1000 / runs)

    return sum(times) / len(times)

# --------------------------------------------------
# Model info helpers
# --------------------------------------------------
def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def model_params(model):
    return model.count_params() / 1e6

# --------------------------------------------------
# Main comparison
# --------------------------------------------------
def main():
    out_dir = "./out"

    models = [
        ("LPIENet-like", "lpienet_like_x2.keras"),
        ("Edge-1Step-Diff", "edge_1step_diffusion_x2.keras"),
        ("Tiny-AE", "lped_tiny_ae.keras"),
    ]

    print("Model | Size(MB) | Params(M) | Latency(ms)")
    print("-" * 60)

    for name, fname in models:
        path = os.path.join(out_dir, fname)
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)

        size = model_size_mb(path)
        params = model_params(model)
        latency = measure_latency(model)

        print(f"{name:16s} | {size:7.2f} | {params:8.2f} | {latency:10.2f}")

if __name__ == "__main__":
    main()
