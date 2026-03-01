import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# --------------------------------------------------
# DATASET
# --------------------------------------------------
VAL_HR_DIR = "../dataset/common/val_HR"
IMAGE_LIST = sorted(os.listdir(VAL_HR_DIR))[:50]


# --------------------------------------------------
# IMAGE LOADERS
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
# LATENCY + PSNR + SSIM
# --------------------------------------------------
def evaluate_model(model, runs=5):

    times = []
    psnr_list = []
    ssim_list = []

    for img_name in IMAGE_LIST:

        img_path = os.path.join(VAL_HR_DIR, img_name)
        hr = load_hr_image(img_path)

        # -------- INPUT HANDLING --------
        if model.name == "lpienet_like_x2":
            inputs = load_lr_image(img_path)

        elif model.name == "edge_1step_diffusion_x2":
            lr = load_lr_image(img_path)
            edge = make_edge(lr)
            noisy = np.random.normal(0, 0.05, (1, 256, 256, 3)).astype(np.float32)
            inputs = [lr, edge, noisy]

        elif model.name == "tiny_ae":
            inputs = hr

        elif model.name == "latent_1step_denoiser":
            z_guid = np.random.normal(0, 1, (1, 64, 64, 8)).astype(np.float32)
            z_cond = np.random.normal(0, 1, (1, 64, 64, 8)).astype(np.float32)
            inputs = [z_guid, z_cond]

        else:
            continue

        # Warm-up
        model(inputs)

        # -------- LATENCY --------
        start = time.time()
        for _ in range(runs):
            sr = model(inputs)
        end = time.time()

        times.append((end - start) * 1000 / runs)

        # -------- METRICS (ONLY FOR IMAGE MODELS) --------
        if model.name != "latent_1step_denoiser":

            sr = tf.clip_by_value(sr, 0.0, 1.0)

            psnr = tf.image.psnr(sr, hr, max_val=1.0)
            ssim = tf.image.ssim(sr, hr, max_val=1.0)

            psnr_list.append(psnr.numpy()[0])
            ssim_list.append(ssim.numpy()[0])

    avg_latency = np.mean(times)

    if len(psnr_list) > 0:
        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))

    else:
        avg_psnr = 0.0
        avg_ssim = 0.0

    return avg_latency, avg_psnr, avg_ssim


# --------------------------------------------------
# GFLOPs
# --------------------------------------------------
def compute_gflops(model):

    if model.name == "lpienet_like_x2":

        @tf.function
        def forward(x):
            return model(x)

        signature = [tf.TensorSpec([1, 128, 128, 3], tf.float32)]

    elif model.name == "edge_1step_diffusion_x2":

        @tf.function
        def forward(lr, edge, noisy):
            return model([lr, edge, noisy])

        signature = [
            tf.TensorSpec([1, 128, 128, 3], tf.float32),
            tf.TensorSpec([1, 256, 256, 1], tf.float32),
            tf.TensorSpec([1, 256, 256, 3], tf.float32),
        ]

    elif model.name == "tiny_ae":

        @tf.function
        def forward(x):
            return model(x)

        signature = [tf.TensorSpec([1, 256, 256, 3], tf.float32)]

    elif model.name == "latent_1step_denoiser":

        @tf.function
        def forward(z_guid, z_cond):
            return model([z_guid, z_cond])

        signature = [
            tf.TensorSpec([1, 64, 64, 8], tf.float32),
            tf.TensorSpec([1, 64, 64, 8], tf.float32),
        ]

    else:
        return 0.0

    concrete = forward.get_concrete_function(*signature)
    frozen_func = convert_variables_to_constants_v2(concrete)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph,
        run_meta=run_meta,
        cmd="op",
        options=opts
    )

    if flops is None:
        return 0.0

    return flops.total_float_ops / 1e9


# --------------------------------------------------
# MODEL INFO
# --------------------------------------------------
def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def model_params(model):
    return model.count_params() / 1e6


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    out_dir = "./out"

    models = [
        ("LPIENet-like", "lpienet_like_x2.keras"),
        ("Edge-1Step Diffusion", "edge_1step_diffusion_x2.keras"),
        ("Tiny-AE", "lped_tiny_ae.keras"),
        ("Latent-1Step", "lped_latent_1step_denoiser.keras"),
    ]

    print("\nModel | Size(MB) | Params(M) | GFLOPs | Latency(ms) | PSNR | SSIM")
    print("-" * 110)

    for name, fname in models:

        path = os.path.join(out_dir, fname)
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)

        size = model_size_mb(path)
        params = model_params(model)
        gflops = compute_gflops(model)
        latency, avg_psnr, avg_ssim = evaluate_model(model)

        print(f"{name:20s} | {size:8.2f} | {params:9.2f} | {gflops:7.2f} | {latency:11.2f} | {avg_psnr:10.10f} | {avg_ssim:10.6f}")

    print("\n================ End of Report ================")


if __name__ == "__main__":
    main()