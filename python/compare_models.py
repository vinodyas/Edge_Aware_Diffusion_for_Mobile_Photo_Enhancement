import os
import time
import csv
import numpy as np
import tensorflow as tf
import torch
import lpips
from PIL import Image
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2
)

# ==========================================================
# DATASET
# ==========================================================

VAL_HR_DIR = "../dataset/common/valed_HR"

IMAGE_LIST = sorted(
    os.listdir(VAL_HR_DIR)
)[:50]


# ==========================================================
# LPIPS MODEL
# ==========================================================

# LPIPS is a perceptual image-quality metric.
# Lower LPIPS values indicate better perceptual similarity.

LPIPS_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("LPIPS device:", LPIPS_DEVICE)

lpips_model = lpips.LPIPS(
    net="alex"
).to(LPIPS_DEVICE)

lpips_model.eval()


# ==========================================================
# LPIPS CALCULATION
# ==========================================================

def calculate_lpips(sr, hr):
    """
    Calculate LPIPS between the super-resolved image
    and the high-resolution reference image.

    Input:
        sr: [1, H, W, 3], values in [0, 1]
        hr: [1, H, W, 3], values in [0, 1]

    LPIPS expects:
        [N, C, H, W], values in [-1, 1]

    Lower LPIPS is better.
    """

    # Convert NumPy arrays to PyTorch tensors

    sr_tensor = torch.from_numpy(
        np.asarray(sr, dtype=np.float32)
    )

    hr_tensor = torch.from_numpy(
        np.asarray(hr, dtype=np.float32)
    )

    # Convert:
    #
    # [N, H, W, C]
    #
    # to:
    #
    # [N, C, H, W]

    sr_tensor = sr_tensor.permute(
        0, 3, 1, 2
    )

    hr_tensor = hr_tensor.permute(
        0, 3, 1, 2
    )

    # LPIPS expects values in [-1, 1]
    #
    # Current images are in [0, 1]

    sr_tensor = (
        sr_tensor * 2.0
    ) - 1.0

    hr_tensor = (
        hr_tensor * 2.0
    ) - 1.0

    # Move to CPU/GPU

    sr_tensor = sr_tensor.to(
        LPIPS_DEVICE
    )

    hr_tensor = hr_tensor.to(
        LPIPS_DEVICE
    )

    # Calculate LPIPS

    with torch.no_grad():

        score = lpips_model(
            sr_tensor,
            hr_tensor
        )

    return float(
        score.mean().item()
    )


# ==========================================================
# IMAGE LOADERS
# ==========================================================

def load_lr_image(
    path,
    size=(128, 128)
):

    img = Image.open(
        path
    ).convert("RGB")

    img = img.resize(
        size,
        Image.BICUBIC
    )

    arr = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    return arr[
        np.newaxis,
        ...
    ]


def load_hr_image(
    path,
    size=(256, 256)
):

    img = Image.open(
        path
    ).convert("RGB")

    img = img.resize(
        size,
        Image.BICUBIC
    )

    arr = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    return arr[
        np.newaxis,
        ...
    ]


# ==========================================================
# EDGE
# ==========================================================

def make_edge(lr):

    lr_up = tf.image.resize(
        lr,
        (256, 256),
        method="bicubic"
    )

    gray = tf.image.rgb_to_grayscale(
        lr_up
    )

    edges = tf.image.sobel_edges(
        gray
    )

    mag = tf.sqrt(
        edges[..., 0] ** 2
        +
        edges[..., 1] ** 2
        +
        1e-6
    )

    mag = tf.reduce_mean(
        mag,
        axis=-1
    )

    return mag.numpy()


# ==========================================================
# EVALUATION
# Latency + PSNR + SSIM + LPIPS
# ==========================================================

def evaluate_model(
    model,
    runs=5
):

    times = []

    psnr_list = []

    ssim_list = []

    lpips_list = []


    for img_name in IMAGE_LIST:

        img_path = os.path.join(
            VAL_HR_DIR,
            img_name
        )

        hr = load_hr_image(
            img_path
        )


        # ==================================================
        # INPUT PREPARATION
        # ==================================================

        if model.name == "lpienet_like_x2":

            inputs = load_lr_image(
                img_path
            )


        elif model.name == "edge_1step_diffusion_x2":

            lr = load_lr_image(
                img_path
            )

            edge = make_edge(
                lr
            )

            noisy = np.random.normal(
                0,
                0.05,
                (1, 256, 256, 3)
            ).astype(
                np.float32
            )

            inputs = [
                lr,
                edge,
                noisy
            ]


        elif model.name == "tiny_ae":

            inputs = hr


        elif model.name == "latent_1step_denoiser":

            # Keep the same latent-input procedure
            # from your original compare_models.py.

            hr_small = tf.image.resize(
                hr,
                (64, 64)
            )

            latent = tf.image.rgb_to_grayscale(
                hr_small
            )

            latent = tf.tile(
                latent,
                [1, 1, 1, 8]
            )

            inputs = [
                latent,
                latent
            ]


        else:

            continue


        # ==================================================
        # WARM-UP
        # ==================================================

        model(
            inputs
        )


        # ==================================================
        # LATENCY
        # ==================================================

        # LPIPS is NOT included in latency.

        start = time.time()

        for _ in range(runs):

            sr = model(
                inputs
            )

        end = time.time()

        times.append(
            (end - start)
            * 1000
            / runs
        )


        # ==================================================
        # METRIC FIX FOR LATENT-1STEP
        # ==================================================

        if model.name == "latent_1step_denoiser":

            # Keep your existing conversion.

            sr = sr[
                ...,
                :3
            ]

            sr = tf.image.resize(
                sr,
                (256, 256)
            )


        # ==================================================
        # CLIP OUTPUT
        # ==================================================

        sr = tf.clip_by_value(
            sr,
            0.0,
            1.0
        )


        # ==================================================
        # ENSURE SAME SHAPE
        # ==================================================

        if sr.shape != hr.shape:

            sr = tf.image.resize(
                sr,
                (256, 256)
            )


        # ==================================================
        # PSNR
        # ==================================================

        psnr = tf.image.psnr(
            sr,
            hr,
            max_val=1.0
        )


        # ==================================================
        # SSIM
        # ==================================================

        ssim = tf.image.ssim(
            sr,
            hr,
            max_val=1.0
        )


        # ==================================================
        # LPIPS
        # ==================================================

        lpips_score = calculate_lpips(
            sr.numpy(),
            hr
        )


        # ==================================================
        # SAVE METRICS
        # ==================================================

        psnr_list.append(
            float(
                psnr.numpy()[0]
            )
        )

        ssim_list.append(
            float(
                ssim.numpy()[0]
            )
        )

        lpips_list.append(
            lpips_score
        )


    # ======================================================
    # AVERAGE
    # ======================================================

    return (

        float(
            np.mean(times)
        ),

        float(
            np.mean(psnr_list)
        ),

        float(
            np.mean(ssim_list)
        ),

        float(
            np.mean(lpips_list)
        )

    )


# ==========================================================
# GFLOPs
# ==========================================================

def compute_gflops(
    model
):

    # ------------------------------------------------------
    # LPIENet-like
    # ------------------------------------------------------

    if model.name == "lpienet_like_x2":

        @tf.function
        def forward(x):

            return model(x)

        signature = [

            tf.TensorSpec(
                [1, 128, 128, 3],
                tf.float32
            )

        ]


    # ------------------------------------------------------
    # Edge-1Step Diffusion
    # ------------------------------------------------------

    elif model.name == "edge_1step_diffusion_x2":

        @tf.function
        def forward(
            lr,
            edge,
            noisy
        ):

            return model(
                [
                    lr,
                    edge,
                    noisy
                ]
            )

        signature = [

            tf.TensorSpec(
                [1, 128, 128, 3],
                tf.float32
            ),

            tf.TensorSpec(
                [1, 256, 256, 1],
                tf.float32
            ),

            tf.TensorSpec(
                [1, 256, 256, 3],
                tf.float32
            )

        ]


    # ------------------------------------------------------
    # Tiny-AE
    # ------------------------------------------------------

    elif model.name == "tiny_ae":

        @tf.function
        def forward(x):

            return model(x)

        signature = [

            tf.TensorSpec(
                [1, 256, 256, 3],
                tf.float32
            )

        ]


    # ------------------------------------------------------
    # Latent-1Step
    # ------------------------------------------------------

    elif model.name == "latent_1step_denoiser":

        @tf.function
        def forward(
            z1,
            z2
        ):

            return model(
                [
                    z1,
                    z2
                ]
            )

        signature = [

            tf.TensorSpec(
                [1, 64, 64, 8],
                tf.float32
            ),

            tf.TensorSpec(
                [1, 64, 64, 8],
                tf.float32
            )

        ]


    else:

        return 0.0


    # ------------------------------------------------------
    # CONCRETE FUNCTION
    # ------------------------------------------------------

    concrete = (
        forward.get_concrete_function(
            *signature
        )
    )


    # ------------------------------------------------------
    # FREEZE MODEL
    # ------------------------------------------------------

    frozen_func = (
        convert_variables_to_constants_v2(
            concrete
        )
    )


    # ------------------------------------------------------
    # FLOP PROFILING
    # ------------------------------------------------------

    run_meta = (
        tf.compat.v1.RunMetadata()
    )

    opts = (
        tf.compat.v1.profiler
        .ProfileOptionBuilder
        .float_operation()
    )


    flops = (
        tf.compat.v1.profiler.profile(
            graph=frozen_func.graph,
            run_meta=run_meta,
            cmd="op",
            options=opts
        )
    )


    if flops is None:

        return 0.0


    return (
        flops.total_float_ops
        / 1e9
    )


# ==========================================================
# MODEL INFO
# ==========================================================

def model_size_mb(
    path
):

    return (
        os.path.getsize(path)
        /
        (1024 * 1024)
    )


def model_params(
    model
):

    return (
        model.count_params()
        /
        1e6
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    out_dir = "./out"


    # ======================================================
    # FOUR MODELS
    # ======================================================

    models = [

        (
            "LPIENet-like",
            "lpienet_like_x2.keras"
        ),

        (
            "Edge-1Step Diffusion",
            "edge_1step_diffusion_x2.keras"
        ),

        (
            "Tiny-AE",
            "lped_tiny_ae.keras"
        ),

        (
            "Latent-1Step",
            "lped_latent_1step_denoiser.keras"
        )

    ]


    # ======================================================
    # HEADER
    # ======================================================

    print()

    print(
        "Model | Size(MB) | Params(M) | "
        "GFLOPs | Latency(ms) | PSNR | SSIM | LPIPS"
    )

    print(
        "-" * 120
    )


    # ======================================================
    # RESULTS
    # ======================================================

    results = []


    # ======================================================
    # RUN EACH MODEL
    # ======================================================

    for name, fname in models:

        path = os.path.join(
            out_dir,
            fname
        )


        # --------------------------------------------------
        # CHECK FILE
        # --------------------------------------------------

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Model file not found: {path}"
            )


        # --------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------

        model = (
            tf.keras.models.load_model(
                path,
                compile=False,
                safe_mode=False
            )
        )


        # --------------------------------------------------
        # MODEL SIZE
        # --------------------------------------------------

        size = model_size_mb(
            path
        )


        # --------------------------------------------------
        # PARAMETERS
        # --------------------------------------------------

        params = model_params(
            model
        )


        # --------------------------------------------------
        # GFLOPs
        # --------------------------------------------------

        gflops = compute_gflops(
            model
        )


        # --------------------------------------------------
        # EVALUATE
        # --------------------------------------------------

        (
            latency,
            avg_psnr,
            avg_ssim,
            avg_lpips
        ) = evaluate_model(
            model
        )


        # --------------------------------------------------
        # PRINT
        # --------------------------------------------------

        print(
            f"{name:20s} | "
            f"{size:8.2f} | "
            f"{params:9.2f} | "
            f"{gflops:7.2f} | "
            f"{latency:11.2f} | "
            f"{avg_psnr:10.6f} | "
            f"{avg_ssim:10.6f} | "
            f"{avg_lpips:8.6f}"
        )


        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------

        results.append(
            [
                name,
                f"{size:.2f}",
                f"{params:.2f}",
                f"{gflops:.2f}",
                f"{latency:.2f}",
                f"{avg_psnr:.6f}",
                f"{avg_ssim:.6f}",
                f"{avg_lpips:.6f}"
            ]
        )


    # ======================================================
    # SAVE CSV
    # ======================================================

    csv_path = os.path.join(
        out_dir,
        "model_comparison_lpips.csv"
    )


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "Model",
                "Size_MB",
                "Parameters_M",
                "GFLOPs",
                "Latency_ms",
                "PSNR",
                "SSIM",
                "LPIPS"
            ]
        )


        writer.writerows(
            results
        )


    # ======================================================
    # END
    # ======================================================

    print()

    print(
        "Results saved to:"
    )

    print(
        csv_path
    )

    print()

    print(
        "PSNR: Higher is better"
    )

    print(
        "SSIM: Higher is better"
    )

    print(
        "LPIPS: Lower is better"
    )

    print()

    print(
        "================ End of Report ================"
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()

