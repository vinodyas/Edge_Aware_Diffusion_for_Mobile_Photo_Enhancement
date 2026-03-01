import os
import numpy as np
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

VAL_DIR = os.path.join(BASE_DIR, "dataset", "common", "val_HR")
MODEL_DIR = os.path.join(SCRIPT_DIR, "out")
SAVE_DIR = os.path.join(BASE_DIR, "figure2_png")
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# SELECT 5 TEST IMAGES
# (Change filenames if needed)
# --------------------------------------------------

IMAGES = {
    "row1_noise": "0804.png",
    "row2_texture": "0825.png",
    "row3_edges": "0846.png",
    "row4_dark": "0882.png",
    "row5_face": "0857.png",
}

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

lpienet = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "lpienet_like_x2.keras"),
    compile=False
)

tiny = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "lped_tiny_ae.keras"),
    compile=False
)

latent = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "lped_latent_1step_denoiser.keras"),
    compile=False
)

edge = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "edge_1step_diffusion_x2.keras"),
    compile=False
)

# --------------------------------------------------
# PROCESS EACH IMAGE
# --------------------------------------------------

for i, (key, filename) in enumerate(IMAGES.items(), start=1):

    print(f"Processing {filename}...")

    # Load HR
    hr_path = os.path.join(VAL_DIR, filename)
    hr = Image.open(hr_path).convert("RGB").resize((256, 256))
    hr_np = np.array(hr) / 255.0
    hr_tensor = hr_np[np.newaxis, ...].astype(np.float32)

    # --------------------------------------------------
    # CREATE DEGRADATION
    # --------------------------------------------------

    if "noise" in key:
        degraded = hr_np + np.random.normal(0, 0.1, hr_np.shape)
        degraded = np.clip(degraded, 0, 1)

    elif "dark" in key:
        degraded = hr_np * 0.3

    else:
        lr = hr.resize((128, 128), Image.BICUBIC)
        degraded = np.array(
            lr.resize((256, 256), Image.BICUBIC)
        ) / 255.0

    degraded_tensor = degraded[np.newaxis, ...].astype(np.float32)

    # Save input
    Image.fromarray((degraded * 255).astype(np.uint8)).save(
        os.path.join(SAVE_DIR, f"row{i}_input.png")
    )

    # --------------------------------------------------
    # TINY-AE (256x256 input)
    # --------------------------------------------------
    tiny_out = tiny(hr_tensor).numpy()[0]
    tiny_out = np.clip(tiny_out, 0, 1)

    Image.fromarray((tiny_out * 255).astype(np.uint8)).save(
        os.path.join(SAVE_DIR, f"row{i}_tiny.png")
    )

    # --------------------------------------------------
    # LPIENET (128x128 input)
    # --------------------------------------------------
    lp_input = tf.image.resize(degraded_tensor, (128, 128))
    lp_input = tf.cast(lp_input, tf.float32)
    lp_out = lpienet(lp_input).numpy()[0]
    lp_out = np.clip(lp_out, 0, 1)

    Image.fromarray((lp_out * 255).astype(np.uint8)).save(
        os.path.join(SAVE_DIR, f"row{i}_lpienet.png")
    )

    # --------------------------------------------------
    # LATENT MODEL
    # --------------------------------------------------
    z1 = tf.random.normal((1, 64, 64, 8))
    z2 = tf.random.normal((1, 64, 64, 8))

    latent_out = latent([z1, z2]).numpy()[0]

    latent_out = tf.image.resize(latent_out, (256, 256)).numpy()

    # Convert 8 channels → 3 channels
    latent_out = latent_out[..., :3]
    latent_out = np.clip(latent_out, 0, 1)

    Image.fromarray((latent_out * 255).astype(np.uint8)).save(
        os.path.join(SAVE_DIR, f"row{i}_latent.png")
    )

    # --------------------------------------------------
    # EDGE MODEL
    # --------------------------------------------------

    degraded_256 = tf.image.resize(degraded_tensor, (256, 256))
    lr_128 = tf.image.resize(degraded_tensor, (128, 128))

    gray_256 = tf.image.rgb_to_grayscale(degraded_256)
    sobel = tf.image.sobel_edges(gray_256)

    gx = sobel[..., 0]
    gy = sobel[..., 1]

    edge_map = tf.sqrt(gx**2 + gy**2 + 1e-6)

    noisy = tf.random.normal((1, 256, 256, 3), mean=0.0, stddev=0.05)

    edge_out = edge([lr_128, edge_map, noisy]).numpy()[0]
    edge_out = np.clip(edge_out, 0, 1)

    Image.fromarray((edge_out * 255).astype(np.uint8)).save(
        os.path.join(SAVE_DIR, f"row{i}_edge.png")
    )

    # --------------------------------------------------
    # SAVE GROUND TRUTH
    # --------------------------------------------------
    Image.fromarray((hr_np * 255).astype(np.uint8)).save(
        os.path.join(SAVE_DIR, f"row{i}_gt.png")
    )

print("\nAll 5 degradation cases generated successfully (PNG format).")