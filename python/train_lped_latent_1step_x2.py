import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataset_div2k import DIV2KPatchDataset

def build_tiny_autoencoder(hr_patch=256, latent_ch=8):
    inp = layers.Input((hr_patch, hr_patch, 3), name="hr")

    # encoder
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D(2)(x)  # 128
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(2)(x)  # 64
    x = layers.Conv2D(latent_ch, 3, padding="same", activation="linear", name="z")(x)  # [B,64,64,C]

    # decoder
    y = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    y = layers.UpSampling2D(2, interpolation="nearest")(y)  # 128
    y = layers.Conv2D(32, 3, padding="same", activation="relu")(y)
    y = layers.UpSampling2D(2, interpolation="nearest")(y)  # 256
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="recon")(y)

    ae = keras.Model(inp, out, name="tiny_ae")
    enc = keras.Model(inp, x, name="encoder")
    z_inp = layers.Input((64,64,latent_ch), name="z_in")
    # rebuild decoder by reusing layers (simple)
    # easiest: wrap decode as a new model using the same graph:
    # for clarity we define a separate decode model:
    yy = ae.layers[-5](z_inp)
    yy = ae.layers[-4](yy)
    yy = ae.layers[-3](yy)
    yy = ae.layers[-2](yy)
    yy = ae.layers[-1](yy)
    dec = keras.Model(z_inp, yy, name="decoder")
    return ae, enc, dec

def build_latent_1step_denoiser(latent_ch=8):
    # inputs: latent from LR-up (guidance) and noisy latent
    z_guid = layers.Input((64,64,latent_ch), name="z_guid")
    z_noisy = layers.Input((64,64,latent_ch), name="z_noisy")
    x = layers.Concatenate()([z_guid, z_noisy])

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    out = layers.Conv2D(latent_ch, 3, padding="same", activation="linear", name="z_clean")(x)
    return keras.Model([z_guid, z_noisy], out, name="latent_1step_denoiser")

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def main():
    div2k_root = os.environ.get("DIV2K_ROOT", "../dataset")
    out_dir = os.environ.get("OUT_DIR", "./out")
    os.makedirs(out_dir, exist_ok=True)

    lr_patch = 128
    hr_patch = 256
    ds = DIV2KPatchDataset(div2k_root, split="train", lr_patch=lr_patch, scale=2)

    # 1) train tiny AE
    ae, enc, dec = build_tiny_autoencoder(hr_patch=hr_patch, latent_ch=8)
    ae.compile(optimizer=keras.optimizers.Adam(2e-4), loss="mae", metrics=[psnr])

    for step in range(1500):
        lr, hr = ds.sample_batch(8)
        metrics = ae.train_on_batch(hr, hr, return_dict=True)
        if (step+1) % 100 == 0:
            print(f"[AE {step+1}/1500] " + " ".join([f"{k}={v:.4f}" for k,v in metrics.items()]))

    ae.save(os.path.join(out_dir, "lped_tiny_ae.keras"))
    print("Saved AE")

    # 2) train latent 1-step denoiser
    den = build_latent_1step_denoiser(latent_ch=8)
    den.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")

    for step in range(2000):
        lr, hr = ds.sample_batch(8)
        # get latent HR
        z_hr = enc.predict(hr, verbose=0)

        # "guidance" latent from LR-up: upsample LR to HR then encode
        lr_up = tf.image.resize(lr, (hr_patch, hr_patch), method="bicubic").numpy()
        z_guid = enc.predict(lr_up, verbose=0)

        noise = np.random.normal(0.0, 0.15, size=z_hr.shape).astype(np.float32)
        z_noisy = z_hr + noise

        den.train_on_batch([z_guid, z_noisy], z_hr)

        if (step+1) % 200 == 0:
            print(f"[DEN {step+1}/2000] ok")

    den.save(os.path.join(out_dir, "lped_latent_1step_denoiser.keras"))
    print("Saved denoiser")

if __name__ == "__main__":
    main()
