import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataset_div2k import DIV2KPatchDataset


def conv_block(x, ch, name):
    x = layers.Conv2D(ch, 3, padding="same", activation="relu", name=f"{name}_c1")(x)
    x = layers.Conv2D(ch, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
    return x


def build_edge_1step_x2(lr_patch=128, base=32):
    # ----- inputs -----
    lr_inp = layers.Input((lr_patch, lr_patch, 3), name="lr")
    edge_inp = layers.Input((lr_patch * 2, lr_patch * 2, 1), name="edge")
    noisy_hr = layers.Input((lr_patch * 2, lr_patch * 2, 3), name="noisy_hr")

    # ----- upsample LR -----
    lr_up = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="lr_up"
    )(lr_inp)


    # ----- concatenate -----
    x = layers.Concatenate(name="concat")([lr_up, edge_inp, noisy_hr])

    # ----- UNet-like backbone -----
    d1 = conv_block(x, base, "d1")
    p1 = layers.MaxPool2D(2)(d1)

    d2 = conv_block(p1, base * 2, "d2")
    p2 = layers.MaxPool2D(2)(d2)

    b = conv_block(p2, base * 4, "b")

    u2 = layers.UpSampling2D(2)(b)
    u2 = layers.Concatenate()([u2, d2])
    u2 = conv_block(u2, base * 2, "u2b")

    u1 = layers.UpSampling2D(2)(u2)
    u1 = layers.Concatenate()([u1, d1])
    u1 = conv_block(u1, base, "u1b")

    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="hr")(u1)

    # ✅ correct return
    return keras.Model(
        inputs=[lr_inp, edge_inp, noisy_hr],
        outputs=out,
        name="edge_1step_diffusion_x2"
    )


def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def main():
    div2k_root = os.environ.get("DIV2K_ROOT", "../dataset")
    out_dir = os.environ.get("OUT_DIR", "./out")
    os.makedirs(out_dir, exist_ok=True)

    lr_patch = 128
    ds = DIV2KPatchDataset(div2k_root, split="train", lr_patch=lr_patch, scale=2)
    model = build_edge_1step_x2(lr_patch=lr_patch, base=32)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="mae",
        metrics=[psnr]
    )

    steps = 2000
    batch = 6

    for step in range(steps):
        lr, hr = ds.sample_batch(batch)

        # noisy HR
        noise = np.random.normal(0.0, 0.08, size=hr.shape).astype(np.float32)
        noisy = np.clip(hr + noise, 0.0, 1.0)

        # edge map (outside model)
        lr_up = tf.image.resize(lr, (256, 256), method="bilinear")
        gray = tf.image.rgb_to_grayscale(lr_up)
        edges = tf.image.sobel_edges(gray)
        edge_mag = tf.sqrt(edges[..., 0]**2 + edges[..., 1]**2 + 1e-6)

        model.train_on_batch([lr, edge_mag, noisy], hr)

        if (step + 1) % 100 == 0:
            print(f"[{step+1}/{steps}] ok")

    model.save(os.path.join(out_dir, "edge_1step_diffusion_x2.keras"))
    print("Saved edge model")


if __name__ == "__main__":
    main()
