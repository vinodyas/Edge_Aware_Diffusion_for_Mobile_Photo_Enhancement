import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataset_div2k import DIV2KPatchDataset

def residual_block(x, channels, name):
    skip = x
    x = layers.Conv2D(channels, 3, padding="same", activation="relu", name=f"{name}_c1")(x)
    x = layers.Conv2D(channels, 3, padding="same", name=f"{name}_c2")(x)
    x = layers.Add(name=f"{name}_add")([skip, x])
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x

def build_lpienet_like_x2(lr_patch=128, channels=32, num_blocks=6):
    inp = layers.Input(shape=(lr_patch, lr_patch, 3), name="lr")
    x = layers.Conv2D(channels, 3, padding="same", activation="relu", name="stem")(inp)

    # progressive residual refinement
    for i in range(num_blocks):
        x = residual_block(x, channels, name=f"rb{i+1}")

    # upsample x2
    x = layers.Conv2D(channels * 4, 3, padding="same", activation="relu", name="pre_up")(x)
    x = layers.Reshape(
    (lr_patch, lr_patch, 2, 2, channels),
    name="ps_reshape"
)(x)

    x = layers.Permute((1, 3, 2, 4, 5), name="ps_permute")(x)

    x = layers.Reshape(
    (lr_patch * 2, lr_patch * 2, channels),
    name="pixel_shuffle_x2"
)(x)


    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="hr")(x)
    return keras.Model(inp, out, name="lpienet_like_x2")

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def main():
    div2k_root = os.environ.get("DIV2K_ROOT", "../dataset")
    out_dir = os.environ.get("OUT_DIR", "./out")
    os.makedirs(out_dir, exist_ok=True)

    lr_patch = 128
    ds = DIV2KPatchDataset(div2k_root, split="train", lr_patch=lr_patch, scale=2)
    model = build_lpienet_like_x2(lr_patch=lr_patch)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="mae",
        metrics=[psnr, ssim],
    )

    steps = 2000  # adjust up for better quality
    batch = 8

    for step in range(steps):
        lr, hr = ds.sample_batch(batch)
        metrics = model.train_on_batch(lr, hr, return_dict=True)
        if (step + 1) % 100 == 0:
            print(f"[{step+1}/{steps}] " + " ".join([f"{k}={v:.4f}" for k,v in metrics.items()]))

    save_path = os.path.join(out_dir, "lpienet_like_x2.keras")
    model.save(save_path)
    print("Saved:", save_path)

if __name__ == "__main__":
    main()
