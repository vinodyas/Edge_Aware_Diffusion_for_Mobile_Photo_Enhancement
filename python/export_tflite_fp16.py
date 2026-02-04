import os
import tensorflow as tf

def export_fp16(keras_path, out_tflite):
    model = tf.keras.models.load_model(
        keras_path, compile=False, safe_mode=False
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite = converter.convert()

    with open(out_tflite, "wb") as f:
        f.write(tflite)
    print("Wrote:", out_tflite)

def main():
    out_dir = os.environ.get("OUT_DIR", "./out")
    tfl_dir = os.path.join(out_dir, "tflite")
    os.makedirs(tfl_dir, exist_ok=True)

    # ✅ Model 1: LPIENet
    export_fp16(
        os.path.join(out_dir, "lpienet_like_x2.keras"),
        os.path.join(tfl_dir, "lpienet_x2_fp16.tflite")
    )

    # ✅ Model 2: Edge-aware 1-step diffusion
    export_fp16(
        os.path.join(out_dir, "edge_1step_diffusion_x2.keras"),
        os.path.join(tfl_dir, "edge1step_x2_fp16.tflite")
    )

    # ✅ Model 3: LPED latent denoiser (ONLY this part is deployed)
    export_fp16(
        os.path.join(out_dir, "lped_latent_1step_denoiser.keras"),
        os.path.join(tfl_dir, "lped_latent_denoiser_fp16.tflite")
    )

if __name__ == "__main__":
    main()
