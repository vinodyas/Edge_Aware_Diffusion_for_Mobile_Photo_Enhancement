import os
import numpy as np
import tensorflow as tf
from PIL import Image

def load_img(path, size=128):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    x = np.array(img, dtype=np.float32)/255.0
    return x[None, ...]

def save_img(x, path):
    x = np.clip(x[0]*255.0,0,255).astype(np.uint8)
    Image.fromarray(x).save(path)

def main():
    out_dir = os.environ.get("OUT_DIR","./out")
    img_path = os.environ.get("IMG","./sample.png")

    lr = load_img(img_path, 128)
    lr_up = tf.image.resize(lr, (256,256), method="bicubic").numpy()

    lp = tf.keras.models.load_model(os.path.join(out_dir,"lpienet_like_x2.keras"), compile=False)
    hr_lp = lp.predict(lr, verbose=0)
    save_img(hr_lp, os.path.join(out_dir,"lp_out.png"))

    edge = tf.keras.models.load_model(os.path.join(out_dir,"edge_1step_diffusion_x2.keras"), compile=False)
    hr_edge = edge.predict([lr, lr_up], verbose=0)  # use lr_up as "noisy"
    save_img(hr_edge, os.path.join(out_dir,"edge_out.png"))

    print("Wrote outputs to", out_dir)

if __name__ == "__main__":
    main()
