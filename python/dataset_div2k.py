import os, glob, random
import numpy as np
from PIL import Image

def _load_img(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def _bicubic_resize(img_np, size_wh):
    img = Image.fromarray(np.clip(img_np * 255.0, 0, 255).astype(np.uint8))
    img = img.resize(size_wh, Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0

class DIV2KPatchDataset:
    def __init__(self, div2k_root, split="train", lr_patch=128, scale=2):
        self.scale = scale
        self.lr_patch = lr_patch
        self.hr_patch = lr_patch * scale

        if split == "train":
            folder = os.path.join(div2k_root, "common", "train_HR")
        else:
            folder = os.path.join(div2k_root, "common", "val_HR")

        self.paths = sorted(
            glob.glob(os.path.join(folder, "*.png")) +
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.jpeg"))
            )
        if not self.paths:
            raise FileNotFoundError(f"No PNGs found in {folder}")

    def sample_batch(self, batch_size=8):
        lr_batch = []
        hr_batch = []
        for _ in range(batch_size):
            p = random.choice(self.paths)
            hr = _load_img(p)
            H, W, _ = hr.shape
            if H < self.hr_patch or W < self.hr_patch:
                hr = _bicubic_resize(hr, (max(W, self.hr_patch), max(H, self.hr_patch)))
                H, W, _ = hr.shape
            y = random.randint(0, H - self.hr_patch)
            x = random.randint(0, W - self.hr_patch)
            hr_crop = hr[y:y+self.hr_patch, x:x+self.hr_patch, :]
            lr_crop = _bicubic_resize(hr_crop, (self.lr_patch, self.lr_patch))
            lr_batch.append(lr_crop)
            hr_batch.append(hr_crop)
        return np.stack(lr_batch), np.stack(hr_batch)
