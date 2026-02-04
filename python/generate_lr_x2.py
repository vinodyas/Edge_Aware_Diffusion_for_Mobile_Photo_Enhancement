import os
import cv2

HR_DIR = "../dataset/common/HR"
LR_DIR = "../dataset/common/LR"
SCALE = 2

os.makedirs(LR_DIR, exist_ok=True)

files = os.listdir(HR_DIR)
print(f"Found {len(files)} HR images")

for img_name in files:
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    hr_path = os.path.join(HR_DIR, img_name)
    lr_path = os.path.join(LR_DIR, img_name)

    hr = cv2.imread(hr_path)
    if hr is None:
        print(f"Failed to read {img_name}")
        continue

    h, w = hr.shape[:2]
    lr = cv2.resize(hr, (w // SCALE, h // SCALE), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(lr_path, lr)

print("✅ LR image generation completed")
