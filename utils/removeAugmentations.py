import cv2
import numpy as np
import os
import shutil

# --- PATHS ---
SOURCE_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_imgs"
CLEAN_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_imgs_clean"
MASKED_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_masked"

# --- SETTINGS ---
# 1. Darkness Sensitivity (0 to 255)
# Lower = stricter (only pure black).
# Higher = looser (includes dark greys, shadows, or "near-black").
# Try 30-50 if you want to catch "nearly black" pixels.
DARKNESS_LEVEL = 60

# 2. Quantity Threshold (0.0 to 1.0)
# 0.02 means if more than 2% of the ENTIRE image is "dark," it's masked.
BLACK_PERCENT_THRESHOLD = 0.1

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MASKED_DIR, exist_ok=True)


def is_masked(path):
    img = cv2.imread(path)
    if img is None:
        return False

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Count pixels below our DARKNESS_LEVEL
    # If DARKNESS_LEVEL is 40, any pixel from 0 (pure black) to 40 (dark grey) counts.
    dark_pixels = np.sum(gray < DARKNESS_LEVEL)

    # 3. Calculate proportion based on full image size
    total_pixels = gray.size
    proportion = dark_pixels / total_pixels

    # Optional: Print the result for debugging
    # print(f"Found {proportion:.2%} dark pixels in {os.path.basename(path)}")

    return proportion > BLACK_PERCENT_THRESHOLD


# --- EXECUTION ---
files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"Processing {len(files)} images...")
print(f"Filtering images with > {BLACK_PERCENT_THRESHOLD * 100}% pixels darker than level {DARKNESS_LEVEL}.")

for filename in files:
    src_path = os.path.join(SOURCE_DIR, filename)

    if is_masked(src_path):
        shutil.copy(src_path, os.path.join(MASKED_DIR, filename))
    else:
        shutil.copy(src_path, os.path.join(CLEAN_DIR, filename))

print(f"Done.")
