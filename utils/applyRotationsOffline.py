import cv2
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
IMG_DIR = r"/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/WallFeaturesDataset/train/images"
LBL_DIR = r"/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/WallFeaturesDataset/train/labels"


def rotate_image(mat, angle):
    """Rotates an image by any angle without cropping corners."""
    height, width = mat.shape[:2]
    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions to ensure no data is lost
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    new_w = int(height * abs_sin + width * abs_cos)
    new_h = int(height * abs_cos + width * abs_sin)

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    # INTER_NEAREST is critical for masks to keep class values pure
    return cv2.warpAffine(mat, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST)


def augment_data():
    img_path = Path(IMG_DIR)
    lbl_path = Path(LBL_DIR)

    # Filtering for original files only to avoid re-rotating already augmented files
    # This checks that the filename doesn't already contain '_rot'
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in img_path.iterdir() if f.suffix.lower() in valid_extensions and "_rot" not in f.name]

    print(f"Processing {len(image_files)} original images...")

    for img_file in image_files:
        # Match mask: labels are .png
        lbl_file = lbl_path / (img_file.stem + ".png")

        if not lbl_file.exists():
            print(f"Skipping {img_file.name}: Mask not found.")
            continue

        img = cv2.imread(str(img_file))
        lbl = cv2.imread(str(lbl_file), cv2.IMREAD_UNCHANGED)

        # Logic for new multipliers:
        if "__segment_tape_joint" in img_file.name:
            # 7 new + 1 original = 8x total
            angles = [45, 90, 135, 180, 225, 270, 315]
        else:
            # 4 new + 1 original = 5x total
            angles = [72, 144, 216, 288]

        for angle in angles:
            img_rot = rotate_image(img, angle)
            lbl_rot = rotate_image(lbl, angle)

            # Save with rotation suffix
            new_img_name = f"{img_file.stem}_rot{angle}{img_file.suffix}"
            new_lbl_name = f"{lbl_file.stem}_rot{angle}.png"

            cv2.imwrite(str(img_path / new_img_name), img_rot)
            cv2.imwrite(str(lbl_path / new_lbl_name), lbl_rot)

    print("Augmentation complete! Check your folders for the new files.")


if __name__ == "__main__":
    augment_data()
