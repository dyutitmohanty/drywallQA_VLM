import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- HARDCODE YOUR PATHS HERE ---
INPUT_IMAGE_PATH = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_imgsCleaned/IMG_8253_JPG_jpg.rf.a47d48ff9e4efaa6b0adac3285459185__segment_tape_joint.jpg"
OUTPUT_SAVE_PATH = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/00018_comparison.png"


def process_and_compare(img_path, save_path):
    # 1. Load the image
    # We load with PIL to match your Dataset class logic, then convert to OpenCV
    image_pil = Image.open(img_path).convert("RGB")
    image_np = np.array(image_pil)

    # 2. Setup CLAHE (Matching your dataset parameters)
    # clipLimit=2.0, tileGridSize=(8, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 3. Apply CLAHE
    # CLAHE usually works on the L (Lightness) channel of LAB or Y (Luminance) of YCrCb
    # to avoid messing up colors.
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # 4. Create Side-by-Side Comparison
    plt.figure(figsize=(15, 7))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Original Image")
    plt.axis("off")

    # CLAHE
    plt.subplot(1, 2, 2)
    plt.imshow(image_clahe)
    plt.title("CLAHE Applied (Preprocessing)")
    plt.axis("off")

    # 5. Save and Show
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Comparison saved successfully to: {save_path}")


if __name__ == "__main__":
    process_and_compare(INPUT_IMAGE_PATH, OUTPUT_SAVE_PATH)
