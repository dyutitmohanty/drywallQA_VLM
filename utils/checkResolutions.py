import os
from PIL import Image
import matplotlib.pyplot as plt

# --- HARDCODED PATHS ---
IMAGE_FOLDER = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/WallFeaturesDataset/train/images"
OUTPUT_PLOT_PATH = "/home/ml_team/Documents/Dyutit/drywall_ductape/resolution_analysis.png"


def analyze_resolutions(folder_path, save_path):
    widths = []
    heights = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    print(f"Scanning folder: {folder_path}...")

    # Iterate through files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                print(f"Could not open {filename}: {e}")

    if not widths:
        print("No images found in the specified path.")
        return

    # Create Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Scatter Plot (Width vs Height) ---
    ax1.scatter(widths, heights, alpha=0.5, color="blue", edgecolors="white")
    ax1.set_title("Image Resolutions (Scatter)")
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Height (pixels)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Add a diagonal line for square images
    max_val = max(max(widths), max(heights))
    ax1.plot([0, max_val], [0, max_val], "r--", alpha=0.3, label="Square Ratio")
    ax1.legend()

    # --- Histogram (Distribution) ---
    ax2.hist(widths, bins=30, alpha=0.7, color="green", label="Widths", edgecolor="black")
    ax2.hist(heights, bins=30, alpha=0.5, color="orange", label="Heights", edgecolor="black")
    ax2.set_title("Distribution of Widths & Heights")
    ax2.set_xlabel("Pixels")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    plt.tight_layout()

    # --- SAVE INSTEAD OF SHOW ---
    plt.savefig(save_path, dpi=300)
    plt.close()  # Clean up memory
    print(f"Analysis saved successfully to: {save_path}")


if __name__ == "__main__":
    analyze_resolutions(IMAGE_FOLDER, OUTPUT_PLOT_PATH)
