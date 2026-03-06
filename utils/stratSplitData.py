import os
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
IMG_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_imgsCleaned"
LBL_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_labelsCleaned"
OUTPUT_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/WallFeaturesDataset"

# Split sizes (must sum to 1.0)
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def split_data():
    # 1. Gather all image filenames
    all_images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]

    # 2. Extract labels for stratification
    # We define class 0 for 'cracks' and class 1 for 'tape_joint'
    labels = []
    for f in all_images:
        if "__segment_cracks" in f:
            labels.append(0)
        elif "__segment_tape_joint" in f:
            labels.append(1)
        else:
            labels.append(2)  # Unknown/Misc

    # 3. Perform Stratified Splits
    # First, split into Train and 'Temp' (Val + Test)
    train_imgs, temp_imgs, _, temp_labels = train_test_split(all_images, labels, test_size=(1 - TRAIN_SIZE), stratify=labels, random_state=42)

    # Second, split 'Temp' into Val and Test
    # Calculate relative size: (0.15 / 0.30) = 0.5
    relative_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val_imgs, test_imgs, _, _ = train_test_split(temp_imgs, temp_labels, test_size=(1 - relative_val_size), stratify=temp_labels, random_state=42)

    # 4. Define helper to move files
    def copy_files(files, subset_name):
        img_out = os.path.join(OUTPUT_DIR, subset_name, "images")
        lbl_out = os.path.join(OUTPUT_DIR, subset_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for f in files:
            # Move Image
            shutil.copy(os.path.join(IMG_DIR, f), os.path.join(img_out, f))
            # Move Label (swap .jpg for .png)
            label_name = f.replace(".jpg", ".png")
            shutil.copy(os.path.join(LBL_DIR, label_name), os.path.join(lbl_out, label_name))

    # 5. Execute move
    copy_files(train_imgs, "train")
    copy_files(val_imgs, "val")
    copy_files(test_imgs, "test")

    print(f"Done! Split {len(all_images)} images into Train({len(train_imgs)}), Val({len(val_imgs)}), Test({len(test_imgs)})")


if __name__ == "__main__":
    split_data()
