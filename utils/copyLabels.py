import os
import shutil

# --- CONFIGURATION ---
TEXT_FILE_PATH = "/home/ml_team/Documents/Dyutit/drywall_ductape/filenames_keep.txt"
SOURCE_IMG_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_labels"
OUTPUT_IMG_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_labelsCleaned"

# Ensure output directory exists
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def copy_processed_files(txt_path, src_dir, out_dir):
    # 1. Read the filenames from the text file
    if not os.path.exists(txt_path):
        print(f"Error: Text file not found at {txt_path}")
        return

    with open(txt_path, "r") as f:
        # .strip() removes newlines and extra spaces
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} entries in text file. Processing...")

    copied_count = 0
    missing_count = 0

    for original_name in lines:
        # 2. Transform the filename: remove .jpg and add .png
        # This replaces the literal string ".jpg" with ".png"
        target_filename = original_name.replace(".jpg", ".png")

        src_path = os.path.join(src_dir, target_filename)
        dst_path = os.path.join(out_dir, target_filename)

        # 3. Check if the .png actually exists in the source dir
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copied_count += 1
        else:
            print(f"Warning: File not found: {target_filename}")
            missing_count += 1

    print("--- Results ---")
    print(f"Successfully copied: {copied_count}")
    print(f"Files missing in source: {missing_count}")
    print(f"Done! Check your output at: {out_dir}")


if __name__ == "__main__":
    copy_processed_files(TEXT_FILE_PATH, SOURCE_IMG_DIR, OUTPUT_IMG_DIR)
