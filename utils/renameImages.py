import os
from pathlib import Path


def rename_images():
    # --- CONFIGURATION ---
    # Replace the string below with your actual folder path
    # Example: "/home/user/Documents/cracks_project" or "C:/Photos/Project"
    TARGET_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/tape/valid/images"
    SUFFIX = "__segment_tape_joint"
    DRY_RUN = False  # Set to False to actually apply changes
    # ---------------------

    path = Path(TARGET_DIR)

    # Safety check: does the path exist?
    if not path.exists() or not path.is_dir():
        print(f"Error: The directory '{TARGET_DIR}' does not exist.")
        return

    # Find .jpg and .jpeg files (case-insensitive search)
    files = list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.JPG")) + list(path.glob("*.JPEG"))

    if not files:
        print(f"No JPG files found in: {path.absolute()}")
        return

    status = "DRY RUN (No changes made)" if DRY_RUN else "EXECUTING"
    print(f"--- {status} ---")

    for file_path in files:
        # Avoid double-suffixing if you run the script twice
        if SUFFIX in file_path.stem:
            print(f"Skipping (already renamed): {file_path.name}")
            continue

        new_name = f"{file_path.stem}{SUFFIX}{file_path.suffix}"
        new_path = file_path.with_name(new_name)

        if DRY_RUN:
            print(f"Would rename: {file_path.name}  -->  {new_name}")
        else:
            try:
                file_path.rename(new_path)
                print(f"Renamed: {file_path.name}  -->  {new_name}")
            except Exception as e:
                print(f"Failed to rename {file_path.name}: {e}")


if __name__ == "__main__":
    rename_images()
