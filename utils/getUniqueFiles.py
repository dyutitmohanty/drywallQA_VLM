import os
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/imagesWithoutRandomMasks"
UNIQUE_OUT_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_imgsCleaned"

# Create output folder
os.makedirs(UNIQUE_OUT_DIR, exist_ok=True)


def get_one_of_each(source_dir, output_dir):
    # Dictionary to keep track of unique prefixes
    # Key: "IMG_20220627_111320-jpg_1500x2000_jpg"
    # Value: Full filename
    unique_images = {}

    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    for filename in files:
        # Check if the filename follows the Roboflow pattern
        if ".rf." in filename:
            # Split and take the part before '.rf.'
            prefix = filename.split(".rf.")[0]

            # If we haven't saved this prefix yet, add it to our dictionary
            if prefix not in unique_images:
                unique_images[prefix] = filename
        else:
            # If the file doesn't have '.rf.', treat the whole name as unique
            if filename not in unique_images:
                unique_images[filename] = filename

    # Copy the unique files to the new folder
    print(f"Found {len(unique_images)} unique image types. Copying...")

    for prefix, filename in unique_images.items():
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        shutil.copy(src_path, dst_path)

    print(f"Done! Unique images are in: {output_dir}")


# Execute
get_one_of_each(SOURCE_DIR, UNIQUE_OUT_DIR)
