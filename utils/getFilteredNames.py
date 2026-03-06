import os

# --- CONFIGURATION ---
# The folder you want to scan
TARGET_DIRECTORY = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/all_imgsCleaned"

# The name of the text file to create
OUTPUT_FILE = "filenames_delete.txt"


def export_filenames(source_dir, output_path):
    # 1. Check if directory exists
    if not os.path.exists(source_dir):
        print(f"Error: The directory '{source_dir}' does not exist.")
        return

    # 2. Get list of all files (excluding folders)
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 3. Sort alphabetically (optional but helpful)
    files.sort()

    # 4. Write to the text file
    with open(output_path, "w") as f:
        for filename in files:
            f.write(filename + "\n")

    print(f"Successfully saved {len(files)} filenames to '{output_path}'.")


# Run the function
export_filenames(TARGET_DIRECTORY, OUTPUT_FILE)
