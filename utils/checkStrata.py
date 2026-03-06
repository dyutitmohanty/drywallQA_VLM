import os

# --- CONFIGURATION ---
# Hardcode your image directory path here
IMG_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/WallFeaturesDataset/train/images"


def count_classes():
    # 1. Check if directory exists
    if not os.path.exists(IMG_DIR):
        print(f"Error: The directory '{IMG_DIR}' does not exist.")
        return

    # 2. List all files (filtering for .jpg)
    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]

    # 3. Initialize counters
    crack_count = 0
    tape_joint_count = 0
    other_count = 0

    # 4. Iterate and identify
    for f in files:
        if "__segment_cracks" in f:
            crack_count += 1
        elif "__segment_tape_joint" in f:
            tape_joint_count += 1
        else:
            other_count += 1

    # 5. Display Results
    total = len(files)
    print("-" * 30)
    print(f"Dataset Summary for: {IMG_DIR}")
    print("-" * 30)
    print(f"Crack Images:       {crack_count}")
    print(f"Tape Joint Images:  {tape_joint_count}")

    if other_count > 0:
        print(f"Unrecognized Name:  {other_count}")

    print("-" * 30)
    print(f"Total .jpg files:   {total}")

    if total > 0:
        p_cracks = (crack_count / total) * 100
        p_tape = (tape_joint_count / total) * 100
        print(f"Ratio: {p_cracks:.1f}% Cracks | {p_tape:.1f}% Tape Joints")
    print("-" * 30)


if __name__ == "__main__":
    count_classes()
