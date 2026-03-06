import os
import json
from PIL import Image, ImageDraw

# --- Configuration ---
ANNOTATIONS_PATH = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/tape/valid/_annotations.coco.json"
OUTPUT_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/tape/valid/labels"
SUFFIX = "__segment_tape_joint"


def create_masks():
    with open(ANNOTATIONS_PATH, "r") as f:
        coco_data = json.load(f)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Map image IDs to filenames and dimensions
    images_info = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image_id
    image_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        image_to_anns.setdefault(img_id, []).append(ann)

    for img_id, annotations in image_to_anns.items():
        img_metadata = images_info.get(img_id)
        if not img_metadata:
            continue

        width = img_metadata["width"]
        height = img_metadata["height"]
        file_base = os.path.splitext(img_metadata["file_name"])[0]

        output_filename = f"{file_base}{SUFFIX}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Create L (8-bit grayscale) image initialized to 0 (black)
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for ann in annotations:
            # COCO bbox: [x_min, y_min, width, height]
            bbox = ann.get("bbox")
            if not bbox:
                continue

            # Ensure coordinates are integers for Pillow
            x_min = int(round(bbox[0]))
            y_min = int(round(bbox[1]))
            w = int(round(bbox[2]))
            h = int(round(bbox[3]))

            x_max = x_min + w
            y_max = y_min + h

            # Draw the rectangle: [x0, y0, x1, y1]
            draw.rectangle([x_min, y_min, x_max, y_max], fill=255, outline=255)

        mask.save(output_path)
        print(f"Generated mask for: {output_filename} (Boxes: {len(annotations)})")


if __name__ == "__main__":
    create_masks()
