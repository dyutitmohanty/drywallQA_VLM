import os
import json
import numpy as np
from PIL import Image, ImageDraw

# --- Configuration ---
ANNOTATIONS_PATH = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/tape/train/_annotations.coco.json"
OUTPUT_DIR = "/home/ml_team/Documents/Dyutit/drywall_ductape/datasets/tape/train/labels"
SUFFIX = "__segment_tape_joint"  # Options: "__segment_tape" or "__segment_crack"


def create_masks():
    # Load COCO JSON
    with open(ANNOTATIONS_PATH, "r") as f:
        coco_data = json.load(f)

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create a lookup for images: {image_id: filename}
    images_lookup = {img["id"]: img["file_name"] for img in coco_data["images"]}

    # Map image_id to a list of annotations (in case one image has multiple polygons)
    image_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)

    for img_id, annotations in image_to_anns.items():
        # Get filename and prep output path
        original_filename = images_lookup[img_id]
        file_base = os.path.splitext(original_filename)[0]
        output_filename = f"{file_base}{SUFFIX}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Get image dimensions from COCO metadata
        img_info = next(item for item in coco_data["images"] if item["id"] == img_id)
        width, height = img_info["width"], img_info["height"]

        # Create a black background (L mode = 8-bit pixels, black and white)
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for ann in annotations:
            for seg in ann["segmentation"]:
                # seg is [x1, y1, x2, y2...] - convert to [(x1, y1), (x2, y2)...]
                poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                draw.polygon(poly, outline=255, fill=255)

        # Save the binary mask
        mask.save(output_path)
        print(f"Saved: {output_filename}")


if __name__ == "__main__":
    create_masks()
