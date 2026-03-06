import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPSegProcessor
import matplotlib.pyplot as plt
import matplotlib

# Force a GUI pop-up backend for Linux/Windows/macOS local displays
matplotlib.use("TkAgg")


class CLIPSegCrossPromptDataset(Dataset):
    def __init__(self, root_dir, split="train", cross_prompt_rate=0.1, processor_id="CIDAS/clipseg-rd64-refined"):
        self.root_dir = root_dir
        self.split = split
        self.cross_prompt_rate = cross_prompt_rate
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "labels")

        # Ensure directories exist
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(".jpg")])
        self.prompts_tape = ["segment taping area", "segment joint/tape", "segment drywall seam"]
        self.prompts_crack = ["segment crack", "segment wall crack"]
        self.processor = CLIPSegProcessor.from_pretrained(processor_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        is_crack_img = "__segment_cracks" in img_name
        is_tape_img = "__segment_tape_joint" in img_name

        # Handle split-based randomness
        if self.split == "train":
            gen = random
        else:
            # Deterministic for val/test based on index
            gen = random.Random(idx)

        do_cross_prompt = gen.random() < self.cross_prompt_rate

        # Logic: If it's a cross-prompt, give it the 'wrong' text
        if is_crack_img:
            prompt = gen.choice(self.prompts_tape) if do_cross_prompt else gen.choice(self.prompts_crack)
        else:
            prompt = gen.choice(self.prompts_crack) if do_cross_prompt else gen.choice(self.prompts_tape)

        image = Image.open(img_path).convert("RGB")

        # If cross-prompted, label is a black mask (negative example)
        if do_cross_prompt:
            mask = Image.new("L", (352, 352), 0)
        else:
            # Assuming labels are .png and match the .jpg image name
            mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))
            if not os.path.exists(mask_path):
                # Fallback if your masks are also .jpg
                mask_path = os.path.join(self.mask_dir, img_name)
            mask = Image.open(mask_path).convert("L")

        # Processor handles resizing to 352x352 and normalization
        inputs = self.processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Resize mask to CLIPSeg output size and binarize
        mask = mask.resize((352, 352), resample=Image.NEAREST)
        mask_tensor = (torch.tensor(list(mask.getdata())).view(352, 352).float() > 128).float()

        return {"pixel_values": inputs["pixel_values"], "label": mask_tensor, "prompt": prompt, "is_cross": do_cross_prompt, "filename": img_name}


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Adjust root_dir to your actual folder path
    DATASET_PATH = "datasets/Tape_Crack_Dataset"

    dataset = CLIPSegCrossPromptDataset(
        root_dir=DATASET_PATH,
        split="train",
        cross_prompt_rate=0.5,  # High rate just for visual testing
    )

    if len(dataset) == 0:
        print(f"No images found in {DATASET_PATH}/train/images. Check your path!")
    else:
        # Pick a random image each time the script runs
        random_idx = random.randint(0, len(dataset) - 1)
        item = dataset[random_idx]

        print(f"--- Dataset Sample ---")
        print(f"File: {item['filename']}")
        print(f"Prompt: '{item['prompt']}'")
        print(f"Cross-Prompt Active: {item['is_cross']}")

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Denormalize image for visualization
        img = item["pixel_values"].numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        ax1.imshow(img)
        ax1.set_title(f"Original Image\n(Cross: {item['is_cross']})")
        ax1.axis("off")

        ax2.imshow(item["label"], cmap="gray")
        ax2.set_title(f"Target Mask\nPrompt: {item['prompt']}")
        ax2.axis("off")

        print("Opening window...")
        plt.show()
