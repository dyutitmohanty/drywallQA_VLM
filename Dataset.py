import os
import torch
import random
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import albumentations as A


# --- 1. MODEL WRAPPER ---
class CLIPSegWrapper(nn.Module):
    def __init__(self, model_id="CIDAS/clipseg-rd64-refined", image_size=512):
        super().__init__()
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id)
        self.image_size = image_size

    def forward(self, input_ids, pixel_values, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            interpolate_pos_encoding=True,
        )
        logits = outputs.logits.unsqueeze(1)
        upscaled_logits = F.interpolate(logits, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return upscaled_logits.squeeze(1)


# --- 2. LOSS FUNCTION ---
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            targets = F.interpolate(targets.unsqueeze(1), size=logits.shape[-2:], mode="nearest").squeeze(1)

        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky_index) ** (1 / self.gamma)


# --- 3. DATASET WITH PRE-PROC CLAHE ---
class CLIPSegCrossPromptDataset(Dataset):
    def __init__(self, root_dir, split="train", cross_prompt_rate=0.2, processor_id="CIDAS/clipseg-rd64-refined", image_size=352):
        self.root_dir = root_dir
        self.split = split
        self.cross_prompt_rate = cross_prompt_rate
        self.image_size = image_size

        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "labels")
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith((".jpg", ".png"))])

        self.prompts_tape = ["segment taping area", "segment joint/tape", "segment drywall seam", "a drywall tape joint with printed text"]
        self.prompts_crack = ["segment crack", "segment wall crack"]

        self.processor = CLIPSegProcessor.from_pretrained(processor_id)

        # FIX: always_apply -> p=1.0
        self.clahe_preprocessor = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)

        def get_core_augs():
            return [
                A.OneOf(
                    [
                        A.CropNonEmptyMaskIfExists(width=self.image_size, height=self.image_size, p=0.8),
                        A.RandomCrop(width=self.image_size, height=self.image_size, p=0.2),
                    ],
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]

        common_visual_augs = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]

        if self.split == "train":
            self.transform_standard = A.Compose([*get_core_augs(), *common_visual_augs])

            # FIX: Updated CoarseDropout arguments for Albumentations v1.4+
            self.transform_tape = A.Compose(
                [
                    A.Resize(height=512, width=512, interpolation=1),
                    *get_core_augs(),
                    *common_visual_augs,
                    A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(10, 40), hole_width_range=(10, 40), fill_value=0, p=0.4),
                ]
            )
        else:
            self.transform_val_standard = A.Compose([A.CenterCrop(width=self.image_size, height=self.image_size)])
            self.transform_val_tape = A.Compose([A.Resize(height=512, width=512, interpolation=1), A.CenterCrop(width=self.image_size, height=self.image_size)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)

        # Apply CLAHE Preprocessing
        image_np = self.clahe_preprocessor(image=image_np)["image"]

        if os.path.exists(mask_path):
            mask_np = np.array(Image.open(mask_path).convert("L"))
        else:
            mask_np = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

        is_crack_img = "__segment_cracks" in img_name
        is_tape_img = "__segment_tape_joint" in img_name
        gen = random if self.split == "train" else random.Random(idx)
        do_cross_prompt = gen.random() < self.cross_prompt_rate

        if is_crack_img:
            prompt = gen.choice(self.prompts_tape) if do_cross_prompt else gen.choice(self.prompts_crack)
        elif is_tape_img:
            prompt = gen.choice(self.prompts_crack) if do_cross_prompt else gen.choice(self.prompts_tape)
        else:
            prompt = "segment background"

        transform = (self.transform_tape if is_tape_img else self.transform_standard) if self.split == "train" else (self.transform_val_tape if is_tape_img else self.transform_val_standard)

        augmented = transform(image=image_np, mask=mask_np)
        crop_img_np, crop_mask_np = augmented["image"], augmented["mask"]

        if do_cross_prompt:
            crop_mask_np = np.zeros_like(crop_mask_np)

        image_inputs = self.processor.image_processor(images=[crop_img_np], return_tensors="pt", do_resize=False)
        text_inputs = self.processor.tokenizer(text=[prompt], padding="max_length", truncation=True, return_tensors="pt")
        mask_tensor = (torch.from_numpy(crop_mask_np).float() / 255.0 > 0.5).float()

        return {
            "pixel_values": image_inputs.pixel_values.squeeze(0),
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "label": mask_tensor,
            "is_cross_prompt": do_cross_prompt,  # <--- FIXED: Added this key back
            "prompt": prompt,
        }


# --- 4. FACTORY FUNCTION ---
def get_model_and_loss(model_id="CIDAS/clipseg-rd64-refined", image_size=512):
    model = CLIPSegWrapper(model_id, image_size=image_size)
    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)
    return model, criterion


class CLIPSegTestDataset(Dataset):
    def __init__(self, root_dir, processor_id="CIDAS/clipseg-rd64-refined"):
        self.root_dir = root_dir

        # Explicitly point to the test split
        self.img_dir = os.path.join(root_dir, "test", "images")
        self.mask_dir = os.path.join(root_dir, "test", "labels")

        # Get all image files
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith((".jpg", ".png"))])

        # Available prompts for evaluation logic in the script
        self.prompts_tape = [
            "segment taping area",
            "segment joint/tape",
            "segment drywall seam",
            "a drywall tape joint with printed text",
        ]

        self.prompts_crack = [
            "segment crack",
            "segment wall crack",
        ]

        self.processor = CLIPSegProcessor.from_pretrained(processor_id)

        # CLAHE preprocessing to match training pipeline
        self.clahe_preprocessor = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 1. Load Image
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)

        # 2. Apply CLAHE (Critical for consistency with your training)
        image_np = self.clahe_preprocessor(image=image_np)["image"]

        # 3. Load Mask (Always load the REAL mask)
        if os.path.exists(mask_path):
            mask_np = np.array(Image.open(mask_path).convert("L"))
        else:
            # Fallback for missing masks (shouldn't happen in test set)
            mask_np = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

        # 4. CLIPSeg Image Preprocessing (No Resizing)
        image_inputs = self.processor.image_processor(images=[image_np], return_tensors="pt", do_resize=False)

        # 5. Convert Mask to Tensor
        mask_tensor = (torch.from_numpy(mask_np).float() / 255.0 > 0.5).float()

        # NOTE: We no longer pick a prompt here.
        # The test loop will iterate through the prompt lists based on image_name.
        return {
            "pixel_values": image_inputs.pixel_values.squeeze(0),
            "label": mask_tensor,
            "image_name": img_name,
            "original_size": mask_tensor.shape,
        }
