import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation
import random
import numpy as np


def set_seed(seed=42):
    """Sets the seed for reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        # CLIPSeg outputs are usually 64x64; we upscale to the target resolution
        logits = outputs.logits.unsqueeze(1)
        upscaled_logits = F.interpolate(logits, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return upscaled_logits.squeeze(1)


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0, pos_weight_val=2.0):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma

        # Registering as a buffer ensures it moves with the model to GPU/CPU
        self.register_buffer("pos_weight", torch.tensor([pos_weight_val]))

    def dice_loss(self, logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        return 1 - dice

    def forward(self, logits, targets):
        # --- DEVICE SAFETY CHECK ---
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)

        # Ensure targets match logit dimensions (H, W)
        if logits.shape != targets.shape:
            targets = F.interpolate(targets.unsqueeze(1), size=logits.shape[-2:], mode="nearest").squeeze(1)

        # --- FOCAL LOSS (Numerically Stable) ---
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_factor = (1 - pt) ** self.gamma
        focal_loss = (self.alpha * focal_factor * bce_loss).mean()

        # --- DICE LOSS ---
        d_loss = self.dice_loss(logits, targets)

        # Weighted combination
        return (self.focal_weight * focal_loss) + (self.dice_weight * d_loss)


def get_model_and_loss(model_id="CIDAS/clipseg-rd64-refined", image_size=512, seed=42):
    """
    Factory function to initialize the model and loss criterion with a fixed seed.
    """
    set_seed(seed)

    model = CLIPSegWrapper(model_id, image_size=image_size)
    criterion = FocalDiceLoss(focal_weight=0.5, dice_weight=0.5, pos_weight_val=2.0)
    return model, criterion
