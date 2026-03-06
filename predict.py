import os
import torch
import numpy as np
import argparse
import random
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from transformers import CLIPSegProcessor

from model import get_model_and_loss


def set_seed(seed=42):
    """Sets the seed for reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="CLIPSeg Single Image Inference")
    # Required is set to True since we only want single image mode
    parser.add_argument("--img", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--prompt", type=str, default="segment crack", help="Text prompt for segmentation")
    parser.add_argument("--out_name", type=str, default="results.jpg", help="Output filename")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def sliding_window_inference(model, image_tensor, input_ids, attn_mask, window_size=352, stride=256):
    _, _, H, W = image_tensor.shape
    device = image_tensor.device
    full_probs = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    # Create weight map to favor center of patches
    y, x = torch.meshgrid(torch.linspace(-1, 1, window_size), torch.linspace(-1, 1, window_size), indexing="ij")
    weights = torch.exp(-(torch.sqrt(x**2 + y**2) ** 2) / (2 * 0.5**2)).to(device)

    y_starts = list(range(0, H - window_size + 1, stride))
    if not y_starts or y_starts[-1] + window_size < H:
        y_starts.append(max(0, H - window_size))
    x_starts = list(range(0, W - window_size + 1, stride))
    if not x_starts or x_starts[-1] + window_size < W:
        x_starts.append(max(0, W - window_size))

    for y_s in y_starts:
        for x_s in x_starts:
            patch = image_tensor[:, :, y_s : y_s + window_size, x_s : x_s + window_size]
            logits = model(input_ids, patch, attn_mask)
            prob = torch.sigmoid(logits).squeeze()

            full_probs[y_s : y_s + window_size, x_s : x_s + window_size] += prob * weights
            count_map[y_s : y_s + window_size, x_s : x_s + window_size] += weights

    return full_probs / (count_map + 1e-8)


def main():
    set_seed(42)
    args = get_args()

    # --- MODEL LOAD ---
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"🚀 Loading weights from: {args.checkpoint}")
    model, _ = get_model_and_loss(image_size=352)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device).eval()

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clahe = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)

    # --- PROCESSING ---
    print(f"🖼️  Processing image: {args.img}")
    raw_img = cv2.imread(args.img)
    if raw_img is None:
        print(f"❌ Error: Could not read image at {args.img}")
        return

    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_aug = clahe(image=raw_img)["image"]

    # Preprocess image for model
    img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(args.device)

    # Tokenize prompt
    tokens = processor(text=[args.prompt], padding="max_length", return_tensors="pt").to(args.device)

    # Run Inference
    prob_map = sliding_window_inference(model, img_tensor, tokens.input_ids, tokens.attention_mask)
    mask_np = prob_map.cpu().numpy()

    # --- VISUALIZATION & SAVING ---
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img)

    # Create colored overlay for the mask
    h, w = mask_np.shape
    colored_mask = np.zeros((h, w, 4), dtype=np.uint8)
    # Threshold at 0.5; color is Red with partial transparency (140)
    colored_mask[mask_np > 0.5] = [255, 0, 0, 140]

    plt.imshow(colored_mask)
    plt.axis("off")
    plt.title(f"Prompt: {args.prompt}", fontsize=15)

    plt.savefig(args.out_name, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"✅ Prediction saved to: {os.path.abspath(args.out_name)}")


if __name__ == "__main__":
    main()
