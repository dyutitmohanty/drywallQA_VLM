import os
import torch
import numpy as np
import argparse
import random
import glob
import cv2
from tqdm import tqdm
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
    parser = argparse.ArgumentParser(description="Sample Inference Gallery")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_name", type=str, default="sample_inference.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def sliding_window_inference(model, image_tensor, input_ids, attn_mask, window_size=352, stride=256):
    _, _, H, W = image_tensor.shape
    device = image_tensor.device
    full_probs = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    y, x = torch.meshgrid(torch.linspace(-1, 1, window_size), torch.linspace(-1, 1, window_size), indexing="ij")
    weights = torch.exp(-(torch.sqrt(x**2 + y**2) ** 2) / (2 * 0.5**2)).to(device)

    y_starts = list(range(0, H - window_size + 1, stride))
    if not y_starts or y_starts[-1] + window_size < H:
        y_starts.append(max(0, H - window_size))
    x_starts = list(range(0, W - window_size + 1, stride))
    if not x_starts or x_starts[-1] + window_size < W:
        x_starts.append(max(0, W - window_size))

    for y in y_starts:
        for x in x_starts:
            patch = image_tensor[:, :, y : y + window_size, x : x + window_size]
            logits = model(input_ids, patch, attn_mask)
            prob = torch.sigmoid(logits).squeeze()

            full_probs[y : y + window_size, x : x + window_size] += prob * weights
            count_map[y : y + window_size, x : x + window_size] += weights

    return full_probs / (count_map + 1e-8)


def main():
    # Set seed for reproducibility (Shuffling, model init, etc.)
    set_seed(42)

    args = get_args()
    model, _ = get_model_and_loss(image_size=352)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device).eval()

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clahe = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)

    # --- BALANCED FILE DISCOVERY ---
    all_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_files.extend(glob.glob(os.path.join(args.img_dir, ext)))

    crack_files = [f for f in all_files if "crack" in f.lower() or "__segment_cracks" in f.lower()]
    tape_files = [f for f in all_files if f not in crack_files]

    # These shuffles are now deterministic due to set_seed(42)
    random.shuffle(crack_files)
    random.shuffle(tape_files)

    # Select exactly 10 of each (or as many as available if folder is small)
    selected_paths = crack_files[:10] + tape_files[:10]

    if len(selected_paths) < 20:
        print(f"⚠️ Warning: Found {len(crack_files)} cracks and {len(tape_files)} tape joints. Gallery will be smaller than 20.")

    results = []
    print(f"🚀 Processing {len(selected_paths)} balanced samples...")
    for path in tqdm(selected_paths):
        raw_img = cv2.imread(path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_aug = clahe(image=raw_img)["image"]

        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(args.device)

        is_crack = path in crack_files
        prompt = "segment crack" if is_crack else "segment joint/tape"
        tokens = processor(text=[prompt], padding="max_length", return_tensors="pt").to(args.device)

        prob_map = sliding_window_inference(model, img_tensor, tokens.input_ids, tokens.attention_mask)
        results.append({"img": raw_img, "mask": prob_map.cpu().numpy(), "is_crack": is_crack, "prompt": prompt})

    # --- GALLERY RENDERING ---
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(25, 20))
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        h, w, _ = res["img"].shape
        ax.imshow(res["img"])

        mask_vis = (res["mask"] > 0.6).astype(np.uint8)
        color = np.array([255, 0, 0]) if res["is_crack"] else np.array([0, 120, 255])

        colored_mask = np.zeros((h, w, 4), dtype=np.uint8)
        colored_mask[mask_vis == 1] = [*color, 140]  # Slightly higher opacity for visibility

        ax.imshow(colored_mask)
        ax.set_title(res["prompt"], fontsize=12, pad=10)
        ax.axis("off")

    # Clean up empty subplots if folder didn't have 20 images
    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(args.out_name, bbox_inches="tight", dpi=150)
    print(f"✅ Sample inference gallery saved as: {args.out_name}")


if __name__ == "__main__":
    main()
