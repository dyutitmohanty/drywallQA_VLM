import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

# Import your custom modules
from Dataset import CLIPSegTestDataset
from model import get_model_and_loss


# --- EVALUATION CONFIG ---
class EvalConfig:
    DATA_ROOT = "datasets/WallFeaturesDataset"
    CHECKPOINT = "checkpoints/run_20260306_215931/best_model.pt"
    WINDOW_SIZE = 352
    STRIDE = 256  # Overlap of 96 pixels
    BATCH_SIZE = 8  # Patches per batch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Thresholds for final reporting
    THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]

    # Failure Triggers
    IOU_FAIL_LIMIT = 0.35
    NEG_AREA_LIMIT = 0.005  # Fail if > 0.5% of image is hallucinated

    OUT_DIR = "test_results_fullres"
    MAX_SAVES = 20


# Create output directories
for d in ["low_iou_fails", "cross_prompt_fails"]:
    os.makedirs(os.path.join(EvalConfig.OUT_DIR, d), exist_ok=True)


def get_gaussian_weights(size=352):
    """Creates a 2D Gaussian weight map to favor window centers."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    dist = torch.sqrt(x**2 + y**2)
    weights = torch.exp(-(dist**2) / (2 * 0.5**2))
    return weights.to(EvalConfig.DEVICE)


def calculate_metrics(probs, target, threshold):
    """Calculates IoU and F1 for a specific threshold."""
    preds = (probs > threshold).float()
    inter = (preds * target).sum()
    union = (preds + target).clamp(0, 1).sum()

    iou = (inter + 1e-7) / (union + 1e-7)
    precision = (inter + 1e-7) / (preds.sum() + 1e-7)
    recall = (inter + 1e-7) / (target.sum() + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return iou.item(), f1.item()


@torch.no_grad()
def sliding_window_reconstruction(model, pixel_values, input_ids, attn_mask):
    """Reconstructs full-res probability map from 352x352 windows, ensuring edge coverage."""
    _, _, H, W = pixel_values.shape
    device = EvalConfig.DEVICE

    full_probs = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)
    weights = get_gaussian_weights(EvalConfig.WINDOW_SIZE)

    patches = []
    coords = []

    # Generate lists of start coordinates, ensuring we hit the far edges
    y_starts = list(range(0, H - EvalConfig.WINDOW_SIZE + 1, EvalConfig.STRIDE))
    if len(y_starts) == 0 or y_starts[-1] + EvalConfig.WINDOW_SIZE < H:
        y_starts.append(max(0, H - EvalConfig.WINDOW_SIZE))

    x_starts = list(range(0, W - EvalConfig.WINDOW_SIZE + 1, EvalConfig.STRIDE))
    if len(x_starts) == 0 or x_starts[-1] + EvalConfig.WINDOW_SIZE < W:
        x_starts.append(max(0, W - EvalConfig.WINDOW_SIZE))

    # Iterate over the guaranteed coordinates
    for y in y_starts:
        for x in x_starts:
            patches.append(pixel_values[:, :, y : y + EvalConfig.WINDOW_SIZE, x : x + EvalConfig.WINDOW_SIZE])
            coords.append((y, x))

    # Batch processing for speed
    for i in range(0, len(patches), EvalConfig.BATCH_SIZE):
        batch = torch.cat(patches[i : i + EvalConfig.BATCH_SIZE], dim=0)
        b_size = batch.shape[0]
        b_ids = input_ids.repeat(b_size, 1)
        b_mask = attn_mask.repeat(b_size, 1)

        logits = model(b_ids, batch, b_mask)
        batch_probs = torch.sigmoid(logits)

        for j in range(b_size):
            y, x = coords[i + j]
            full_probs[y : y + EvalConfig.WINDOW_SIZE, x : x + EvalConfig.WINDOW_SIZE] += batch_probs[j] * weights
            count_map[y : y + EvalConfig.WINDOW_SIZE, x : x + EvalConfig.WINDOW_SIZE] += weights

    return full_probs / (count_map + 1e-8)


def save_visual(image, gt, pred, prompt, path, info):
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    # Basic normalization for visualization
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img_np)
    ax[0].set_title(f"Original Image\nPrompt: {prompt}", fontsize=10)
    ax[1].imshow(gt.cpu().numpy(), cmap="gray")
    ax[1].set_title("Target Mask")
    ax[2].imshow(pred.cpu().numpy(), cmap="gray")
    ax[2].set_title(f"Model Prediction\n{info}")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    # --- REPRODUCIBILITY ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model Init
    model, _ = get_model_and_loss(image_size=EvalConfig.WINDOW_SIZE)
    ckpt = torch.load(EvalConfig.CHECKPOINT, map_location=EvalConfig.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(EvalConfig.DEVICE).eval()

    dataset = CLIPSegTestDataset(EvalConfig.DATA_ROOT)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("🔥 Warming up GPU...")
    w_img = torch.randn(1, 3, 352, 352).to(EvalConfig.DEVICE)
    w_id = torch.randint(0, 100, (1, 77)).to(EvalConfig.DEVICE)
    w_mask = torch.ones((1, 77), dtype=torch.long).to(EvalConfig.DEVICE)  # Fixed warmup mask
    for _ in range(10):
        _ = model(w_id, w_img, w_mask)
    torch.cuda.synchronize()

    metrics_storage = {cat: {t: {"iou": [], "f1": []} for t in EvalConfig.THRESHOLDS} for cat in ["crack", "tape"]}
    neg_rejection_scores = {"crack": [], "tape": []}
    latencies = []

    saves_low = 0
    saves_neg = 0

    print(f"🚀 Scanning {len(dataset)} High-Res images...")

    for idx, batch in enumerate(tqdm(loader)):
        pixel_values = batch["pixel_values"].to(EvalConfig.DEVICE)
        # Fixed shape safety by squeezing down to 2D
        gt_mask = batch["label"].to(EvalConfig.DEVICE)[0].squeeze()
        img_name = batch["image_name"][0]

        # Determine category and prompt lists
        is_crack = "__segment_cracks" in img_name
        cat = "crack" if is_crack else "tape"
        pos_prompts = dataset.prompts_crack if is_crack else dataset.prompts_tape
        neg_prompts = dataset.prompts_tape if is_crack else dataset.prompts_crack

        best_img_probs = None
        best_iou_at_05 = -1
        best_p_str = ""

        # --- A. POSITIVE ENSEMBLE EVALUATION ---
        start_t = time.perf_counter()
        for p in pos_prompts:
            tokens = dataset.processor.tokenizer(text=[p], padding="max_length", truncation=True, return_tensors="pt").to(EvalConfig.DEVICE)

            # Generate full probability map via sliding window
            full_res_probs = sliding_window_reconstruction(model, pixel_values, tokens.input_ids, tokens.attention_mask)

            # Select the "winner" prompt based on standard T=0.5
            cur_iou, _ = calculate_metrics(full_res_probs, gt_mask, 0.5)
            if cur_iou > best_iou_at_05:
                best_iou_at_05 = cur_iou
                best_img_probs = full_res_probs
                best_p_str = p

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start_t)

        # Log metrics for all specified thresholds using the best performing prompt
        for t in EvalConfig.THRESHOLDS:
            iou, f1 = calculate_metrics(best_img_probs, gt_mask, t)
            metrics_storage[cat][t]["iou"].append(iou)
            metrics_storage[cat][t]["f1"].append(f1)

        # --- B. NEGATIVE (CROSS-PROMPT) EVALUATION ---
        neg_p = random.choice(neg_prompts)
        neg_tokens = dataset.processor.tokenizer(text=[neg_p], padding="max_length", return_tensors="pt").to(EvalConfig.DEVICE)
        neg_probs = sliding_window_reconstruction(model, pixel_values, neg_tokens.input_ids, neg_tokens.attention_mask)

        # For cross prompts, anything predicted is a False Positive
        neg_pred_bin = (neg_probs > 0.5).float()
        fp_area = neg_pred_bin.sum() / (gt_mask.shape[0] * gt_mask.shape[1])
        is_neg_ok = fp_area < EvalConfig.NEG_AREA_LIMIT
        neg_rejection_scores[cat].append(1.0 if is_neg_ok else 0.0)

        # Visualization: Low IoU (Uses Real GT)
        if best_iou_at_05 < EvalConfig.IOU_FAIL_LIMIT and saves_low < EvalConfig.MAX_SAVES:
            path = os.path.join(EvalConfig.OUT_DIR, "low_iou_fails", f"{cat}_{idx}.png")
            save_visual(pixel_values[0], gt_mask, (best_img_probs > 0.5).float(), best_p_str, path, f"IoU@0.5: {best_iou_at_05:.3f}")
            saves_low += 1

        # Visualization: Hallucinations (Cross-Prompt Fails)
        if not is_neg_ok and saves_neg < EvalConfig.MAX_SAVES:
            path = os.path.join(EvalConfig.OUT_DIR, "cross_prompt_fails", f"hallucination_{cat}_{idx}.png")
            blank_gt = torch.zeros_like(gt_mask)
            save_visual(pixel_values[0], blank_gt, neg_pred_bin, neg_p, path, f"FP Area: {fp_area:.2%}")
            saves_neg += 1

    # --- FINAL SUMMARY REPORT ---
    print("\n" + "═" * 65)
    print(f"{'CATEGORY':<10} | {'T':<4} | {'mIoU':<8} | {'F1-Score':<8} | {'Neg.Rej':<8}")
    print("─" * 65)
    for c in ["crack", "tape"]:
        for t in EvalConfig.THRESHOLDS:
            miou = np.mean(metrics_storage[c][t]["iou"])
            mf1 = np.mean(metrics_storage[c][t]["f1"])
            nrej = np.mean(neg_rejection_scores[c]) * 100
            print(f"{c.upper():<10} | {t:<4} | {miou:.4f} | {mf1:.4f}   | {nrej:.1f}%")
        print("─" * 65)

    avg_lat = np.mean(latencies)
    print(f"⏱️ Avg Time per Wall: {avg_lat:.2f}s | Throughput: {60 / avg_lat:.1f} walls/min")
    print("═" * 65)


if __name__ == "__main__":
    main()
