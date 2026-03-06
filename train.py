import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
import argparse
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import your custom classes
from Dataset import CLIPSegCrossPromptDataset
from model import get_model_and_loss

warnings.filterwarnings("ignore", message="The following named arguments are not valid for `ViTImageProcessor.preprocess` and were ignored: 'padding'")


# --- CONFIGURATION ---
class Config:
    SEED = 42  # Fixed seed for reproducibility
    DATA_ROOT = "datasets/WallFeaturesDataset"
    IMAGE_SIZE = 352
    BATCH_SIZE = 16
    LR_BACKBONE = 1e-6
    LR_DECODER = 1e-4
    EPOCHS = 20
    CROSS_PROMPT_RATE = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN_ID = time.strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.path.join("checkpoints", f"run_{RUN_ID}")
    VIS_DIR = os.path.join("results", f"run_{RUN_ID}")
    LOG_FILE = os.path.join(SAVE_DIR, "train_log.txt")
    SIGMOID_THRESHOLD = 0.5
    EVAL_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]


os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.VIS_DIR, exist_ok=True)


def set_seed(seed):
    """Sets the seed for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior for cuDNN convolution algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_message(message):
    print(message)
    with open(Config.LOG_FILE, "a") as f:
        f.write(message + "\n")


def save_visual_results(image_tensor, label_tensor, pred_tensor, prompt, epoch, idx, is_cross, threshold=0.5):
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    label = label_tensor.cpu().numpy()
    pred = torch.sigmoid(pred_tensor).detach().cpu().numpy()
    pred_binary = (pred > threshold).astype(np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    status = "[CROSS]" if is_cross else "[MATCHED]"
    ax[0].imshow(img)
    ax[0].set_title(f"{status}\n{prompt}", fontsize=9)
    ax[1].imshow(label, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_binary, cmap="gray")
    ax[2].set_title(f"Pred (T={threshold})")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{Config.VIS_DIR}/epoch_{epoch}_sample_{idx}.png")
    plt.close()


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
            outputs = model(input_ids, pixel_values, attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader), time.time() - start_time


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, vis_count, max_vis = 0, 0, 5
    threshold_stats = {t: {"seg_iou_sum": 0.0, "seg_count": 0, "neg_correct": 0, "neg_total": 0} for t in Config.EVAL_THRESHOLDS}

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        is_cross = batch["is_cross_prompt"]
        prompts = batch["prompt"]

        outputs = model(input_ids, pixel_values, attention_mask)
        total_loss += criterion(outputs, labels).item()
        probs = torch.sigmoid(outputs)

        for t in Config.EVAL_THRESHOLDS:
            preds_bin = (probs > t).float()
            for i in range(len(is_cross)):
                if not is_cross[i]:
                    intersection = (preds_bin[i] * labels[i]).sum()
                    union = (preds_bin[i] + labels[i]).clamp(0, 1).sum()
                    threshold_stats[t]["seg_iou_sum"] += (intersection + 1e-6) / (union + 1e-6)
                    threshold_stats[t]["seg_count"] += 1
                else:
                    threshold_stats[t]["neg_total"] += 1
                    if preds_bin[i].sum() == 0:
                        threshold_stats[t]["neg_correct"] += 1

        for i in range(len(is_cross)):
            if vis_count < max_vis:
                save_visual_results(pixel_values[i], labels[i], outputs[i], prompts[i], epoch, vis_count, is_cross[i])
                vis_count += 1

    final_results = {}
    for t in Config.EVAL_THRESHOLDS:
        miou = (threshold_stats[t]["seg_iou_sum"] / threshold_stats[t]["seg_count"]).item() if threshold_stats[t]["seg_count"] > 0 else 0.0
        neg_acc = (threshold_stats[t]["neg_correct"] / threshold_stats[t]["neg_total"]) if threshold_stats[t]["neg_total"] > 0 else 1.0
        final_results[t] = {"miou": miou, "neg_acc": neg_acc}

    return total_loss / len(loader), final_results


def main():
    # Set the seed before anything else happens
    set_seed(Config.SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to checkpoint (.pt)")
    args = parser.parse_args()

    log_message(f"🚀 Starting run: {Config.RUN_ID} | Res: {Config.IMAGE_SIZE} | Seed: {Config.SEED}")
    total_start = time.time()

    model, criterion = get_model_and_loss(image_size=Config.IMAGE_SIZE)
    model.to(Config.DEVICE)
    criterion.to(Config.DEVICE)

    # --- DIFFERENTIAL LEARNING RATE SETUP ---
    backbone_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if "clip_model.text_model" in name:
            param.requires_grad = False
        elif "clip_model.visual" in name:
            param.requires_grad = True
            backbone_params.append(param)
        else:
            param.requires_grad = True
            decoder_params.append(param)

    optimizer = AdamW([{"params": backbone_params, "lr": Config.LR_BACKBONE}, {"params": decoder_params, "lr": Config.LR_DECODER}], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = torch.amp.GradScaler()

    start_epoch, best_iou = 0, 0.0

    if args.resume and os.path.exists(args.resume):
        log_message(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_iou = checkpoint.get("best_iou", 0.0)

    train_ds = CLIPSegCrossPromptDataset(Config.DATA_ROOT, split="train", cross_prompt_rate=Config.CROSS_PROMPT_RATE, image_size=Config.IMAGE_SIZE)
    val_ds = CLIPSegCrossPromptDataset(Config.DATA_ROOT, split="val", cross_prompt_rate=Config.CROSS_PROMPT_RATE, image_size=Config.IMAGE_SIZE)

    # Note: num_workers > 0 can introduce slight non-determinism in data loading order
    # but is generally acceptable for fine-tuning.
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    try:
        for epoch in range(start_epoch, Config.EPOCHS):
            curr_epoch = epoch + 1
            log_message(f"\n--- Epoch {curr_epoch}/{Config.EPOCHS} ---")

            train_loss, train_time = train_one_epoch(model, train_loader, optimizer, criterion, scaler, Config.DEVICE)
            val_loss, thresh_metrics = validate(model, val_loader, criterion, Config.DEVICE, curr_epoch)

            lrs = [group["lr"] for group in optimizer.param_groups]
            log_message(f"⏱️ Time: {train_time:.2f}s | LRs: [B: {lrs[0]:.1e} D: {lrs[1]:.1e}] | Loss: [T: {train_loss:.4f} V: {val_loss:.4f}]")

            metric_msg = "📈 Metrics:"
            for t in Config.EVAL_THRESHOLDS:
                m = thresh_metrics[t]
                metric_msg += f" | T@{t}: [mIoU: {m['miou']:.4f} Neg: {m['neg_acc'] * 100:.1f}%]"
            log_message(metric_msg)

            scheduler.step()

            current_iou_05 = thresh_metrics[0.5]["miou"]
            if current_iou_05 > best_iou:
                best_iou = current_iou_05
                checkpoint_data = {"epoch": curr_epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_iou": best_iou}
                torch.save(checkpoint_data, os.path.join(Config.SAVE_DIR, "best_model.pt"))
                log_message(f"⭐ New Best mIoU (0.5)! Checkpoint saved.")

    except KeyboardInterrupt:
        log_message("\n🛑 Training interrupted.")

    log_message(f"\n🏁 FINISHED | Total Time: {(time.time() - total_start) / 60:.2f} min | Best mIoU (0.5): {best_iou:.4f}")


if __name__ == "__main__":
    main()
