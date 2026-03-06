# CLIPSeg Fine-Tuning for Wall Feature Segmentation

This repository provides a complete pipeline for fine-tuning the CLIPSeg model to detect and segment specific wall features, such as cracks and drywall tape joints, using text-based prompts.

## Key Features

* **Custom Loss Function:** Combines Focal and Dice loss to effectively handle class imbalances and improve segmentation accuracy on fine features like narrow cracks.
* **Differential Learning Rates:** Preserves the generalized knowledge of the pre-trained CLIP backbone by using a lower learning rate (1e-6), while more aggressively training the decoder (1e-4) for this specific segmentation task.

---

## Installation & Setup

Ensure you have Python installed, then install the required core machine learning and computer vision dependencies:

```bash
pip install -r requirements.txt
```

---

## Data & Weights Setup

To run the training, evaluation, or inference scripts, you need the corresponding dataset and model weights.

### Dataset

Download the **Wall Features dataset** from the provided Google Drive link.

Extract the contents into a `datasets/` directory at the root of your project.  
The scripts expect the following path:

```
datasets/WallFeaturesDataset
```

Example project structure:

```
project_root/
│
├── datasets/
│   └── WallFeaturesDataset/
│
├── checkpoints/
│
├── train.py
├── eval.py
├── predict.py
└── predict_gallery.py
```

### Model Weights

Download the **pre-trained model weights**.

Extract them into a `checkpoints/` directory at the root of your project.

```
checkpoints/
```

---

## Usage Guide

### 1. Training (train.py)

Train the model on the dataset. The script utilizes mixed precision (AMP) and automatically saves the best checkpoint based on the **mIoU validation metric**.

#### Start training from scratch

```bash
python train.py
```

#### Resume training from an existing checkpoint

```bash
python train.py --resume checkpoints/best_model.pt
```

---

### 2. Evaluation (eval.py)

Evaluate the model's performance on the test set.  
It calculates **mIoU** and **F1-Scores** across multiple thresholds.

**Important:** You must explicitly pass the path to your weights.

Open `eval.py` and update the checkpoint path:

```python
EvalConfig.CHECKPOINT = "checkpoints/your_model.pt"
```

Run evaluation:

```bash
python eval.py
```

---

### 3. Single Image Inference (predict.py)

Run inference on a single image.  
This script outputs the original image with a blended, colored segmentation mask overlay saved as a new file.

Command-line usage:

```bash
python predict.py \
  --img path/to/image.jpg \
  --checkpoint checkpoints/best_model.pt \
  --prompt "segment crack" \
  --out_name result.jpg
```

---

### 4. Gallery Inference (predict_gallery.py)

Point this script to a directory of images, and it will process a balanced sample of **20 images (cracks and tape joints)** to generate a visually appealing **4×5 grid gallery** of the results.

Command-line usage:

```bash
python predict_gallery.py \
  --img_dir path/to/images/ \
  --checkpoint checkpoints/best_model.pt \
  --out_name sample_gallery.png
```

---

## Model Details

### Base Model

This project fine-tunes the **CIDAS/clipseg-rd64-refined** model from Hugging Face.

### Prompts Utilized

The dataset uses specific textual prompts for segmentation targets, primarily:

* `"segment crack"`
* `"segment taping area"`
* `"segment joint/tape"`
* `"segment drywall seam"`
