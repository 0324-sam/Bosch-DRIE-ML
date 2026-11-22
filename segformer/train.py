
from pathlib import Path
import json
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation
)

# ============================
# 1. Settings
# ============================

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images_original"
MASK_DIR = ROOT / "masks"
SAVE_DIR = ROOT / "checkpoints"

SAVE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

PATCH_SIZE = 256
BATCH_SIZE = 1
EPOCHS = 12
STEPS_PER_EPOCH = 40
LEARNING_RATE = 5e-5
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)

# ============================
# 2. Load image pairs
# ============================

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def find_files(folder):
    result = {}
    for p in folder.iterdir():
        if p.suffix.lower() in EXTENSIONS:
            key = p.stem.lower()
            if key in result:
                raise ValueError(f"Duplicate file stem: {key}")
            result[key] = p
    return result

images = find_files(IMG_DIR)
masks = find_files(MASK_DIR)

if set(images) != set(masks):
    raise RuntimeError(
        f"Image/mask mismatch!\n"
        f"Missing masks: {set(images) - set(masks)}\n"
        f"Extra masks: {set(masks) - set(images)}"
    )

names = sorted(images.keys())

if len(names) < 3:
    raise RuntimeError("Need at least 3 labeled images.")

random.shuffle(names)

train_names = names[:-2]
val_names = [names[-2]]
test_names = [names[-1]]

splits = {
    "train": train_names,
    "validation": val_names,
    "test": test_names
}

with open(SAVE_DIR / "split.json", "w") as f:
    json.dump(splits, f, indent=2)

print("Training:", train_names)
print("Validation:", val_names)
print("Test:", test_names)

# ============================
# 3. Dataset
# ============================

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_pair(name):
    image = cv2.imread(str(images[name]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(masks[name]), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise RuntimeError(f"Cannot read image/mask: {name}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(f"Size mismatch: {name}")

    # Binary labels: 0 = background, 1 = boundary
    mask = (mask > 127).astype(np.uint8)

    # Padding only; no resizing of the original SEM.
    h, w = mask.shape
    bottom = max(0, PATCH_SIZE - h)
    right = max(0, PATCH_SIZE - w)

    image = cv2.copyMakeBorder(
        image, 0, bottom, 0, right,
        cv2.BORDER_REFLECT_101
    )

    mask = cv2.copyMakeBorder(
        mask, 0, bottom, 0, right,
        cv2.BORDER_CONSTANT, value=0
    )

    return image, mask

def to_tensors(image, mask):
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD

    image = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).float()

    mask = torch.from_numpy(
        mask.copy()
    ).long()

    return image, mask

class SEMTrainDataset(Dataset):
    def __init__(self, selected_names):
        self.data = [load_pair(n) for n in selected_names]

    def __len__(self):
        return STEPS_PER_EPOCH * BATCH_SIZE

    def __getitem__(self, index):
        image, mask = random.choice(self.data)
        h, w = mask.shape

        # Prefer crops containing a labeled boundary.
        ys, xs = np.where(mask == 1)

        if len(xs) > 0 and random.random() < 0.85:
            idx = random.randrange(len(xs))
            center_x = int(xs[idx])
            center_y = int(ys[idx])

            x = max(0, min(center_x - PATCH_SIZE // 2,
                           w - PATCH_SIZE))
            y = max(0, min(center_y - PATCH_SIZE // 2,
                           h - PATCH_SIZE))
        else:
            x = random.randint(0, w - PATCH_SIZE)
            y = random.randint(0, h - PATCH_SIZE)

        img_patch = image[
            y:y + PATCH_SIZE, x:x + PATCH_SIZE
        ].copy()

        mask_patch = mask[
            y:y + PATCH_SIZE, x:x + PATCH_SIZE
        ].copy()

        # Horizontal flip only.
        if random.random() < 0.5:
            img_patch = cv2.flip(img_patch, 1)
            mask_patch = cv2.flip(mask_patch, 1)

        return to_tensors(img_patch, mask_patch)

class SEMValidationDataset(Dataset):
    def __init__(self, selected_names):
        self.patches = []

        for name in selected_names:
            image, mask = load_pair(name)
            h, w = mask.shape

            # Fixed non-overlapping validation tiles.
            for y in range(0, h, PATCH_SIZE):
                for x in range(0, w, PATCH_SIZE):
                    img_patch = image[
                        y:y + PATCH_SIZE,
                        x:x + PATCH_SIZE
                    ]
                    mask_patch = mask[
                        y:y + PATCH_SIZE,
                        x:x + PATCH_SIZE
                    ]

                    ph, pw = mask_patch.shape

                    if ph < PATCH_SIZE or pw < PATCH_SIZE:
                        img_patch = cv2.copyMakeBorder(
                            img_patch, 0, PATCH_SIZE - ph,
                            0, PATCH_SIZE - pw,
                            cv2.BORDER_REFLECT_101
                        )
                        mask_patch = cv2.copyMakeBorder(
                            mask_patch, 0, PATCH_SIZE - ph,
                            0, PATCH_SIZE - pw,
                            cv2.BORDER_CONSTANT, value=0
                        )

                    # Exclude padded pixels from metrics.
                    valid = np.zeros(
                        (PATCH_SIZE, PATCH_SIZE),
                        dtype=np.uint8
                    )
                    valid[:ph, :pw] = 1

                    self.patches.append(
                        (*to_tensors(img_patch, mask_patch),
                         torch.from_numpy(valid).bool())
                    )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return self.patches[index]

train_loader = DataLoader(
    SEMTrainDataset(train_names),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    SEMValidationDataset(val_names),
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# ============================
# 4. Build model
# ============================

config = SegformerConfig.from_pretrained(MODEL_NAME)
config.num_labels = 2
config.id2label = {0: "background", 1: "boundary"}
config.label2id = {"background": 0, "boundary": 1}

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=True
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

scaler = torch.amp.GradScaler(
    "cuda", enabled=USE_AMP
)

class_weights = torch.tensor(
    [1.0, 10.0],
    dtype=torch.float32,
    device=DEVICE
)

# ============================
# 5. Loss function
# ============================

def compute_loss(logits, targets):
    logits = F.interpolate(
        logits.float(),
        size=targets.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights
    )

    probabilities = torch.softmax(logits, dim=1)[:, 1]
    target_binary = (targets == 1).float()

    intersection = (
        probabilities * target_binary
    ).sum(dim=(1, 2))

    denominator = (
        probabilities.sum(dim=(1, 2))
        + target_binary.sum(dim=(1, 2))
    )

    dice_loss = 1 - (
        (2 * intersection + 1)
        / (denominator + 1)
    ).mean()

    return ce + dice_loss

# ============================
# 6. Validation
# ============================

@torch.no_grad()
def validate():
    model.eval()

    intersection = 0
    pred_total = 0
    true_total = 0

    for inputs, targets, valid in val_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        valid = valid.to(DEVICE)

        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16 if USE_AMP else torch.bfloat16,
            enabled=USE_AMP
        ):
            logits = model(pixel_values=inputs).logits

        logits = F.interpolate(
            logits.float(),
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predictions = logits.argmax(dim=1)

        pred = (predictions == 1) & valid
        truth = (targets == 1) & valid

        intersection += (pred & truth).sum().item()
        pred_total += pred.sum().item()
        true_total += truth.sum().item()

    dice = (2 * intersection + 1) / (
        pred_total + true_total + 1
    )

    return dice, pred_total, true_total

# ============================
# 7. Training
# ============================

best_dice = -1.0
history = []

print("Starting training...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16 if USE_AMP else torch.bfloat16,
            enabled=USE_AMP
        ):
            logits = model(pixel_values=inputs).logits

        loss = compute_loss(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_dice, pred_pixels, true_pixels = validate()

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"Pred pixels: {pred_pixels} | "
        f"GT pixels: {true_pixels}",
        flush=True
    )

    history.append({
        "epoch": epoch,
        "train_loss": avg_loss,
        "val_dice": val_dice
    })

    if val_dice > best_dice:
        best_dice = val_dice

        model.save_pretrained(
            str(SAVE_DIR / "best_model")
        )

        print("Saved new best model!")

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_dice": best_dice,
            "history": history
        },
        SAVE_DIR / "latest_training.pt"
    )

    with open(SAVE_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

print("Training complete!")
print("Best validation Dice:", round(best_dice, 4))
print("Model saved to:", SAVE_DIR / "best_model")
