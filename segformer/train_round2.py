
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

# ==================================================
# 1. SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent

IMG_DIR = ROOT / "trench_crops" / "images"
MASK_DIR = ROOT / "trench_crops" / "masks"

SAVE_DIR = ROOT / "checkpoints" / "round2"
ORIGINAL_SPLIT = ROOT / "checkpoints" / "split.json"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

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

if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

print("=" * 50)
print("SEGFORMER-B0 - ROUND 2")
print("=" * 50)
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA unavailable. Training on CPU.")

# ==================================================
# 2. FIND IMAGE AND MASK PAIRS
# ==================================================

EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}

def find_files(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    result = {}

    for path in folder.iterdir():
        if path.suffix.lower() in EXTENSIONS:
            name = path.stem.lower()

            if name in result:
                raise ValueError(
                    f"Duplicate filename stem: {name}"
                )

            result[name] = path

    return result


images = find_files(IMG_DIR)
masks = find_files(MASK_DIR)

if not images:
    raise RuntimeError("No trench crop images found.")

if set(images) != set(masks):
    raise RuntimeError(
        f"Image/mask mismatch!\n"
        f"Missing masks: {set(images) - set(masks)}\n"
        f"Extra masks: {set(masks) - set(images)}"
    )

names = sorted(images.keys())

print("\nTotal trench crops:", len(names))

if len(names) != 13:
    print(
        f"NOTE: Found {len(names)} crops instead of 13."
    )

# ==================================================
# 3. ORIGINAL SEM-LEVEL SPLIT
# ==================================================

if not ORIGINAL_SPLIT.exists():
    raise FileNotFoundError(
        f"Original split not found: {ORIGINAL_SPLIT}"
    )

with open(ORIGINAL_SPLIT, "r", encoding="utf-8") as f:
    original_split = json.load(f)


def get_source_sem(crop_name):
    if "_trench_" not in crop_name:
        raise ValueError(
            f"Invalid trench crop name: {crop_name}"
        )

    return crop_name.rsplit("_trench_", 1)[0]


def select_crops(source_names):
    sources = {name.lower() for name in source_names}

    return [
        name for name in names
        if get_source_sem(name) in sources
    ]


train_names = select_crops(original_split["train"])
val_names = select_crops(original_split["validation"])
test_names = select_crops(original_split["test"])

splits = {
    "train": train_names,
    "validation": val_names,
    "test": test_names
}

assigned = train_names + val_names + test_names

if len(assigned) != len(set(assigned)):
    raise RuntimeError(
        "Data leakage detected: duplicate crop assignment."
    )

if set(assigned) != set(names):
    raise RuntimeError(
        f"Some crops were not assigned: "
        f"{set(names) - set(assigned)}"
    )

if not train_names or not val_names or not test_names:
    raise RuntimeError(
        "One split has no crops. Check crop filenames "
        "and checkpoints/split.json."
    )

print("\nDATASET SPLIT")
print("-" * 50)

print("Training crops:", len(train_names))
print("Validation crops:", len(val_names))
print("Test crops:", len(test_names))

print("\nTraining:", train_names)
print("Validation:", val_names)
print("Test:", test_names)

with open(
    SAVE_DIR / "split.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(splits, f, indent=2)

# ==================================================
# 4. IMAGE PREPROCESSING
# ==================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def load_pair(name):
    image = cv2.imread(
        str(images[name]),
        cv2.IMREAD_COLOR
    )

    mask = cv2.imread(
        str(masks[name]),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        raise RuntimeError(
            f"Cannot read image or mask: {name}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    if image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(
            f"Image and mask dimensions differ: {name}"
        )

    mask = (mask > 127).astype(np.uint8)

    h, w = mask.shape

    pad_bottom = max(0, PATCH_SIZE - h)
    pad_right = max(0, PATCH_SIZE - w)

    if pad_bottom > 0 or pad_right > 0:
        image = cv2.copyMakeBorder(
            image,
            0, pad_bottom,
            0, pad_right,
            cv2.BORDER_REFLECT_101
        )

        mask = cv2.copyMakeBorder(
            mask,
            0, pad_bottom,
            0, pad_right,
            cv2.BORDER_CONSTANT,
            value=0
        )

    return image, mask


def to_tensors(image, mask):
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD

    image_tensor = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).float()

    mask_tensor = torch.from_numpy(
        mask.copy()
    ).long()

    return image_tensor, mask_tensor

# ==================================================
# 5. TRAINING DATASET
# ==================================================

class SEMTrainDataset(Dataset):

    def __init__(self, selected_names):
        self.data = [
            load_pair(name)
            for name in selected_names
        ]

    def __len__(self):
        return STEPS_PER_EPOCH * BATCH_SIZE

    def __getitem__(self, index):
        image, mask = random.choice(self.data)

        h, w = mask.shape

        ys, xs = np.where(mask == 1)

        if len(xs) > 0 and random.random() < 0.85:
            idx = random.randrange(len(xs))

            center_x = int(xs[idx])
            center_y = int(ys[idx])

            x = max(
                0,
                min(
                    center_x - PATCH_SIZE // 2,
                    w - PATCH_SIZE
                )
            )

            y = max(
                0,
                min(
                    center_y - PATCH_SIZE // 2,
                    h - PATCH_SIZE
                )
            )

        else:
            x = random.randint(0, w - PATCH_SIZE)
            y = random.randint(0, h - PATCH_SIZE)

        img_patch = image[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ].copy()

        mask_patch = mask[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ].copy()

        if random.random() < 0.5:
            img_patch = cv2.flip(img_patch, 1)
            mask_patch = cv2.flip(mask_patch, 1)

        return to_tensors(img_patch, mask_patch)

# ==================================================
# 6. VALIDATION DATASET
# ==================================================

class SEMValidationDataset(Dataset):

    def __init__(self, selected_names):
        self.patches = []

        for name in selected_names:
            image, mask = load_pair(name)

            h, w = mask.shape

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

                    valid = np.zeros(
                        (PATCH_SIZE, PATCH_SIZE),
                        dtype=np.uint8
                    )

                    valid[:ph, :pw] = 1

                    if ph < PATCH_SIZE or pw < PATCH_SIZE:
                        img_patch = cv2.copyMakeBorder(
                            img_patch,
                            0, PATCH_SIZE - ph,
                            0, PATCH_SIZE - pw,
                            cv2.BORDER_REFLECT_101
                        )

                        mask_patch = cv2.copyMakeBorder(
                            mask_patch,
                            0, PATCH_SIZE - ph,
                            0, PATCH_SIZE - pw,
                            cv2.BORDER_CONSTANT,
                            value=0
                        )

                    image_tensor, mask_tensor = to_tensors(
                        img_patch,
                        mask_patch
                    )

                    self.patches.append(
                        (
                            image_tensor,
                            mask_tensor,
                            torch.from_numpy(valid).bool()
                        )
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

print("\nTraining steps per epoch:", len(train_loader))
print("Validation patches:", len(val_loader))

# ==================================================
# 7. LOAD SEGFORMER-B0
# ==================================================

print("\nLoading SegFormer-B0...")

config = SegformerConfig.from_pretrained(
    MODEL_NAME
)

config.num_labels = 2

config.id2label = {
    0: "background",
    1: "boundary"
}

config.label2id = {
    "background": 0,
    "boundary": 1
}

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=True
)

model = model.to(DEVICE)

print("Model loaded successfully!")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)

class_weights = torch.tensor(
    [1.0, 10.0],
    dtype=torch.float32,
    device=DEVICE
)

# ==================================================
# 8. LOSS FUNCTION
# ==================================================

def compute_loss(logits, targets):

    logits = F.interpolate(
        logits.float(),
        size=targets.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    ce_loss = F.cross_entropy(
        logits,
        targets,
        weight=class_weights
    )

    probabilities = torch.softmax(
        logits, dim=1
    )[:, 1]

    target_binary = (
        targets == 1
    ).float()

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

    return ce_loss + dice_loss

# ==================================================
# 9. VALIDATION
# ==================================================

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
            dtype=torch.float16,
            enabled=USE_AMP
        ):
            logits = model(
                pixel_values=inputs
            ).logits

        logits = F.interpolate(
            logits.float(),
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predictions = logits.argmax(dim=1)

        pred = (predictions == 1) & valid
        truth = (targets == 1) & valid

        intersection += (
            pred & truth
        ).sum().item()

        pred_total += pred.sum().item()
        true_total += truth.sum().item()

    dice = (
        (2 * intersection + 1)
        / (pred_total + true_total + 1)
    )

    return dice, pred_total, true_total

# ==================================================
# 10. TRAINING
# ==================================================

best_dice = -1.0
history = []

print("\n" + "=" * 50)
print("STARTING ROUND 2 TRAINING")
print("=" * 50)

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE,
            dtype=torch.float16,
            enabled=USE_AMP
        ):
            logits = model(
                pixel_values=inputs
            ).logits

        loss = compute_loss(
            logits,
            targets
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}: {loss.item()}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
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

    with open(
        SAVE_DIR / "history.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(history, f, indent=2)

# ==================================================
# 11. FINISHED
# ==================================================

print("\n" + "=" * 50)
print("ROUND 2 TRAINING COMPLETE!")
print("=" * 50)

print("Total trench crops:", len(names))
print("Training crops:", len(train_names))
print("Validation crops:", len(val_names))
print("Test crops:", len(test_names))

print("Best Validation Dice:", round(best_dice, 4))
print("Best model saved to:", SAVE_DIR / "best_model")
print("Training history saved to:", SAVE_DIR / "history.json")
