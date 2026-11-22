
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# ==================================
# 1. Settings
# ==================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "checkpoints" / "best_model"
TEST_DIR = ROOT / "test_images"
OUTPUT_DIR = ROOT / "outputs" / "predictions_round2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 256
OVERLAP = 64
THRESHOLD = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}

# ==================================
# 2. Load trained model
# ==================================

print("Loading trained SegFormer-B0...")

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_DIR}"
    )

model = SegformerForSemanticSegmentation.from_pretrained(
    str(MODEL_DIR)
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Device:", DEVICE)

# ==================================
# 3. Image preprocessing
# ==================================

def preprocess(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD

    tensor = torch.from_numpy(
        rgb.transpose(2, 0, 1).copy()
    ).float()

    return tensor.unsqueeze(0)


def tile_positions(length):
    stride = PATCH_SIZE - OVERLAP

    if length <= PATCH_SIZE:
        return [0]

    positions = list(
        range(0, length - PATCH_SIZE + 1, stride)
    )

    last = length - PATCH_SIZE

    if positions[-1] != last:
        positions.append(last)

    return positions

# ==================================
# 4. Predict full SEM image
# ==================================

@torch.inference_mode()
def predict_image(image):

    height, width = image.shape[:2]

    # Pad only if smaller than patch size.
    pad_bottom = max(0, PATCH_SIZE - height)
    pad_right = max(0, PATCH_SIZE - width)

    padded = cv2.copyMakeBorder(
        image,
        0, pad_bottom,
        0, pad_right,
        cv2.BORDER_REFLECT_101
    )

    h, w = padded.shape[:2]

    probability_sum = np.zeros(
        (h, w), dtype=np.float32
    )

    count = np.zeros(
        (h, w), dtype=np.float32
    )

    ys = tile_positions(h)
    xs = tile_positions(w)

    total_tiles = len(ys) * len(xs)

    print(f"Processing {total_tiles} patches...")

    for y in ys:
        for x in xs:

            patch = padded[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            inputs = preprocess(patch).to(DEVICE)

            outputs = model(
                pixel_values=inputs
            )

            logits = F.interpolate(
                outputs.logits.float(),
                size=(PATCH_SIZE, PATCH_SIZE),
                mode="bilinear",
                align_corners=False
            )

            probabilities = torch.softmax(
                logits, dim=1
            )[0, 1]

            probabilities = (
                probabilities.cpu().numpy()
            )

            probability_sum[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += probabilities

            count[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += 1

    full_probability = probability_sum / np.maximum(
        count, 1
    )

    # Restore original SEM dimensions.
    full_probability = full_probability[
        :height, :width
    ]

    binary_mask = (
        full_probability >= THRESHOLD
    ).astype(np.uint8)

    return binary_mask, full_probability

# ==================================
# 5. Save results
# ==================================

def save_results(image, mask, probability, name):

    # 1. Binary mask (white boundary)
    mask_path = OUTPUT_DIR / f"{name}_mask.png"

    cv2.imwrite(
        str(mask_path),
        mask * 255
    )

    # 2. Overlay on original SEM
    overlay = image.copy()

    red = np.zeros_like(image)
    red[:, :] = (0, 0, 255)

    selected = mask.astype(bool)

    overlay[selected] = (
        0.3 * image[selected]
        + 0.7 * red[selected]
    ).astype(np.uint8)

    overlay_path = (
        OUTPUT_DIR / f"{name}_overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    # 3. Probability map
    prob_image = np.uint8(
        np.clip(probability * 255, 0, 255)
    )

    color_map = cv2.applyColorMap(
        prob_image,
        cv2.COLORMAP_JET
    )

    probability_path = (
        OUTPUT_DIR / f"{name}_probability.png"
    )

    cv2.imwrite(
        str(probability_path),
        color_map
    )

    # 4. Side-by-side comparison
    comparison = np.concatenate(
        [
            image,
            cv2.cvtColor(
                mask * 255,
                cv2.COLOR_GRAY2BGR
            ),
            overlay
        ],
        axis=1
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{name}_comparison.png"),
        comparison
    )

    print(f"Saved results for: {name}")

# ==================================
# 6. Run inference
# ==================================

test_files = sorted(
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in EXTENSIONS
)

if not test_files:
    raise RuntimeError(
        "No SEM images found in test_images/"
    )

print(f"Found {len(test_files)} test images.")

for path in test_files:

    print(f"\nPredicting: {path.name}")

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print(f"Skipping unreadable image: {path.name}")
        continue

    mask, probability = predict_image(image)

    save_results(
        image,
        mask,
        probability,
        path.stem
    )

print("\nPrediction complete!")
print("Results saved to:", OUTPUT_DIR)
