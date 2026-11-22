
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "images_original"
ANNOTATED = ROOT / "images_annotated"
MASKS = ROOT / "masks"
OUTPUTS = ROOT / "outputs" / "mask_checks"

MASKS.mkdir(exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def collect_images(folder):
    result = {}
    for path in folder.iterdir():
        if path.suffix.lower() in EXTENSIONS:
            key = path.stem.lower()
            if key in result:
                raise ValueError(f"Duplicate image name: {key}")
            result[key] = path
    return result

def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot open: {path}")
    return image

def extract_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = cv2.inRange(
        hsv, np.array([0, 80, 70]), np.array([12, 255, 255])
    )
    upper_red = cv2.inRange(
        hsv, np.array([168, 80, 70]), np.array([179, 255, 255])
    )

    return cv2.bitwise_or(lower_red, upper_red)

originals = collect_images(ORIGINAL)
annotations = collect_images(ANNOTATED)

if not originals:
    raise RuntimeError("No original SEM images found.")

if originals.keys() != annotations.keys():
    missing = originals.keys() - annotations.keys()
    extra = annotations.keys() - originals.keys()
    raise RuntimeError(
        f"Unmatched filenames. Missing annotations: {missing}; "
        f"Unmatched annotations: {extra}"
    )

print(f"Found {len(originals)} image pairs.")

for key in sorted(originals):
    original = load_image(originals[key])
    annotated = load_image(annotations[key])

    height, width = original.shape[:2]
    ah, aw = annotated.shape[:2]

    # Resize annotation to the original image dimensions.
    if (ah, aw) != (height, width):
        ratio_original = width / height
        ratio_annotated = aw / ah

        if abs(ratio_original - ratio_annotated) > 0.03:
            print(f"WARNING: Different aspect ratios for {key}")

        annotated = cv2.resize(
            annotated, (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    red_mask = extract_red(annotated)

    if cv2.countNonZero(red_mask) == 0:
        print(f"WARNING: No red pixels detected in {key}")
        continue

    # Remove red annotation from registration input.
    clean_annotated = cv2.inpaint(
        annotated, red_mask, 3, cv2.INPAINT_TELEA
    )

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    annotated_gray = cv2.cvtColor(
        clean_annotated, cv2.COLOR_BGR2GRAY
    )

    # Align using ECC registration.
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        500, 1e-6
    )

    try:
        score, warp_matrix = cv2.findTransformECC(
            original_gray,
            annotated_gray,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria
        )
        print(f"{key}: ECC score = {score:.4f}")
    except cv2.error:
        print(f"WARNING: ECC failed for {key}; using initial alignment.")

    aligned_mask = cv2.warpAffine(
        red_mask,
        warp_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Binary label: background = 0, boundary = 1.
    binary_mask = (aligned_mask > 0).astype(np.uint8)

    # Save a viewable mask: 0 = black, 255 = white.
    cv2.imwrite(
        str(MASKS / f"{key}.png"),
        binary_mask * 255
    )

    # Overlay the boundary in red on original SEM.
    overlay = original.copy()
    overlay[binary_mask == 1] = (0, 0, 255)

    cv2.imwrite(
        str(OUTPUTS / f"{key}_overlay.png"),
        overlay
    )

    print(f"Saved mask and overlay for {key}")

print("Finished. Please inspect all mask overlays.")
