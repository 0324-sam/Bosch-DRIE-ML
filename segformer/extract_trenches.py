
from pathlib import Path
import csv
import cv2
import numpy as np

# =====================================
# 1. Paths and settings
# =====================================

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "images_original"
MASK_DIR = ROOT / "masks"

OUTPUT_DIR = ROOT / "trench_crops"
CROP_IMAGE_DIR = OUTPUT_DIR / "images"
CROP_MASK_DIR = OUTPUT_DIR / "masks"
PREVIEW_DIR = OUTPUT_DIR / "previews"

for folder in [CROP_IMAGE_DIR, CROP_MASK_DIR, PREVIEW_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800
MIN_CROP_SIZE = 20

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# =====================================
# 2. Find matching images and masks
# =====================================

def get_files(folder):
    result = {}

    for path in folder.iterdir():
        if path.suffix.lower() in EXTENSIONS:
            name = path.stem.lower()

            if name in result:
                raise ValueError(f"Duplicate name: {name}")

            result[name] = path

    return result


images = get_files(IMAGE_DIR)
masks = get_files(MASK_DIR)

if set(images) != set(masks):
    raise RuntimeError(
        f"Images without masks: {set(images) - set(masks)}\n"
        f"Masks without images: {set(masks) - set(images)}"
    )

print(f"Found {len(images)} image-mask pairs.")

# =====================================
# 3. Interactive crop selector
# =====================================

class CropSelector:
    def __init__(self, image, mask, name):

        self.image = image
        self.mask = mask
        self.name = name

        self.height, self.width = image.shape[:2]

        self.scale = min(
            MAX_DISPLAY_WIDTH / self.width,
            MAX_DISPLAY_HEIGHT / self.height,
            1.0
        )

        self.display_width = round(self.width * self.scale)
        self.display_height = round(self.height * self.scale)

        self.rectangles = []
        self.start = None
        self.current = None
        self.dragging = False
        self.show_overlay = True

        self.window = "Select Trenches"

    def to_original(self, x, y):
        ox = int(x / self.scale)
        oy = int(y / self.scale)

        ox = max(0, min(ox, self.width - 1))
        oy = max(0, min(oy, self.height - 1))

        return ox, oy

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self.to_original(x, y)
            self.current = self.start
            self.dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current = self.to_original(x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:

            self.current = self.to_original(x, y)
            self.dragging = False

            x1, y1 = self.start
            x2, y2 = self.current

            box = (
                min(x1, x2),
                min(y1, y2),
                max(x1, x2) + 1,
                max(y1, y2) + 1
            )

            if (
                box[2] - box[0] >= MIN_CROP_SIZE
                and box[3] - box[1] >= MIN_CROP_SIZE
            ):
                self.rectangles.append(box)

            self.start = None
            self.current = None

    def draw(self):

        display = cv2.resize(
            self.image,
            (self.display_width, self.display_height)
        )

        if self.show_overlay:

            resized_mask = cv2.resize(
                self.mask,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_NEAREST
            )

            selected = resized_mask > 127

            red = np.zeros_like(display)
            red[:, :] = (0, 0, 255)

            display[selected] = (
                0.3 * display[selected]
                + 0.7 * red[selected]
            ).astype(np.uint8)

        for i, (x1, y1, x2, y2) in enumerate(self.rectangles):

            p1 = (
                int(x1 * self.scale),
                int(y1 * self.scale)
            )

            p2 = (
                int(x2 * self.scale),
                int(y2 * self.scale)
            )

            cv2.rectangle(
                display, p1, p2,
                (0, 255, 0), 2
            )

            cv2.putText(
                display,
                f"Trench {i + 1}",
                (p1[0] + 5, max(18, p1[1] + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        if self.dragging and self.start and self.current:

            p1 = tuple(int(v * self.scale) for v in self.start)
            p2 = tuple(int(v * self.scale) for v in self.current)

            cv2.rectangle(
                display, p1, p2,
                (0, 255, 255), 2
            )

        header = np.zeros(
            (65, self.display_width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            header,
            f"{self.name} | Selected: {len(self.rectangles)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

        cv2.putText(
            header,
            "ENTER=Save  U=Undo  R=Reset  V=Overlay  S=Skip  Q=Quit",
            (10, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43, (255, 255, 255), 1
        )

        return np.vstack([header, display])

    def select(self):

        cv2.namedWindow(
            self.window,
            cv2.WINDOW_AUTOSIZE
        )

        # The 65-pixel header is outside the image.
        def handle_mouse(event, x, y, flags, param):
            if y >= 65:
                self.mouse_callback(
                    event, x, y - 65, flags, param
                )

        cv2.setMouseCallback(self.window, handle_mouse)

        while True:

            cv2.imshow(self.window, self.draw())
            key = cv2.waitKey(30) & 0xFF

            if key in [13, 10]:
                cv2.destroyWindow(self.window)
                return self.rectangles, "save"

            elif key == ord("u"):
                if self.rectangles:
                    self.rectangles.pop()

            elif key == ord("r"):
                self.rectangles.clear()

            elif key == ord("v"):
                self.show_overlay = not self.show_overlay

            elif key == ord("s"):
                cv2.destroyWindow(self.window)
                return [], "skip"

            elif key in [ord("q"), 27]:
                cv2.destroyWindow(self.window)
                return [], "quit"

# =====================================
# 4. Crop and save
# =====================================

records = []
total_saved = 0

for name in sorted(images):

    image = cv2.imread(str(images[name]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(masks[name]), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise RuntimeError(f"Cannot load: {name}")

    if image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(f"Size mismatch: {name}")

    selector = CropSelector(image, mask, name)

    rectangles, action = selector.select()

    if action == "quit":
        print("Stopped by user.")
        break

    if action == "skip":
        print(f"Skipped: {name}")
        continue

    for i, (x1, y1, x2, y2) in enumerate(rectangles, start=1):

        crop_image = image[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].copy()

        crop_mask = np.where(
            crop_mask > 127, 255, 0
        ).astype(np.uint8)

        if cv2.countNonZero(crop_mask) == 0:
            print(f"WARNING: {name} trench {i} has no boundary.")
            continue

        filename = f"{name}_trench_{i:02d}"

        cv2.imwrite(
            str(CROP_IMAGE_DIR / f"{filename}.png"),
            crop_image
        )

        cv2.imwrite(
            str(CROP_MASK_DIR / f"{filename}.png"),
            crop_mask
        )

        preview = crop_image.copy()
        selected = crop_mask > 0
        preview[selected] = (0, 0, 255)

        cv2.imwrite(
            str(PREVIEW_DIR / f"{filename}_preview.png"),
            preview
        )

        records.append([
            name, filename, x1, y1, x2, y2
        ])

        total_saved += 1

        print(f"Saved: {filename}")

# =====================================
# 5. Save crop information
# =====================================

with open(
    OUTPUT_DIR / "crop_metadata.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "source_sem",
        "crop_name",
        "x1", "y1", "x2", "y2"
    ])

    writer.writerows(records)

print("\nFinished!")
print(f"Total trenches saved: {total_saved}")
print(f"Output: {OUTPUT_DIR}")
