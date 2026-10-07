import cv2
from pathlib import Path

from sharpening import basic_sharpen
from Laplacian import laplacian_sharpen
from Unsharp_mask import unsharp_mask
from High_Boost import high_boost_filter
# FIND PROJECT ROOT
# ==========================================

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in current_path.parents:

    if (
        (parent / "dataset" / "train" / "images").exists()
        and
        (parent / "dataset" / "val" / "images").exists()
        and
        (parent / "dataset" / "test" / "images").exists()
    ):
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not find INTELLIVISION project root."
    )


# ==========================================
# DATASET PATH
# ==========================================

IMAGE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "images"
)


# ==========================================
# OUTPUT PATH
# ==========================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "layer2"
    / "sharpening"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# PROJECT INFORMATION
# ==========================================

print("\n========================================")
print("       INTELLIVISION - SHARPENING")
print("========================================")

print("\nProject root:")
print(PROJECT_ROOT)

print("\nImage directory:")
print(IMAGE_DIR)

print("\nDataset exists:")
print(IMAGE_DIR.exists())

print("\nOutput directory:")
print(OUTPUT_DIR)


# ==========================================
# FIND IMAGE
# ==========================================

image_files = [
    p
    for p in IMAGE_DIR.iterdir()
    if p.suffix.lower()
    in {".jpg", ".jpeg", ".png", ".bmp"}
]


if not image_files:
    raise FileNotFoundError(
        "No images found in dataset."
    )


image_path = image_files[0]


# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(
    str(image_path)
)


if image is None:
    raise ValueError(
        f"Unable to load image: {image_path}"
    )


print("\nOriginal image:")
print(image_path)

print("Image shape:")
print(image.shape)


# ==========================================
# APPLY SHARPENING METHODS
# ==========================================

print("\nApplying sharpening methods...")


basic = basic_sharpen(
    image
)


laplacian = laplacian_sharpen(
    image
)


unsharp = unsharp_mask(
    image,
    sigma=1.0,
    amount=1.5
)


high_boost = high_boost_filter(
    image,
    A=2.0,
    sigma=1.0
)


# ==========================================
# SAVE RESULTS
# ==========================================

cv2.imwrite(
    str(OUTPUT_DIR / "original.jpg"),
    image
)

cv2.imwrite(
    str(OUTPUT_DIR / "basic_sharpen.jpg"),
    basic
)

cv2.imwrite(
    str(OUTPUT_DIR / "laplacian_sharpen.jpg"),
    laplacian
)

cv2.imwrite(
    str(OUTPUT_DIR / "unsharp_mask.jpg"),
    unsharp
)

cv2.imwrite(
    str(OUTPUT_DIR / "high_boost.jpg"),
    high_boost
)


# ==========================================
# COMPLETION
# ==========================================

print("\n========================================")
print("       SHARPENING COMPLETED")
print("========================================")

print("\nMethods tested:")

print("✓ Basic Sharpening")
print("✓ Laplacian Sharpening")
print("✓ Unsharp Masking")
print("✓ High-Boost Filtering")

print("\nResults saved at:")
print(OUTPUT_DIR)

print("\n========================================")