import cv2
import sys
from pathlib import Path


# ------------------------------------------------------------
# Find project root
# ------------------------------------------------------------

CURRENT = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in CURRENT.parents:

    if (parent / "dataset" / "train" / "images").exists():

        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "INTELLIVISION project root not found."
    )


# ------------------------------------------------------------
# Import sharpening methods
# ------------------------------------------------------------

sharpening_folder = (
    PROJECT_ROOT /
    "layer2" /
    "sharpening"
)

sys.path.append(
    str(sharpening_folder)
)

from sharpening import basic_sharpen
from Laplacian import laplacian_sharpen
from Unsharp_mask import unsharp_mask
from High_Boost import high_boost_filter


# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

image_folder = (
    PROJECT_ROOT /
    "dataset" /
    "train" /
    "images"
)

image_files = list(
    image_folder.glob("*.jpg")
)

if not image_files:
    raise FileNotFoundError(
        "No images found."
    )

image = cv2.imread(
    str(image_files[0])
)

if image is None:
    raise ValueError(
        "Image could not be loaded."
    )


# ------------------------------------------------------------
# Apply sharpening
# ------------------------------------------------------------

results = {

    "original":
        image,

    "basic_sharpen":
        basic_sharpen(image),

    "laplacian":
        laplacian_sharpen(image),

    "unsharp_mask":
        unsharp_mask(image),

    "high_boost":
        high_boost_filter(image)
}


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_folder = (
    PROJECT_ROOT /
    "outputs" /
    "layer2" /
    "comparisons" /
    "sharpening"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

for name, result in results.items():

    cv2.imwrite(
        str(output_folder / f"{name}.jpg"),
        result
    )


print("Sharpening comparison completed.")
print(output_folder)