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
# Import enhancement methods
# ------------------------------------------------------------

enhancement_folder = (
    PROJECT_ROOT /
    "layer2" /
    "enhancement"
)

sys.path.append(
    str(enhancement_folder)
)

from histogram_equalization import (
    histogram_equalization
)

from clahe import clahe_enhancement

from negative import negative

from log_transform import log_transform

from gamma import gamma_correction

from contrast_stretching import (
    contrast_stretching
)

from color_enhancement import (
    enhance_color
)


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


# ------------------------------------------------------------
# Apply enhancement
# ------------------------------------------------------------

results = {

    "original":
        image,

    "histogram_equalization":
        histogram_equalization(image),

    "clahe":
        clahe_enhancement(image),

    "negative":
        negative(image),

    "log":
        log_transform(image),

    "gamma":
        gamma_correction(
            image,
            gamma=0.8
        ),

    "contrast_stretching":
        contrast_stretching(image),

    "color_enhancement":
        enhance_color(
            image,
            brightness=1.1,
            saturation=1.2
        )
}


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_folder = (
    PROJECT_ROOT /
    "outputs" /
    "layer2" /
    "comparisons" /
    "enhancement"
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


print("Enhancement comparison completed.")
print(output_folder)