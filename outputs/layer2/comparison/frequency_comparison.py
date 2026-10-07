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
# Frequency filter folder
# ------------------------------------------------------------

frequency_folder = (
    PROJECT_ROOT /
    "layer2" /
    "frequency_domain" /
    "filters"
)

sys.path.append(
    str(frequency_folder)
)


from low_pass import low_pass_filter
from high_pass import high_pass_filter
from band_pass import band_pass_filter
from band_reject import band_reject_filter
from notch import notch_filter


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
# Apply filters
# ------------------------------------------------------------

results = {

    "original":
        image,

    "low_pass":
        low_pass_filter(
            image,
            radius=30
        ),

    "high_pass":
        high_pass_filter(
            image,
            radius=30
        ),

    "band_pass":
        band_pass_filter(
            image,
            low_radius=20,
            high_radius=60
        ),

    "band_reject":
        band_reject_filter(
            image,
            low_radius=20,
            high_radius=60
        ),

    "notch":
        notch_filter(
            image,
            notch_points=[
                (50, 0),
                (-50, 0)
            ],
            notch_radius=5
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
    "frequency"
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


print("Frequency comparison completed.")
print(output_folder)