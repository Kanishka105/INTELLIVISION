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
# Import filters
# ------------------------------------------------------------

sys.path.append(
    str(PROJECT_ROOT / "layer2" / "filtering")
)

from mean_filter import mean_filter
from weighted_mean import weighted_mean_filter
from gaussian import gaussian_smoothing
from median_filter import median_filter
from min_filter import min_filter
from max_filter import max_filter
from bilateral_filter import bilateral_filter

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
# Apply filters
# ------------------------------------------------------------

results = {
    "original": image,

    "mean": mean_filter(image),

    "weighted_mean":
        weighted_mean_filter(image),

    "gaussian":
        gaussian_smoothing(image),

    "median":
        median_filter(image),

    "min":
        min_filter(image),

    "max":
        max_filter(image),

    "bilateral":
        bilateral_filter(image)
}

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

output_folder = (
    PROJECT_ROOT /
    "outputs" /
    "layer2" /
    "comparisons" /
    "filtering"
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

print("Filtering comparison completed.")

print(f"Results saved to:")
print(output_folder)