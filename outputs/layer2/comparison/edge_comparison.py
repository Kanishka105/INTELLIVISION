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
# Import edge detectors
# ------------------------------------------------------------

edge_folder = (
    PROJECT_ROOT /
    "layer2" /
    "edge_detection"
)

sys.path.append(
    str(edge_folder)
)

from sobel import sobel_edge_detection
from scharr import scharr_edge_detection
from prewitt import prewitt_edge_detection
from roberts import roberts_edge_detection
from laplacian import laplacian_edge_detection
from canny import canny_edge_detection


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
# Apply edge detectors
# ------------------------------------------------------------

results = {

    "original":
        image,

    "sobel":
        sobel_edge_detection(image),

    "scharr":
        scharr_edge_detection(image),

    "prewitt":
        prewitt_edge_detection(image),

    "roberts":
        roberts_edge_detection(image),

    "laplacian":
        laplacian_edge_detection(image),

    "canny":
        canny_edge_detection(
            image,
            50,
            150
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
    "edge_detection"
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


print("Edge comparison completed.")
print(output_folder)