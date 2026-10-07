import cv2
import sys
from pathlib import Path
import numpy as np


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
# Import operations
# ------------------------------------------------------------

kernel_folder = (
    PROJECT_ROOT /
    "layer2" /
    "kernel_operations"
)

sys.path.append(
    str(kernel_folder)
)

from correlation import correlation
from convolution import convolution


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
# Asymmetric kernel
# ------------------------------------------------------------

kernel = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=np.float32)


# ------------------------------------------------------------
# Apply
# ------------------------------------------------------------

correlation_result = correlation(
    image,
    kernel
)

convolution_result = convolution(
    image,
    kernel
)


# ------------------------------------------------------------
# Difference
# ------------------------------------------------------------

difference = cv2.absdiff(
    correlation_result,
    convolution_result
)

mean_difference = np.mean(
    difference
)

max_difference = np.max(
    difference
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_folder = (
    PROJECT_ROOT /
    "outputs" /
    "layer2" /
    "comparisons" /
    "kernel"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

cv2.imwrite(
    str(output_folder / "original.jpg"),
    image
)

cv2.imwrite(
    str(output_folder / "correlation.jpg"),
    correlation_result
)

cv2.imwrite(
    str(output_folder / "convolution.jpg"),
    convolution_result
)

cv2.imwrite(
    str(output_folder / "difference.jpg"),
    difference
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("Kernel comparison completed.")

print(
    "Mean difference:",
    mean_difference
)

print(
    "Maximum difference:",
    max_difference
)

print(
    "Results saved to:",
    output_folder
)