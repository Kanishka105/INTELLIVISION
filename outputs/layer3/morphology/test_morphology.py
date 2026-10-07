import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATH
# ============================================================

MORPHOLOGY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MORPHOLOGY_DIR.parent.parent.parent

sys.path.insert(
    0,
    str(MORPHOLOGY_DIR)
)


# ============================================================
# IMPORTS
# ============================================================

from structuring_element import create_structuring_element
from Erosion import erosion
from Dilation import dilation
from Opening import opening
from Closing import closing
from gradient import morphological_gradient
from top_hat import top_hat
from black_hat import black_hat


# ============================================================
# LOAD IMAGE
# ============================================================

image_folder = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "images"
)

image_files = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp"
]:

    image_files.extend(
        image_folder.glob(extension)
    )

if not image_files:

    raise FileNotFoundError(
        f"No images found in:\n{image_folder}"
    )


image_path = image_files[0]

image = cv2.imread(
    str(image_path)
)

if image is None:

    raise ValueError(
        "Image could not be loaded"
    )


# ============================================================
# CONVERT TO GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# VALIDATION
# ============================================================

def validate_output(
    name,
    output
):

    if output is None:
        raise ValueError(
            f"{name} returned None"
        )

    if not isinstance(
        output,
        np.ndarray
    ):
        raise TypeError(
            f"{name} did not return NumPy array"
        )

    if output.size == 0:
        raise ValueError(
            f"{name} returned empty output"
        )

    if not np.isfinite(output).all():
        raise ValueError(
            f"{name} contains invalid values"
        )


# ============================================================
# TEST
# ============================================================

print("=" * 60)
print("INTELLIVISION - LAYER 3 MORPHOLOGY TEST")
print("=" * 60)

print("\nTest Image:")
print(image_path.name)

print("\nImage Shape:")
print(image.shape)


# ------------------------------------------------------------
# STRUCTURING ELEMENT
# ------------------------------------------------------------

try:

    kernel = create_structuring_element(
        shape="ellipse",
        kernel_size=(5, 5)
    )

    print("\n[PASS] Structuring Element")
    print(kernel)

except Exception as e:

    print("\n[FAIL] Structuring Element")
    print("Error:", e)


# ------------------------------------------------------------
# EROSION
# ------------------------------------------------------------

try:

    result = erosion(gray)

    validate_output(
        "Erosion",
        result
    )

    print("[PASS] Erosion")

except Exception as e:

    print("[FAIL] Erosion")
    print("Error:", e)


# ------------------------------------------------------------
# DILATION
# ------------------------------------------------------------

try:

    result = dilation(gray)

    validate_output(
        "Dilation",
        result
    )

    print("[PASS] Dilation")

except Exception as e:

    print("[FAIL] Dilation")
    print("Error:", e)


# ------------------------------------------------------------
# OPENING
# ------------------------------------------------------------

try:

    result = opening(gray)

    validate_output(
        "Opening",
        result
    )

    print("[PASS] Opening")

except Exception as e:

    print("[FAIL] Opening")
    print("Error:", e)


# ------------------------------------------------------------
# CLOSING
# ------------------------------------------------------------

try:

    result = closing(gray)

    validate_output(
        "Closing",
        result
    )

    print("[PASS] Closing")

except Exception as e:

    print("[FAIL] Closing")
    print("Error:", e)


# ------------------------------------------------------------
# MORPHOLOGICAL GRADIENT
# ------------------------------------------------------------

try:

    result = morphological_gradient(gray)

    validate_output(
        "Morphological Gradient",
        result
    )

    print("[PASS] Morphological Gradient")

except Exception as e:

    print("[FAIL] Morphological Gradient")
    print("Error:", e)


# ------------------------------------------------------------
# TOP HAT
# ------------------------------------------------------------

try:

    result = top_hat(gray)

    validate_output(
        "Top Hat",
        result
    )

    print("[PASS] Top Hat")

except Exception as e:

    print("[FAIL] Top Hat")
    print("Error:", e)


# ------------------------------------------------------------
# BLACK HAT
# ------------------------------------------------------------

try:

    result = black_hat(gray)

    validate_output(
        "Black Hat",
        result
    )

    print("[PASS] Black Hat")

except Exception as e:

    print("[FAIL] Black Hat")
    print("Error:", e)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("LAYER 3 MORPHOLOGY TEST FINISHED")
print("=" * 60)