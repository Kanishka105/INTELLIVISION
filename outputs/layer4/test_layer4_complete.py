import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

LAYER4_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAYER4_DIR.parent.parent

FEATURE_DIR = LAYER4_DIR / "Feature_extraction"
FUSION_DIR = LAYER4_DIR / "feature_fusion"
VECTOR_DIR = LAYER4_DIR / "feature_vector"


# ============================================================
# ADD FOLDERS TO PYTHON PATH
# ============================================================

sys.path.insert(
    0,
    str(FEATURE_DIR)
)

sys.path.insert(
    0,
    str(FUSION_DIR)
)

sys.path.insert(
    0,
    str(VECTOR_DIR)
)


# ============================================================
# IMPORTS
# ============================================================

from color_features import extract_color_features
from gradient_features import extract_gradient_features
from hog_features import extract_hog_features
from orb import extract_orb_features
from sift_features import extract_sift_features
from texture_features import extract_texture_features

from feature_fusion import fuse_features

from build_feature_vector import build_feature_vector


# ============================================================
# FIND TEST IMAGE
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
        "Test image could not be loaded"
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate_vector(
    name,
    vector
):

    if vector is None:

        raise ValueError(
            f"{name} returned None"
        )

    if not isinstance(
        vector,
        np.ndarray
    ):

        raise TypeError(
            f"{name} is not a NumPy array"
        )

    if vector.size == 0:

        raise ValueError(
            f"{name} is empty"
        )

    if not np.isfinite(
        vector.astype(np.float64)
    ).all():

        raise ValueError(
            f"{name} contains NaN/Inf"
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("INTELLIVISION - LAYER 4 COMPLETE TEST")
print("=" * 70)


print("\nProject Root:")
print(PROJECT_ROOT)


print("\nTest Image:")
print(image_path.name)


print("\nImage Shape:")
print(image.shape)


# ============================================================
# 1. COLOR FEATURES
# ============================================================

print("\n" + "=" * 70)
print("1. COLOR FEATURES")
print("=" * 70)

try:

    color_features = extract_color_features(
        image
    )

    validate_vector(
        "Color Features",
        color_features
    )

    print(
        "Feature Count:",
        len(color_features)
    )

    print(
        "[PASS] Color Features"
    )

except Exception as e:

    print(
        "[FAIL] Color Features"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 2. TEXTURE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("2. TEXTURE FEATURES")
print("=" * 70)

try:

    texture_features = extract_texture_features(
        image
    )

    validate_vector(
        "Texture Features",
        texture_features
    )

    print(
        "Feature Count:",
        len(texture_features)
    )

    print(
        "[PASS] Texture Features"
    )

except Exception as e:

    print(
        "[FAIL] Texture Features"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 3. GRADIENT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("3. GRADIENT FEATURES")
print("=" * 70)

try:

    gradient_features = extract_gradient_features(
        image
    )

    validate_vector(
        "Gradient Features",
        gradient_features
    )

    print(
        "Feature Count:",
        len(gradient_features)
    )

    print(
        "[PASS] Gradient Features"
    )

except Exception as e:

    print(
        "[FAIL] Gradient Features"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 4. SIFT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("4. SIFT FEATURES")
print("=" * 70)

try:

    sift_features = extract_sift_features(
        image
    )

    validate_vector(
        "SIFT Features",
        sift_features
    )

    print(
        "Feature Count:",
        len(sift_features)
    )

    print(
        "[PASS] SIFT Features"
    )

except Exception as e:

    print(
        "[FAIL] SIFT Features"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 5. ORB FEATURES
# ============================================================

print("\n" + "=" * 70)
print("5. ORB FEATURES")
print("=" * 70)

try:

    orb_features = extract_orb_features(
        image
    )

    validate_vector(
        "ORB Features",
        orb_features
    )

    print(
        "Feature Count:",
        len(orb_features)
    )

    print(
        "[PASS] ORB Features"
    )

except Exception as e:

    print(
        "[FAIL] ORB Features"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 6. HOG FEATURES
# ============================================================

print("\n" + "=" * 70)
print("6. HOG FEATURES")
print("=" * 70)

try:

    hog_features = extract_hog_features(
        image
    )

    validate_vector(
        "HOG Features",
        hog_features
    )

    print(
        "Feature Count:",
        len(hog_features)
    )

    print(
        "[PASS] HOG Features"
    )

except Exception as e:

    print(
        "[FAIL] HOG Features"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 7. SHAPE FEATURES FROM LAYER 3
# ============================================================

print("\n" + "=" * 70)
print("7. LAYER 3 SHAPE FEATURES")
print("=" * 70)

try:

    height, width = image.shape[:2]

    # Sample contour representing an object region.
    contour = np.array(
        [
            [[20, 20]],
            [[width - 20, 20]],
            [[width - 20, height - 20]],
            [[20, height - 20]]
        ],
        dtype=np.int32
    )

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    aspect_ratio = (
        w / h
        if h != 0
        else 0.0
    )

    shape_features = np.array(
        [
            float(area),
            float(perimeter),
            float(w),
            float(h),
            float(aspect_ratio),
            0.0,
            1.0,
            1.0
        ],
        dtype=np.float32
    )

    validate_vector(
        "Shape Features",
        shape_features
    )

    print(
        "Feature Count:",
        len(shape_features)
    )

    print(
        "Shape Features:",
        shape_features
    )

    print(
        "[PASS] Layer 3 Shape Features"
    )

except Exception as e:

    print(
        "[FAIL] Layer 3 Shape Features"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 8. FEATURE FUSION
# ============================================================

print("\n" + "=" * 70)
print("8. FEATURE FUSION")
print("=" * 70)

try:

    final_vector = fuse_features(

        shape_features,

        color_features,

        texture_features,

        gradient_features,

        sift_features,

        orb_features,

        hog_features
    )

    validate_vector(
        "Final Feature Vector",
        final_vector
    )

    print(
        "Shape Features:",
        len(shape_features)
    )

    print(
        "Color Features:",
        len(color_features)
    )

    print(
        "Texture Features:",
        len(texture_features)
    )

    print(
        "Gradient Features:",
        len(gradient_features)
    )

    print(
        "SIFT Features:",
        len(sift_features)
    )

    print(
        "ORB Features:",
        len(orb_features)
    )

    print(
        "HOG Features:",
        len(hog_features)
    )

    print(
        "\nFinal Feature Vector Length:",
        len(final_vector)
    )

    print(
        "\n[PASS] Feature Fusion"
    )

except Exception as e:

    print(
        "[FAIL] Feature Fusion"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 9. BUILD FEATURE VECTOR PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("9. BUILD FEATURE VECTOR PIPELINE")
print("=" * 70)

try:

    built_vector = build_feature_vector(
        image,
        shape_features
    )

    validate_vector(
        "Built Feature Vector",
        built_vector
    )

    print(
        "Built Vector Length:",
        len(built_vector)
    )

    print(
        "[PASS] build_feature_vector()"
    )

except Exception as e:

    print(
        "[FAIL] build_feature_vector()"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# 10. CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. FEATURE VECTOR CONSISTENCY")
print("=" * 70)

try:

    if len(final_vector) != len(built_vector):

        raise ValueError(
            "Fusion vector and built vector have different sizes"
        )

    print(
        "Fusion Vector Length:",
        len(final_vector)
    )

    print(
        "Built Vector Length:",
        len(built_vector)
    )

    print(
        "[PASS] Feature Vector Size Consistency"
    )

except Exception as e:

    print(
        "[FAIL] Feature Vector Consistency"
    )

    print(
        "Error:",
        e
    )

    raise


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL LAYER 4 SUMMARY")
print("=" * 70)

print("[PASS] Color Features")
print("[PASS] Texture Features")
print("[PASS] Gradient Features")
print("[PASS] SIFT Features")
print("[PASS] ORB Features")
print("[PASS] HOG Features")
print("[PASS] Layer 3 Shape Features")
print("[PASS] Feature Fusion")
print("[PASS] Feature Vector Pipeline")
print("[PASS] Feature Vector Consistency")

print("\n" + "=" * 70)
print("✅ LAYER 4 CORE FEATURE PIPELINE PASSED")
print("=" * 70)

print("\nArchitecture:")
print(
    "Layer 3 Shape Features"
    " + Color + Texture + Gradient"
    " + SIFT + ORB + HOG"
)
print("                    ↓")
print("             Feature Fusion")
print("                    ↓")
print("             Final Feature Vector")
print("                    ↓")
print("                 Layer 5")