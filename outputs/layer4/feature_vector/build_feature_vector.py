import sys
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
LAYER4_DIR = CURRENT_DIR.parent

FEATURE_DIR = LAYER4_DIR / "Feature_extraction"
FUSION_DIR = LAYER4_DIR / "feature_fusion"


# ============================================================
# ADD FEATURE FOLDERS TO PYTHON PATH
# ============================================================

sys.path.insert(
    0,
    str(FEATURE_DIR)
)

sys.path.insert(
    0,
    str(FUSION_DIR)
)


# ============================================================
# IMPORT FEATURE EXTRACTORS
# ============================================================

from color_features import extract_color_features

from texture_features import extract_texture_features

from gradient_features import extract_gradient_features

from sift_features import extract_sift_features

# IMPORTANT:
# Your actual file is orb.py
from orb import extract_orb_features

from hog_features import extract_hog_features

from feature_fusion import fuse_features


# ============================================================
# BUILD FINAL FEATURE VECTOR
# ============================================================

def build_feature_vector(
    object_image,
    shape_features
):

    # -----------------------------
    # Layer 4 feature extraction
    # -----------------------------

    color_features = extract_color_features(
        object_image
    )

    texture_features = extract_texture_features(
        object_image
    )

    gradient_features = extract_gradient_features(
        object_image
    )

    sift_features = extract_sift_features(
        object_image
    )

    orb_features = extract_orb_features(
        object_image
    )

    hog_features = extract_hog_features(
        object_image
    )


    # -----------------------------
    # Feature fusion
    # -----------------------------

    final_vector = fuse_features(

        shape_features,

        color_features,

        texture_features,

        gradient_features,

        sift_features,

        orb_features,

        hog_features
    )


    return final_vector
# Instead of manually calling every feature extractor, this file does: