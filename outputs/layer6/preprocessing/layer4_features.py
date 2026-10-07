import sys
from pathlib import Path
import numpy as np


# -------------------------------------------------
# Locate Layer 4
# -------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

# outputs/
OUTPUTS_DIR = CURRENT_FILE.parents[2]

# outputs/layer4/Feature_extraction
LAYER4_FEATURE_DIR = (
    OUTPUTS_DIR
    / "layer4"
    / "Feature_extraction"
)


if str(LAYER4_FEATURE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(LAYER4_FEATURE_DIR)
    )


# -------------------------------------------------
# Existing Layer 4 functions
# -------------------------------------------------

from color_features import extract_color_features
from gradient_features import extract_gradient_features
from hog_features import extract_hog_features
from orb import extract_orb_features
from sift_features import extract_sift_features
from texture_features import extract_texture_features


# -------------------------------------------------
# Layer 3 connection
# -------------------------------------------------

PREPROCESSING_DIR = CURRENT_FILE.parent

if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PREPROCESSING_DIR)
    )


from layer3_features import extract_layer3_features


# -------------------------------------------------
# Layer 3 + Layer 4
# -------------------------------------------------

def extract_combined_feature_vector(image):
    """
    Combine Layer 3 shape features with
    Layer 4 visual features.
    """

    # -----------------------------
    # Layer 3
    # -----------------------------

    shape_features = extract_layer3_features(
        image
    )

    shape_features = np.asarray(
        shape_features,
        dtype=np.float32
    ).flatten()


    # -----------------------------
    # Layer 4
    # -----------------------------

    color_features = np.asarray(
        extract_color_features(image),
        dtype=np.float32
    ).flatten()

    texture_features = np.asarray(
        extract_texture_features(image),
        dtype=np.float32
    ).flatten()

    gradient_features = np.asarray(
        extract_gradient_features(image),
        dtype=np.float32
    ).flatten()

    sift_features = np.asarray(
        extract_sift_features(image),
        dtype=np.float32
    ).flatten()

    orb_features = np.asarray(
        extract_orb_features(image),
        dtype=np.float32
    ).flatten()

    hog_features = np.asarray(
        extract_hog_features(image),
        dtype=np.float32
    ).flatten()


    # -----------------------------
    # Fusion
    # -----------------------------

    combined_features = np.concatenate(
        [
            shape_features,
            color_features,
            texture_features,
            gradient_features,
            sift_features,
            orb_features,
            hog_features
        ]
    )


    return combined_features.astype(
        np.float32
    )