import cv2
import sys
from pathlib import Path
import numpy as np


# -------------------------------------------------
# Find Layer 3
# -------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

# outputs/
OUTPUTS_DIR = CURRENT_FILE.parents[2]

# outputs/layer3
LAYER3_DIR = OUTPUTS_DIR / "layer3"

# outputs/layer3/shape_analysis
SHAPE_ANALYSIS_DIR = LAYER3_DIR / "shape_analysis"


# Add Layer 3 folder to Python path
if str(SHAPE_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHAPE_ANALYSIS_DIR))


# -------------------------------------------------
# Import your existing Layer 3 functions
# -------------------------------------------------

from shape_descriptors import calculate_shape_descriptors
from feature_vector import create_feature_vector


# -------------------------------------------------
# Find largest contour
# -------------------------------------------------

def find_main_contour(image):

    if image is None:
        return None

    if image.size == 0:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    return max(
        contours,
        key=cv2.contourArea
    )


# -------------------------------------------------
# Extract Layer 3 feature vector
# -------------------------------------------------

def extract_layer3_features(image):

    contour = find_main_contour(image)

    if contour is None:
        return np.zeros(
            8,
            dtype=np.float32
        )

    descriptors = calculate_shape_descriptors(
        contour
    )

    feature_vector = create_feature_vector(
        descriptors
    )

    return np.asarray(
        feature_vector,
        dtype=np.float32
    ).flatten()