from pathlib import Path
import sys
import cv2
import joblib
import numpy as np


# =================================================
# PATHS
# =================================================

CURRENT_FILE = Path(__file__).resolve()

OUTPUTS_DIR = CURRENT_FILE.parents[1]

LAYER6_DIR = OUTPUTS_DIR / "layer6"

PREPROCESSING_DIR = (
    LAYER6_DIR
    / "preprocessing"
)


# =================================================
# IMPORT LAYER 6 PREPROCESSING
# =================================================

if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PREPROCESSING_DIR)
    )


from crop_detection import (
    crop_detection
)

from layer4_features import (
    extract_combined_feature_vector
)


# =================================================
# LOAD CLASSICAL ML MODEL
# =================================================

MODEL_PATH = (
    LAYER6_DIR
    / "results"
    / "vehicle_classifier.joblib"
)


classical_model = joblib.load(
    MODEL_PATH
)


# =================================================
# VERIFY TRACK
# =================================================

def verify_track(
    frame,
    track
):

    # -------------------------------------------------
    # Crop detected vehicle
    # -------------------------------------------------

    crop = crop_detection(
        frame,
        track
    )


    # -------------------------------------------------
    # Invalid crop
    # -------------------------------------------------

    if crop is None or crop.size == 0:

        return {
            "classical_class": "unknown",
            "classical_confidence": 0.0
        }


    # -------------------------------------------------
    # Resize vehicle crop
    # Reduces feature extraction time
    # -------------------------------------------------

    crop = cv2.resize(
        crop,
        (224, 224)
    )


    # -------------------------------------------------
    # Layer 3 + Layer 4
    # Combined feature vector
    # -------------------------------------------------

    features = (
        extract_combined_feature_vector(
            crop
        )
    )


    # -------------------------------------------------
    # Prepare feature vector for Random Forest
    # -------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )


    # -------------------------------------------------
    # Layer 6 Random Forest Prediction
    # -------------------------------------------------

    prediction = (
        classical_model.predict(
            features
        )[0]
    )


    # -------------------------------------------------
    # Prediction Probability
    # -------------------------------------------------

    probabilities = (
        classical_model.predict_proba(
            features
        )[0]
    )


    confidence = float(
        probabilities.max()
    )


    # -------------------------------------------------
    # Return Verification Result
    # -------------------------------------------------

    return {

        "classical_class":
            prediction,

        "classical_confidence":
            confidence
    }


# =================================================
# TEST
# =================================================

if __name__ == "__main__":

    print(
        "✅ Classical Verifier Loaded"
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Classes:",
        classical_model.classes_
    )