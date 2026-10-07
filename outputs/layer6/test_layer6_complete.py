from pathlib import Path
import sys
import cv2
import numpy as np
import joblib


# =================================================
# PATHS
# =================================================

CURRENT_FILE = Path(__file__).resolve()

LAYER6_DIR = CURRENT_FILE.parent

PROJECT_DIR = (
    LAYER6_DIR.parent.parent
)

TEST_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "test"
    / "images"
    / "bus_104_jpg.rf.9e59f1c2a1c4681e72fdcad4a754fed1.jpg"
)

MODEL_PATH = (
    LAYER6_DIR
    / "results"
    / "vehicle_classifier.joblib"
)


# =================================================
# IMPORT PATH
# =================================================

if str(LAYER6_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(LAYER6_DIR)
    )


# =================================================
# IMPORT LAYER 6 MODULES
# =================================================

from preprocessing.crop_detection import (
    crop_detection
)

from preprocessing.layer3_features import (
    extract_layer3_features
)

from preprocessing.layer4_features import (
    extract_combined_feature_vector
)

from verification.compare_predictions import (
    compare_predictions
)


# =================================================
# TEST 1 — IMAGE
# =================================================

def test_image_loading():

    image = cv2.imread(
        str(TEST_IMAGE)
    )

    assert image is not None, (
        f"Image not found: {TEST_IMAGE}"
    )

    print("✅ Image Loading")


# =================================================
# TEST 2 — CROPPING
# =================================================

def test_crop():

    image = cv2.imread(
        str(TEST_IMAGE)
    )

    detection = {
        "bbox": [
            10,
            10,
            min(200, image.shape[1]),
            min(200, image.shape[0])
        ]
    }

    crop = crop_detection(
        image,
        detection
    )

    assert crop is not None

    assert crop.size > 0

    print("✅ Vehicle Cropping")

    return crop


# =================================================
# TEST 3 — LAYER 3
# =================================================

def test_layer3_features(crop):

    features = extract_layer3_features(
        crop
    )

    assert isinstance(
        features,
        np.ndarray
    )

    assert len(features) == 8

    print("✅ Layer 3 Features")

    print(
        f"   Shape Feature Length: "
        f"{len(features)}"
    )


# =================================================
# TEST 4 — LAYER 3 + LAYER 4
# =================================================

def test_combined_features(crop):

    features = (
        extract_combined_feature_vector(
            crop
        )
    )

    assert isinstance(
        features,
        np.ndarray
    )

    assert len(features) > 8

    assert np.isfinite(
        features
    ).all()

    print("✅ Layer 3 + Layer 4 Feature Fusion")

    print(
        f"   Combined Feature Length: "
        f"{len(features)}"
    )

    return features


# =================================================
# TEST 5 — RANDOM FOREST MODEL
# =================================================

def test_classifier(features):

    assert MODEL_PATH.exists(), (
        f"Classifier not found: {MODEL_PATH}"
    )

    model = joblib.load(
        MODEL_PATH
    )

    prediction = model.predict(
        features.reshape(1, -1)
    )

    probabilities = model.predict_proba(
        features.reshape(1, -1)
    )

    predicted_class = prediction[0]

    confidence = float(
        probabilities.max()
    )

    assert predicted_class in model.classes_

    print("✅ Random Forest Prediction")

    print(
        f"   Prediction : {predicted_class}"
    )

    print(
        f"   Confidence : {confidence:.4f}"
    )

    print(
        f"   Classes    : "
        f"{list(model.classes_)}"
    )

    return predicted_class, confidence


# =================================================
# TEST 6 — VERIFICATION
# =================================================

def test_verification(
    classical_class,
    classical_confidence
):

    result = compare_predictions(

        yolo_class="bus",

        yolo_confidence=0.90,

        classical_class=classical_class,

        classical_confidence=classical_confidence
    )

    assert "status" in result

    assert "agreement" in result

    print("✅ YOLO / Classical Verification")

    print(
        f"   YOLO       : "
        f"{result['yolo_class']}"
    )

    print(
        f"   Classical  : "
        f"{result['classical_class']}"
    )

    print(
        f"   Status     : "
        f"{result['status']}"
    )


# =================================================
# MAIN
# =================================================

def run_all_tests():

    print("\n========================================")
    print("       LAYER 6 COMPLETE TEST")
    print("========================================\n")

    # 1
    test_image_loading()

    # 2
    crop = test_crop()

    # 3
    test_layer3_features(
        crop
    )

    # 4
    features = test_combined_features(
        crop
    )

    # 5
    classical_class, classical_confidence = (
        test_classifier(
            features
        )
    )

    # 6
    test_verification(
        classical_class,
        classical_confidence
    )

    print("\n========================================")
    print("✅ LAYER 6 COMPLETE")
    print("========================================\n")


if __name__ == "__main__":
    run_all_tests()