from pathlib import Path
import sys
import cv2


# -------------------------------------------------
# Paths
# -------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

LAYER6_DIR = CURRENT_FILE.parent

OUTPUTS_DIR = LAYER6_DIR.parent

PROJECT_DIR = OUTPUTS_DIR.parent


# -------------------------------------------------
# Layer 5 imports
# -------------------------------------------------

LAYER5_DIR = (
    OUTPUTS_DIR / "layer5"
)

YOLO_DIR = (
    LAYER5_DIR / "yolo"
)

POSTPROCESSING_DIR = (
    LAYER5_DIR / "postprocessing"
)

VISUALIZATION_DIR = (
    LAYER5_DIR / "visualization"
)


for directory in [
    YOLO_DIR,
    POSTPROCESSING_DIR,
    VISUALIZATION_DIR
]:

    if str(directory) not in sys.path:

        sys.path.insert(
            0,
            str(directory)
        )


from model_loader import (
    load_model
)

from detector import (
    detect_objects
)

from confidence import (
    filter_by_confidence
)

from draw_detections import (
    draw_detections
)


# -------------------------------------------------
# Layer 6 imports
# -------------------------------------------------

PREPROCESSING_DIR = (
    LAYER6_DIR / "preprocessing"
)

CLASSIFIER_DIR = (
    LAYER6_DIR / "classifier"
)

VERIFICATION_DIR = (
    LAYER6_DIR / "verification"
)


for directory in [
    PREPROCESSING_DIR,
    CLASSIFIER_DIR,
    VERIFICATION_DIR
]:

    if str(directory) not in sys.path:

        sys.path.insert(
            0,
            str(directory)
        )


from crop_detections import (
    crop_detection
)

from layer4_features import (
    extract_layer4_feature_vector
)

from predict import (
    load_classifier,
    predict_vehicle
)

from compare_predictions import (
    compare_predictions
)


# -------------------------------------------------
# Model paths
# -------------------------------------------------

YOLO_MODEL_PATH = (
    LAYER5_DIR
    / "results"
    / "vehicle_detector"
    / "weights"
    / "best.pt"
)


# -------------------------------------------------
# Pipeline
# -------------------------------------------------

def run_layer6(
    image_path,
    output_path,
    yolo_confidence=0.25,
    final_confidence=0.50
):

    # ---------------------------------------------
    # Load image
    # ---------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    # ---------------------------------------------
    # Load Layer 5 YOLO
    # ---------------------------------------------

    print("Loading YOLO...")

    yolo_model = load_model(
        str(YOLO_MODEL_PATH)
    )

    # ---------------------------------------------
    # YOLO detection
    # ---------------------------------------------

    print("Running YOLO detection...")

    detections = detect_objects(
        image,
        yolo_model,
        confidence=yolo_confidence,
        iou_threshold=0.45
    )

    detections = filter_by_confidence(
        detections,
        threshold=final_confidence
    )

    # ---------------------------------------------
    # Load Layer 6 classifier
    # ---------------------------------------------

    print("Loading classical classifier...")

    classifier = load_classifier()

    verification_results = []

    # ---------------------------------------------
    # Process each YOLO detection
    # ---------------------------------------------

    for index, detection in enumerate(
        detections,
        start=1
    ):

        crop = crop_detection(
            image,
            detection
        )

        if crop is None:
            continue

        # -----------------------------------------
        # Layer 4
        # -----------------------------------------

        feature_vector = (
            extract_layer4_feature_vector(
                crop
            )
        )

        # -----------------------------------------
        # Layer 6
        # -----------------------------------------

        classical_class, classical_confidence = (
            predict_vehicle(
                classifier,
                feature_vector
            )
        )

        # -----------------------------------------
        # Compare Layer 5 vs Layer 6
        # -----------------------------------------

        result = compare_predictions(

            yolo_class=detection[
                "class_name"
            ],

            yolo_confidence=detection[
                "confidence"
            ],

            classical_class=classical_class,

            classical_confidence=
                classical_confidence
        )

        result["object_id"] = index

        result["bbox"] = detection[
            "bbox"
        ]

        verification_results.append(
            result
        )

        print(
            f"\nObject {index}"
        )

        print(
            f"YOLO       : "
            f"{detection['class_name']} "
            f"({detection['confidence']:.2f})"
        )

        print(
            f"Classical  : "
            f"{classical_class} "
            f"({classical_confidence:.2f})"
        )

        print(
            f"Status     : "
            f"{result['status']}"
        )

    # ---------------------------------------------
    # Draw YOLO detections
    # ---------------------------------------------

    result_image = draw_detections(
        image,
        detections
    )

    cv2.imwrite(
        str(output_path),
        result_image
    )

    return verification_results


# -------------------------------------------------
# Example execution
# -------------------------------------------------

if __name__ == "__main__":

    test_image = (
        PROJECT_DIR
        / "dataset"
        / "test"
        / "images"
        / "bus_104_jpg.rf.9e59f1c2a1c4681e72fdcad4a754fed1.jpg"
    )

    output_image = (
        LAYER6_DIR
        / "results"
        / "layer6_prediction.jpg"
    )

    results = run_layer6(
        test_image,
        output_image
    )

    print("\n================================")
    print("LAYER 6 FINISHED")
    print("================================")

    print(
        f"Objects processed: "
        f"{len(results)}"
    )