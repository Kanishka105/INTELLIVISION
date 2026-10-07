from pathlib import Path
import sys
import cv2
import joblib


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_DIR / "outputs"

LAYER2_DIR = OUTPUTS_DIR / "layer2"
LAYER5_DIR = OUTPUTS_DIR / "layer5"
LAYER6_DIR = OUTPUTS_DIR / "layer6"

PIPELINE_OUTPUT_DIR = (
    OUTPUTS_DIR / "final_pipeline"
)

PIPELINE_OUTPUT_DIR.mkdir(
    exist_ok=True
)


# =========================================================
# TEST IMAGE
# =========================================================

INPUT_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "test"
    / "images"
    / "bus_104_jpg.rf.9e59f1c2a1c4681e72fdcad4a754fed1.jpg"
)


# =========================================================
# LAYER 2 IMPORT
# =========================================================

LAYER2_SMOOTHING_DIR = (
    LAYER2_DIR
    / "Smoothing"
    / "Non_Linear"
)

sys.path.insert(
    0,
    str(LAYER2_SMOOTHING_DIR)
)

from bilateral_filter import bilateral_filter


# =========================================================
# LAYER 5 IMPORTS
# =========================================================

LAYER5_YOLO_DIR = (
    LAYER5_DIR / "yolo"
)

LAYER5_POSTPROCESSING_DIR = (
    LAYER5_DIR / "postprocessing"
)


sys.path.insert(
    0,
    str(LAYER5_YOLO_DIR)
)

sys.path.insert(
    0,
    str(LAYER5_POSTPROCESSING_DIR)
)

from model_loader import load_model
from detector import detect_objects
from confidence import filter_by_confidence


# =========================================================
# LAYER 6 IMPORTS
# =========================================================

LAYER6_PREPROCESSING_DIR = (
    LAYER6_DIR / "preprocessing"
)

sys.path.insert(
    0,
    str(LAYER6_PREPROCESSING_DIR)
)

from crop_detection import crop_detection

from layer4_features import (
    extract_combined_feature_vector
)


# =========================================================
# LAYER 6 CLASSIFIER
# =========================================================

CLASSIFIER_PATH = (
    LAYER6_DIR
    / "results"
    / "vehicle_classifier.joblib"
)

classifier = joblib.load(
    CLASSIFIER_PATH
)


# =========================================================
# LAYER 5 YOLO MODEL
# =========================================================

YOLO_MODEL_PATH = (
    LAYER5_DIR
    / "results"
    / "vehicle_detector"
    / "weights"
    / "best.pt"
)


# =========================================================
# LAYER 2
# =========================================================

def run_layer2(image):
    """
    Layer 2:
    Image smoothing / preprocessing.
    """

    processed = bilateral_filter(
        image
    )

    return processed


# =========================================================
# LAYER 5
# =========================================================

def run_layer5(image):
    """
    Layer 5:
    YOLO object detection.
    """

    model = load_model(
        str(YOLO_MODEL_PATH)
    )

    detections = detect_objects(
        image,
        model,
        confidence=0.25,
        iou_threshold=0.45
    )

    detections = filter_by_confidence(
        detections,
        threshold=0.50
    )

    return detections


# =========================================================
# LAYER 6
# =========================================================

def run_layer6(
    image,
    detections
):
    """
    Layer 6:

    YOLO crop
       ↓
    Layer 3
       ↓
    Layer 4
       ↓
    Random Forest
    """

    results = []

    for detection in detections:

        # -----------------------------------------
        # Crop YOLO detected vehicle
        # -----------------------------------------

        crop = crop_detection(
            image,
            detection
        )

        if crop is None:
            continue

        if crop.size == 0:
            continue


        # -----------------------------------------
        # Layer 3 + Layer 4
        # -----------------------------------------

        features = (
            extract_combined_feature_vector(
                crop
            )
        )


        # -----------------------------------------
        # Classical ML
        # -----------------------------------------

        prediction = classifier.predict(
            features.reshape(
                1,
                -1
            )
        )[0]


        probabilities = (
            classifier.predict_proba(
                features.reshape(
                    1,
                    -1
                )
            )[0]
        )


        classical_confidence = float(
            probabilities.max()
        )


        # -----------------------------------------
        # Store result
        # -----------------------------------------

        result = {
            "yolo_class":
                detection["class_name"],

            "yolo_confidence":
                detection["confidence"],

            "classical_class":
                prediction,

            "classical_confidence":
                classical_confidence,

            "bbox":
                detection["bbox"],

            "agreement":
                detection["class_name"]
                == prediction
        }


        results.append(
            result
        )


    return results


# =========================================================
# DRAW FINAL RESULT
# =========================================================

def draw_results(
    image,
    results
):

    output = image.copy()


    for result in results:

        x1, y1, x2, y2 = map(
            int,
            result["bbox"]
        )


        yolo_class = (
            result["yolo_class"]
        )

        yolo_conf = (
            result["yolo_confidence"]
        )

        classical_class = (
            result["classical_class"]
        )

        classical_conf = (
            result["classical_confidence"]
        )


        label = (
            f"YOLO: {yolo_class} "
            f"{yolo_conf:.2f}"
        )

        label2 = (
            f"ML: {classical_class} "
            f"{classical_conf:.2f}"
        )


        # Box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # YOLO label

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 25, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


        # Classical label

        cv2.putText(
            output,
            label2,
            (x1, max(y1 - 5, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )


    return output


# =========================================================
# MAIN SEQUENTIAL PIPELINE
# =========================================================

def main():

    print("\n========================================")
    print("      INTELLIVISION MASTER PIPELINE")
    print("========================================")


    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    print("\n[INPUT] Loading image...")

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n"
            f"{INPUT_IMAGE}"
        )

    print("✅ Image loaded")


    # -----------------------------------------------------
    # Layer 2
    # -----------------------------------------------------

    print("\n[LAYER 2] Image Processing...")

    processed_image = run_layer2(
        image
    )

    print("✅ Layer 2 complete")


    # -----------------------------------------------------
    # Layer 5
    # -----------------------------------------------------

    print("\n[LAYER 5] YOLO Detection...")

    detections = run_layer5(
        processed_image
    )

    print(
        f"✅ YOLO detected "
        f"{len(detections)} objects"
    )


    # -----------------------------------------------------
    # Layer 6
    # -----------------------------------------------------

    print(
        "\n[LAYER 6] Classical ML Verification..."
    )

    results = run_layer6(
        processed_image,
        detections
    )


    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nObject {index}"
        )

        print(
            f"YOLO       : "
            f"{result['yolo_class']} "
            f"({result['yolo_confidence']:.2f})"
        )

        print(
            f"Classical  : "
            f"{result['classical_class']} "
            f"({result['classical_confidence']:.2f})"
        )

        if result["agreement"]:

            print(
                "Status     : AGREEMENT ✅"
            )

        else:

            print(
                "Status     : DISAGREEMENT ⚠️"
            )


    # -----------------------------------------------------
    # Draw result
    # -----------------------------------------------------

    final_image = draw_results(
        processed_image,
        results
    )


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = (
        PIPELINE_OUTPUT_DIR
        / "final_result.jpg"
    )

    cv2.imwrite(
        str(output_path),
        final_image
    )


    print(
        f"\n✅ Final output saved to:\n"
        f"{output_path}"
    )


    print("\n========================================")
    print("✅ MASTER PIPELINE COMPLETE")
    print("========================================\n")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()