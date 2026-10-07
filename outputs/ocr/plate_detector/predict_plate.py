from pathlib import Path
import cv2
from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PLATE_DETECTOR_DIR = CURRENT_FILE.parent
OCR_DIR = PLATE_DETECTOR_DIR.parent
PROJECT_DIR = OCR_DIR.parent.parent


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = (
    OCR_DIR
    / "results"
    / "plate_detector"
    / "weights"
    / "best.pt"
)


# =========================================================
# UNSEEN TEST IMAGE
# =========================================================

TEST_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "plate"
    / "test"
    / "images"
    / "image_0001.jpg"
)


# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = (
    OCR_DIR
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 50)
    print("      INTELLIVISION PLATE DETECTOR")
    print("=" * 50)


    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    # -----------------------------------------------------
    # Check test image
    # -----------------------------------------------------

    if not TEST_IMAGE.exists():

        raise FileNotFoundError(
            f"Test image not found:\n{TEST_IMAGE}"
        )


    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("\nLoading model...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("✅ Model loaded")


    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    print("\nTest image:")
    print(TEST_IMAGE)

    image = cv2.imread(
        str(TEST_IMAGE)
    )

    if image is None:

        raise ValueError(
            "Could not load test image."
        )


    print(
        "Image shape:",
        image.shape
    )


    # -----------------------------------------------------
    # Plate detection
    # -----------------------------------------------------

    print(
        "\nRunning plate detection..."
    )

    results = model.predict(
        source=image,

        # Diagnostic threshold
        conf=0.05,

        # Limit maximum detections
        max_det=3,

        device="cpu",

        verbose=False
    )


    output = image.copy()

    detected = 0


    # -----------------------------------------------------
    # Process detections
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:

            continue


        for box in result.boxes:

            detected += 1


            # ---------------------------------------------
            # Bounding box
            # ---------------------------------------------

            bbox = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            x1, y1, x2, y2 = bbox


            # ---------------------------------------------
            # Confidence
            # ---------------------------------------------

            confidence = float(
                box.conf[0]
                .cpu()
                .item()
            )


            print(
                f"Plate {detected}: "
                f"bbox=({x1}, {y1}, "
                f"{x2}, {y2}), "
                f"confidence={confidence:.2f}"
            )


            # ---------------------------------------------
            # Draw bounding box
            # ---------------------------------------------

            cv2.rectangle(
                output,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            label = (
                f"Plate {confidence:.2f}"
            )


            cv2.putText(
                output,

                label,

                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2
            )


    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "plate_prediction.jpg"
    )


    success = cv2.imwrite(
        str(output_path),
        output
    )


    if not success:

        raise RuntimeError(
            "Could not save output image."
        )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        f"\nDetections: {detected}"
    )

    print(
        f"Output saved to:\n{output_path}"
    )


    print(
        "\n" + "=" * 50
    )


    if detected > 0:

        print(
            "✅ MODEL PRODUCED TEST PREDICTION"
        )

    else:

        print(
            "⚠️ NO PLATE DETECTED"
        )


    print(
        "=" * 50
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()