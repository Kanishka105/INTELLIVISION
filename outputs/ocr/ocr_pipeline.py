from pathlib import Path
import sys
import cv2
from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

OCR_DIR = CURRENT_FILE.parent

PROJECT_DIR = OCR_DIR.parent.parent


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = (
    OCR_DIR
    / "results"
    / "plate_detector"
    / "weights"
    / "best.pt"
)


# =========================================================
# TEST IMAGE
# =========================================================

TEST_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "plate"
    / "train"
    / "images"
    / "image_0009.jpg"
)


# =========================================================
# RESULTS
# =========================================================

RESULTS_DIR = (
    OCR_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# OCR MODULES
# =========================================================

if str(OCR_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(OCR_DIR)
    )


from plate_crop import crop_plate

from ocr_reader import OCRReader

from plate_text_cleaner import (
    clean_plate_text
)


# =========================================================
# ANPR PIPELINE
# =========================================================

def run_anpr():

    print("=" * 60)
    print("        INTELLIVISION ANPR PIPELINE")
    print("=" * 60)


    # =====================================================
    # 1. CHECK MODEL
    # =====================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Plate model not found:\n"
            f"{MODEL_PATH}"
        )


    # =====================================================
    # 2. CHECK TEST IMAGE
    # =====================================================

    if not TEST_IMAGE.exists():

        raise FileNotFoundError(
            f"Test image not found:\n"
            f"{TEST_IMAGE}"
        )


    # =====================================================
    # 3. LOAD PLATE DETECTOR
    # =====================================================

    print(
        "\n[1] Loading plate detector..."
    )

    detector = YOLO(
        str(MODEL_PATH)
    )

    print(
        "✅ Plate detector loaded"
    )


    # =====================================================
    # 4. LOAD OCR
    # =====================================================

    print(
        "\n[2] Loading OCR..."
    )

    ocr_reader = OCRReader()

    print(
        "✅ OCR loaded"
    )


    # =====================================================
    # 5. LOAD IMAGE
    # =====================================================

    print(
        "\n[3] Loading image..."
    )

    image = cv2.imread(
        str(TEST_IMAGE)
    )

    if image is None:

        raise ValueError(
            "Could not load test image."
        )


    print(
        "Image:",
        TEST_IMAGE
    )

    print(
        "Shape:",
        image.shape
    )


    # =====================================================
    # 6. PLATE DETECTION
    # =====================================================

    print(
        "\n[4] Detecting license plates..."
    )

    results = detector.predict(
        source=image,

        # Final threshold
        conf=0.25,

        # Avoid too many false detections
        max_det=3,

        device="cpu",

        verbose=False
    )


    # =====================================================
    # OUTPUT IMAGE
    # =====================================================

    output = image.copy()

    detected = 0

    recognized = 0


    # =====================================================
    # 7. PROCESS DETECTED PLATES
    # =====================================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            detected += 1


            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            bbox = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )


            # -------------------------------------------------
            # Detection confidence
            # -------------------------------------------------

            confidence = float(
                box.conf[0]
                .cpu()
                .item()
            )


            x1, y1, x2, y2 = bbox


            print(
                f"\nPlate {detected}"
            )

            print(
                "Detection confidence:",
                f"{confidence:.2f}"
            )


            # =================================================
            # 8. CROP PLATE
            # =================================================

            plate = crop_plate(
                image,
                bbox
            )


            if plate is None:

                print(
                    "⚠️ Invalid plate crop"
                )

                continue


            if plate.size == 0:

                print(
                    "⚠️ Empty plate crop"
                )

                continue


            print(
                "✅ Plate crop created"
            )


            # =================================================
            # SAVE PLATE CROP
            # =================================================

            crop_path = (
                RESULTS_DIR
                / f"plate_crop_{detected}.jpg"
            )

            cv2.imwrite(
                str(crop_path),
                plate
            )


            # =================================================
            # 9. OCR
            # =================================================

            print(
                "Running OCR..."
            )

            ocr_result = (
                ocr_reader.read(
                    plate
                )
            )


            # -------------------------------------------------
            # Raw OCR text
            # -------------------------------------------------

            raw_text = (
                ocr_result["text"]
            )


            # -------------------------------------------------
            # Clean text
            # -------------------------------------------------

            cleaned_text = (
                clean_plate_text(
                    raw_text
                )
            )


            # -------------------------------------------------
            # OCR confidence
            # -------------------------------------------------

            ocr_confidence = float(
                ocr_result[
                    "confidence"
                ]
            )


            print(
                "Raw OCR:",
                raw_text
            )

            print(
                "Cleaned:",
                cleaned_text
            )

            print(
                "OCR confidence:",
                f"{ocr_confidence:.2f}"
            )


            if cleaned_text:

                recognized += 1


            # =================================================
            # 10. DRAW RESULT
            # =================================================

            cv2.rectangle(
                output,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            # Text to display
            if cleaned_text:

                display_text = (
                    f"{cleaned_text} "
                    f"{ocr_confidence:.2f}"
                )

            else:

                display_text = (
                    f"Plate "
                    f"{confidence:.2f}"
                )


            cv2.putText(
                output,

                display_text,

                (
                    x1,
                    max(
                        y1 - 10,
                        25
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2
            )


    # =====================================================
    # 11. SAVE FINAL ANPR RESULT
    # =====================================================

    output_path = (
        RESULTS_DIR
        / "anpr_result.jpg"
    )


    success = cv2.imwrite(
        str(output_path),
        output
    )


    if not success:

        raise RuntimeError(
            "Could not save ANPR output image."
        )


    # =====================================================
    # 12. FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 60)

    print(
        "Plates detected:",
        detected
    )

    print(
        "Plates recognized:",
        recognized
    )

    print(
        "Output saved to:"
    )

    print(
        output_path
    )


    if detected > 0 and recognized > 0:

        print(
            "\n✅ PLATE DETECTION + OCR PIPELINE COMPLETE"
        )

    elif detected > 0:

        print(
            "\n⚠️ Plate detected, but OCR "
            "did not recognize text."
        )

    else:

        print(
            "\n⚠️ No plates detected."
        )


    print(
        "=" * 60
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    run_anpr()