from pathlib import Path
import sys
import cv2


# =================================================
# PATHS
# =================================================

CURRENT_FILE = Path(__file__).resolve()

IMAGE_PROCESSING_DIR = CURRENT_FILE.parent

PROJECT_DIR = (
    IMAGE_PROCESSING_DIR.parent.parent
)

RESULTS_DIR = (
    IMAGE_PROCESSING_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


# =================================================
# IMPORT MODULES
# =================================================

COLOR_DIR = (
    IMAGE_PROCESSING_DIR / "color"
)

RESTORATION_DIR = (
    IMAGE_PROCESSING_DIR / "restoration"
)

COMPRESSION_DIR = (
    IMAGE_PROCESSING_DIR / "compression"
)


for directory in [
    COLOR_DIR,
    RESTORATION_DIR,
    COMPRESSION_DIR
]:

    if str(directory) not in sys.path:
        sys.path.insert(
            0,
            str(directory)
        )


from color_processor import (
    process_color
)

from restoration import (
    restore_image
)

from image_compression import (
    compress_image
)

from compression_metrics import (
    file_size_kb,
    compression_ratio,
    size_reduction_percent
)


# =================================================
# INPUT
# =================================================

INPUT_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "test"
    / "images"
    / "bus_104_jpg.rf.9e59f1c2a1c4681e72fdcad4a754fed1.jpg"
)


# =================================================
# MAIN PIPELINE
# =================================================

def run_image_processing():

    print("\n======================================")
    print("     IMAGE PROCESSING PIPELINE")
    print("======================================")


    # ---------------------------------------------
    # Load image
    # ---------------------------------------------

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{INPUT_IMAGE}"
        )

    print("\n✅ Input image loaded")


    # ---------------------------------------------
    # STEP 1 — RESTORATION
    # ---------------------------------------------

    print(
        "\n[1] Image Restoration..."
    )

    restored = restore_image(
        image,
        method="bilateral"
    )

    print(
        "✅ Restoration complete"
    )


    # ---------------------------------------------
    # STEP 2 — COLOR PROCESSING
    # ---------------------------------------------

    print(
        "\n[2] Color Processing..."
    )

    color_processed = process_color(
        restored,
        brightness=10,
        saturation=1.1,
        hue_shift=0
    )

    print(
        "✅ Color processing complete"
    )


    # ---------------------------------------------
    # Save uncompressed processed image
    # ---------------------------------------------

    processed_path = (
        RESULTS_DIR
        / "processed_image.png"
    )

    cv2.imwrite(
        str(processed_path),
        color_processed
    )


    # ---------------------------------------------
    # STEP 3 — COMPRESSION
    # ---------------------------------------------

    print(
        "\n[3] Compression..."
    )

    compressed_path = (
        RESULTS_DIR
        / "compressed_image.jpg"
    )

    compress_image(
        color_processed,
        compressed_path,
        quality=80
    )

    print(
        "✅ Compression complete"
    )


    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    original_size = file_size_kb(
        INPUT_IMAGE
    )

    compressed_size = file_size_kb(
        compressed_path
    )

    ratio = compression_ratio(
        INPUT_IMAGE,
        compressed_path
    )

    reduction = size_reduction_percent(
        INPUT_IMAGE,
        compressed_path
    )


    print("\n======================================")
    print("          COMPRESSION RESULTS")
    print("======================================")

    print(
        f"Original size     : "
        f"{original_size:.2f} KB"
    )

    print(
        f"Compressed size   : "
        f"{compressed_size:.2f} KB"
    )

    print(
        f"Compression ratio : "
        f"{ratio:.2f}"
    )

    print(
        f"Size reduction    : "
        f"{reduction:.2f}%"
    )


    print("\n======================================")
    print("✅ IMAGE PROCESSING COMPLETE")
    print("======================================")

    print(
        f"\nOutput folder:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    run_image_processing()