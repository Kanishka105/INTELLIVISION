from pathlib import Path
import sys
import cv2


CURRENT_FILE = Path(__file__).resolve()

IMAGE_PROCESSING_DIR = CURRENT_FILE.parent


for folder in [
    IMAGE_PROCESSING_DIR / "color",
    IMAGE_PROCESSING_DIR / "restoration",
    IMAGE_PROCESSING_DIR / "compression"
]:
    sys.path.insert(
        0,
        str(folder)
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


PROJECT_DIR = (
    IMAGE_PROCESSING_DIR.parent.parent
)

TEST_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "test"
    / "images"
    / "bus_104_jpg.rf.9e59f1c2a1c4681e72fdcad4a754fed1.jpg"
)


def test_image_loading():

    image = cv2.imread(
        str(TEST_IMAGE)
    )

    assert image is not None

    print("✅ Image Loading")

    return image


def test_restoration(image):

    result = restore_image(
        image,
        method="bilateral"
    )

    assert result is not None

    assert result.shape == image.shape

    print(
        "✅ Image Restoration"
    )

    return result


def test_color_processing(image):

    result = process_color(
        image,
        brightness=10,
        saturation=1.1
    )

    assert result is not None

    assert result.shape == image.shape

    print(
        "✅ Color Processing"
    )

    return result


def test_compression(image):

    output_path = (
        IMAGE_PROCESSING_DIR
        / "results"
        / "test_compressed.jpg"
    )

    output_path.parent.mkdir(
        exist_ok=True
    )

    compress_image(
        image,
        output_path,
        quality=80
    )

    assert output_path.exists()

    print(
        "✅ Image Compression"
    )


def run_all_tests():

    print("\n================================")
    print(" IMAGE PROCESSING TEST START")
    print("================================\n")

    image = test_image_loading()

    restored = test_restoration(
        image
    )

    color = test_color_processing(
        restored
    )

    test_compression(
        color
    )

    print("\n================================")
    print("✅ IMAGE PROCESSING TEST PASSED")
    print("================================\n")


if __name__ == "__main__":
    run_all_tests()