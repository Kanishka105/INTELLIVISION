import cv2
from pathlib import Path


def load_image(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


def convert_bgr_to_rgb(image):
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )