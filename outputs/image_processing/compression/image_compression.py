import cv2


def compress_image(
    image,
    output_path,
    quality=80
):
    """
    Save an image as JPEG with a selected quality.
    """

    if image is None:
        raise ValueError("Image is None.")

    success = cv2.imwrite(
        str(output_path),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]
    )

    if not success:
        raise RuntimeError(
            "Could not save compressed image."
        )

    return output_path