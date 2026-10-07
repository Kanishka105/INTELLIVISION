import cv2
from pathlib import Path
from collections import Counter


def analyze_dataset(image_dir):

    image_dir = Path(image_dir)

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    image_paths = [
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in valid_extensions
    ]

    print("\n========================================")
    print("       DATASET CHARACTERISTICS")
    print("========================================")

    print(f"\nTotal images: {len(image_paths)}")

    if len(image_paths) == 0:
        print("No images found!")
        return

    # Store characteristics
    resolutions = Counter()
    channels = Counter()
    data_types = Counter()

    min_values = []
    max_values = []
    mean_values = []

    failed_images = []

    # Analyze every image
    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            failed_images.append(str(image_path))
            continue

        # -----------------------------
        # Resolution
        # -----------------------------

        height, width = image.shape[:2]

        resolutions[(width, height)] += 1

        # -----------------------------
        # Channels
        # -----------------------------

        if len(image.shape) == 2:
            channel_count = 1
        else:
            channel_count = image.shape[2]

        channels[channel_count] += 1

        # -----------------------------
        # Data type
        # -----------------------------

        data_types[str(image.dtype)] += 1

        # -----------------------------
        # Pixel statistics
        # -----------------------------

        min_values.append(image.min())
        max_values.append(image.max())
        mean_values.append(image.mean())

    # =================================================
    # RESULTS
    # =================================================

    print("\n----------------------------------------")
    print("RESOLUTION")
    print("----------------------------------------")

    for resolution, count in resolutions.items():

        print(
            f"{resolution[0]} x {resolution[1]}"
            f"  →  {count} images"
        )

    print("\n----------------------------------------")
    print("CHANNELS")
    print("----------------------------------------")

    for channel, count in channels.items():

        print(
            f"{channel} channel(s)"
            f"  →  {count} images"
        )

    print("\n----------------------------------------")
    print("DATA TYPE")
    print("----------------------------------------")

    for dtype, count in data_types.items():

        print(
            f"{dtype}"
            f"  →  {count} images"
        )

    print("\n----------------------------------------")
    print("PIXEL STATISTICS")
    print("----------------------------------------")

    print(
        f"Minimum pixel value across dataset : "
        f"{min(min_values)}"
    )

    print(
        f"Maximum pixel value across dataset : "
        f"{max(max_values)}"
    )

    print(
        f"Average mean pixel value            : "
        f"{sum(mean_values) / len(mean_values):.2f}"
    )

    print("\n----------------------------------------")
    print("IMAGE VALIDATION")
    print("----------------------------------------")

    print(
        f"Successfully read : "
        f"{len(image_paths) - len(failed_images)}"
    )

    print(
        f"Failed to read    : "
        f"{len(failed_images)}"
    )

    print("\n========================================")

    # =================================================
    # CONSISTENCY CHECK
    # =================================================

    print("\n        CONSISTENCY CHECK")
    print("========================================")

    if len(resolutions) == 1:
        print("✓ All images have the same resolution.")
    else:
        print("⚠ Images have different resolutions.")

    if len(channels) == 1:
        print("✓ All images have the same number of channels.")
    else:
        print("⚠ Images have different channel counts.")

    if len(data_types) == 1:
        print("✓ All images have the same data type.")
    else:
        print("⚠ Images have different data types.")

    if len(failed_images) == 0:
        print("✓ All images were successfully loaded.")
    else:
        print("⚠ Some images could not be loaded.")

    print("========================================")