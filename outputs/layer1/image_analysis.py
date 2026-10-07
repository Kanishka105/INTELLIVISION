import cv2


def analyze_image(image):
    height, width = image.shape[:2]

    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]

    print("\n========== IMAGE ANALYSIS ==========")

    print(f"Shape        : {image.shape}")
    print(f"Width        : {width} pixels")
    print(f"Height       : {height} pixels")
    print(f"Resolution   : {width} x {height}")
    print(f"Channels     : {channels}")
    print(f"Data type    : {image.dtype}")
    print(f"Min pixel    : {image.min()}")
    print(f"Max pixel    : {image.max()}")
    # Give idea of overall brightness of the image
    print(f"Mean pixel   : {image.mean():.2f}")
    print(f"Total pixels : {height * width}")

    print("====================================")


def analyze_channels(image):

    if len(image.shape) != 3:
        print("Image is grayscale.")
        return

    blue, green, red = cv2.split(image)

    print("\n========== CHANNEL ANALYSIS ==========")

    print(f"Blue mean  : {blue.mean():.2f}")
    print(f"Green mean : {green.mean():.2f}")
    print(f"Red mean   : {red.mean():.2f}")

    print("======================================")


def analyze_color_spaces(image):
    """
    Convert image into different color spaces.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    print("\n========== COLOR SPACES ==========")

    print(f"BGR       : {image.shape}")
    print(f"RGB       : {rgb.shape}")
    print(f"Grayscale : {gray.shape}")
    print(f"HSV       : {hsv.shape}")

    print("==================================")

    return rgb, gray, hsv