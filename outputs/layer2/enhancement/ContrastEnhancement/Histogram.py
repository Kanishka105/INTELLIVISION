import cv2
import matplotlib.pyplot as plt


def show_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(8, 5))

    plt.hist(
        gray.ravel(),
        bins=256,
        range=[0, 256]
    )

    plt.title("Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")

    plt.show()