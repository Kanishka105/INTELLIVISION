import cv2


def label_components(binary_image):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_image,
            connectivity=8
        )
    )

    return (
        num_labels,
        labels,
        stats,
        centroids
    )