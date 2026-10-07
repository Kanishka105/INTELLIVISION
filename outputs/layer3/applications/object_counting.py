import cv2


def count_objects(
    binary_image,
    min_area=100
):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_image,
            connectivity=8
        )
    )

    count = 0

    for label in range(
        1,
        num_labels
    ):

        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            count += 1

    return count