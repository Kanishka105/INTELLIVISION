import cv2


def draw_detections(image, detections):
    """
    Draw YOLO bounding boxes, class names,
    and confidence scores on the image.
    """

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = map(
            int,
            detection["bbox"]
        )

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output