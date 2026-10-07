from ultralytics import YOLO


def detect_objects(
    image,
    model,
    confidence=0.25,
    iou_threshold=0.45
):
    """
    Detect objects in an image.

    Returns a list of detections containing:
    - bounding box
    - confidence
    - class id
    - class name
    """

    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou_threshold,
        verbose=False
    )

    detections = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    names = result.names

    for box, conf, class_id in zip(
        boxes,
        confidences,
        class_ids
    ):

        class_id = int(class_id)

        detections.append(
            {
                "bbox": box.tolist(),
                "confidence": float(conf),
                "class_id": class_id,
                "class_name": names[class_id]
            }
        )

    return detections