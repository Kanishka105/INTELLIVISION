def filter_by_confidence(
    detections,
    threshold=0.50
):
    """
    Keep detections whose confidence
    is greater than or equal to threshold.
    """

    return [
        detection
        for detection in detections
        if detection["confidence"] >= threshold
    ]