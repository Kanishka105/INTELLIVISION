def compare_predictions(
    yolo_class,
    yolo_confidence,
    classical_class,
    classical_confidence
):

    agreement = (
        yolo_class == classical_class
    )

    if agreement:
        status = "AGREEMENT"
    else:
        status = "DISAGREEMENT"

    return {
        "yolo_class": yolo_class,
        "yolo_confidence": float(
            yolo_confidence
        ),

        "classical_class": classical_class,
        "classical_confidence": float(
            classical_confidence
        ),

        "agreement": agreement,
        "status": status
    }