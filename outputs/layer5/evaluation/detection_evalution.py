from ultralytics import YOLO


def evaluate_model(
    model_path,
    data_yaml,
    split="val"
):
    """
    Evaluate the trained YOLO model.
    """

    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        split=split
    )

    return metrics