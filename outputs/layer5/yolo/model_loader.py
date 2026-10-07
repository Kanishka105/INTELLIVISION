from ultralytics import YOLO


def load_model(model_path="yolo11n.pt"):
    """
    Load a YOLO model.
    """

    model = YOLO(model_path)

    return model