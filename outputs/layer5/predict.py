import cv2

from yolo.model_loader import load_model
from yolo.detector import detect_objects

from postprocessing.confidence import (
    filter_by_confidence
)

from visualization.draw_detections import (
    draw_detections
)


MODEL_PATH = (
    "outputs/layer5/results/"
    "vehicle_detector/weights/best.pt"
)

IMAGE_PATH = "dataset/test/images/sample.jpg"

OUTPUT_PATH = (
    "outputs/layer5/results/"
    "prediction.jpg"
)


def run_prediction():

    print("Loading image...")

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    print("Loading YOLO model...")

    model = load_model(MODEL_PATH)

    print("Running detection...")

    detections = detect_objects(
        image,
        model,
        confidence=0.25,
        iou_threshold=0.45
    )

    print(
        f"Detections before filtering: "
        f"{len(detections)}"
    )

    detections = filter_by_confidence(
        detections,
        threshold=0.50
    )

    print(
        f"Detections after filtering: "
        f"{len(detections)}"
    )

    result_image = draw_detections(
        image,
        detections
    )

    cv2.imwrite(
        OUTPUT_PATH,
        result_image
    )

    print(
        f"Result saved to: {OUTPUT_PATH}"
    )

    for detection in detections:

        print(
            f"{detection['class_name']} | "
            f"{detection['confidence']:.2f} | "
            f"{detection['bbox']}"
        )


if __name__ == "__main__":
    run_prediction()