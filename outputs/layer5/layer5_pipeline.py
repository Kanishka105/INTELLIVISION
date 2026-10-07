import cv2

from yolo.model_loader import load_model
from yolo.detector import detect_objects

from postprocessing.confidence import (
    filter_by_confidence
)

from visualization.draw_detections import (
    draw_detections
)

from results.detection_results import (
    summarize_detections
)


MODEL_PATH = (
    "outputs/layer5/results/"
    "vehicle_detector/weights/best.pt"
)


def run_layer5(
    image_path,
    output_path,
    confidence=0.50
):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_model(MODEL_PATH)

    # YOLO detection
    detections = detect_objects(
        image,
        model,
        confidence=0.25,
        iou_threshold=0.45
    )

    # Confidence filtering
    detections = filter_by_confidence(
        detections,
        threshold=confidence
    )

    # Statistics
    summary = summarize_detections(
        detections
    )

    # Draw boxes
    result_image = draw_detections(
        image,
        detections
    )

    cv2.imwrite(
        output_path,
        result_image
    )

    return detections, summary


if __name__ == "__main__":

    detections, summary = run_layer5(
        "dataset/test/images/sample.jpg",
        "outputs/layer5/results/final_prediction.jpg"
    )

    print("\nDetection Summary:")
    print(summary)