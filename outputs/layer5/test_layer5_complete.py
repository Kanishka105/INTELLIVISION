import os
import cv2
from pathlib import Path
from yolo.model_loader import load_model
from yolo.detector import detect_objects

from postprocessing.confidence import (
    filter_by_confidence
)

from results.detection_results import (
    summarize_detections
)

from visualization.draw_detections import (
    draw_detections
)




BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "results"
    / "vehicle_detector"
    / "weights"
    / "best.pt"
)


IMAGE_PATH = r"C:\Users\well\Downloads\ocr\INTELLIVISION\dataset\test\images\bus_104_jpg.rf.9e59f1c2a1c4681e72fdcad4a754fed1.jpg"


def test_image_loading():

    image = cv2.imread(IMAGE_PATH)

    assert image is not None

    print("✅ Image Loading")


def test_model_loading():

    model = load_model(MODEL_PATH)

    assert model is not None

    print("✅ YOLO Model Loading")


def test_detection():

    image = cv2.imread(IMAGE_PATH)

    model = load_model(MODEL_PATH)

    detections = detect_objects(
        image,
        model
    )

    assert isinstance(
        detections,
        list
    )

    print("✅ Object Detection")


def test_confidence_filter():

    detections = [
        {
            "class_name": "car",
            "confidence": 0.90
        },
        {
            "class_name": "bus",
            "confidence": 0.30
        }
    ]

    filtered = filter_by_confidence(
        detections,
        threshold=0.50
    )

    assert len(filtered) == 1

    print("✅ Confidence Filtering")


def test_summary():

    detections = [
        {"class_name": "car"},
        {"class_name": "car"},
        {"class_name": "bus"}
    ]

    summary = summarize_detections(
        detections
    )

    assert summary["total_objects"] == 3
    assert summary["class_counts"]["car"] == 2

    print("✅ Detection Summary")


def test_visualization():

    image = cv2.imread(IMAGE_PATH)

    model = load_model(MODEL_PATH)

    detections = detect_objects(
        image,
        model
    )

    result = draw_detections(
        image,
        detections
    )

    assert result is not None

    print("✅ Detection Visualization")


def run_all_tests():

    print("\n==============================")
    print("     LAYER 5 TEST START")
    print("==============================\n")

    test_image_loading()
    test_model_loading()
    test_detection()
    test_confidence_filter()
    test_summary()
    test_visualization()

    print("\n==============================")
    print("✅ LAYER 5 COMPLETE")
    print("==============================\n")


if __name__ == "__main__":
    run_all_tests()