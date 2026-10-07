from pathlib import Path
from ultralytics import YOLO


def load_tracker(model_path):
    """
    Load the trained YOLO model used for tracking.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{model_path}"
        )

    return YOLO(str(model_path))


def track_frame(
    model,
    frame,
    confidence=0.25,
    iou_threshold=0.45
):
    """
    Track objects in one video frame using ByteTrack.
    """

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=confidence,
        iou=iou_threshold,
        verbose=False
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()

    confidences = (
        result.boxes.conf.cpu().numpy()
    )

    class_ids = (
        result.boxes.cls.cpu().numpy()
    )

    track_ids = result.boxes.id

    if track_ids is None:
        return []

    track_ids = track_ids.cpu().numpy()

    names = result.names

    tracks = []

    for box, confidence, class_id, track_id in zip(
        boxes,
        confidences,
        class_ids,
        track_ids
    ):

        class_id = int(class_id)
        track_id = int(track_id)

        tracks.append(
            {
                "track_id": track_id,

                "bbox": box.tolist(),

                "confidence":
                    float(confidence),

                "class_id":
                    class_id,

                "class_name":
                    names[class_id]
            }
        )

    return tracks