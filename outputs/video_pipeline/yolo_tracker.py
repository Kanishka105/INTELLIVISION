from pathlib import Path

from ultralytics import YOLO


class YOLOTracker:

    def __init__(self, model_path):

        model_path = Path(
            model_path
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"YOLO model not found:\n"
                f"{model_path}"
            )

        self.model = YOLO(
            str(model_path)
        )


    def track(
        self,
        frame,
        confidence=0.25
    ):

        results = self.model.track(

            source=frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=confidence,

            verbose=False
        )


        if not results:
            return []


        result = results[0]


        if result.boxes is None:
            return []


        if result.boxes.id is None:
            return []


        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )


        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        class_ids = (
            result.boxes.cls
            .cpu()
            .numpy()
        )


        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
        )


        names = result.names


        tracks = []


        for (
            box,
            conf,
            class_id,
            track_id
        ) in zip(
            boxes,
            confidences,
            class_ids,
            track_ids
        ):

            class_id = int(
                class_id
            )

            track_id = int(
                track_id
            )


            tracks.append(
                {
                    "track_id":
                        track_id,

                    "bbox":
                        box.tolist(),

                    "confidence":
                        float(conf),

                    "class_id":
                        class_id,

                    "class_name":
                        names[class_id]
                }
            )


        return tracks