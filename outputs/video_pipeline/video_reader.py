import cv2


class VideoReader:

    def __init__(self, video_path):

        self.video_path = str(video_path)

        self.capture = cv2.VideoCapture(
            self.video_path
        )

        if not self.capture.isOpened():
            raise FileNotFoundError(
                f"Could not open video:\n"
                f"{self.video_path}"
            )

    def get_info(self):

        width = int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = self.capture.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count
        }

    def read(self):

        return self.capture.read()

    def release(self):

        self.capture.release()