import cv2


class VideoWriter:

    def __init__(
        self,
        output_path,
        width,
        height,
        fps
    ):

        self.output_path = str(
            output_path
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                "Could not create output video."
            )

    def write(self, frame):

        self.writer.write(frame)

    def release(self):

        self.writer.release()