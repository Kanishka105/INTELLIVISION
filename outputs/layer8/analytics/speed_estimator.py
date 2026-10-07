import math


class SpeedEstimator:

    def __init__(self):

        self.previous_centers = {}

        self.speed_history = {}


    def update(
        self,
        track_id,
        current_center,
        fps,
        meters_per_pixel=None
    ):

        previous_center = (
            self.previous_centers.get(
                track_id
            )
        )


        if previous_center is None:

            pixel_speed = 0.0

        else:

            x1, y1 = previous_center

            x2, y2 = current_center

            distance = math.sqrt(
                (x2 - x1) ** 2
                +
                (y2 - y1) ** 2
            )

            pixel_speed = (
                distance * fps
            )


        self.previous_centers[
            track_id
        ] = current_center


        # Store history
        self.speed_history.setdefault(
            track_id,
            []
        ).append(
            pixel_speed
        )


        # Real-world speed only when calibrated
        speed_kmh = None

        if meters_per_pixel is not None:

            speed_ms = (
                pixel_speed
                * meters_per_pixel
            )

            speed_kmh = (
                speed_ms * 3.6
            )


        return pixel_speed, speed_kmh


    def average_pixel_speed(self):

        values = []

        for speeds in self.speed_history.values():

            values.extend(speeds)


        if not values:

            return 0.0


        return sum(values) / len(values)