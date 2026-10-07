from .trajectory import TrajectoryManager


class TrackManager:

    def __init__(
        self,
        max_history=50
    ):

        self.trajectory_manager = (
            TrajectoryManager(
                max_history=max_history
            )
        )

        self.objects = {}


    def calculate_center(
        self,
        bbox
    ):

        x1, y1, x2, y2 = bbox

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        return center_x, center_y


    def update(
        self,
        tracks
    ):

        for track in tracks:

            track_id = track["track_id"]

            center = self.calculate_center(
                track["bbox"]
            )

            track["center"] = center

            self.objects[track_id] = track

            self.trajectory_manager.update(
                track_id,
                center
            )


    def get_object(
        self,
        track_id
    ):

        return self.objects.get(
            track_id
        )


    def get_trajectory(
        self,
        track_id
    ):

        return (
            self.trajectory_manager
            .get_trajectory(track_id)
        )


    def get_all_objects(self):

        return self.objects