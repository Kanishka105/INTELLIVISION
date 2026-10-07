from collections import defaultdict


class TrajectoryManager:

    def __init__(
        self,
        max_history=50
    ):

        self.max_history = max_history

        self.history = defaultdict(list)


    def update(
        self,
        track_id,
        center
    ):

        points = self.history[
            track_id
        ]

        points.append(center)

        if len(points) > self.max_history:

            points.pop(0)


    def get_trajectory(
        self,
        track_id
    ):

        return self.history.get(
            track_id,
            []
        )


    def get_all_trajectories(self):

        return dict(
            self.history
        )