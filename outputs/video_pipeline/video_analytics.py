from collections import Counter


class VideoAnalytics:

    def __init__(self):

        self.seen_ids = set()

        self.class_counts = Counter()

        self.track_history = {}


    def update(
        self,
        tracks
    ):

        for track in tracks:

            track_id = (
                track["track_id"]
            )

            class_name = (
                track["class_name"]
            )


            # ---------------------------------
            # Unique vehicle
            # ---------------------------------

            if track_id not in self.seen_ids:

                self.seen_ids.add(
                    track_id
                )

                self.class_counts[
                    class_name
                ] += 1


            # ---------------------------------
            # Center
            # ---------------------------------

            x1, y1, x2, y2 = (
                track["bbox"]
            )

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            track["center"] = center


            # ---------------------------------
            # Track history
            # ---------------------------------

            history = (
                self.track_history
                .setdefault(
                    track_id,
                    []
                )
            )

            history.append(
                center
            )


            if len(history) > 30:

                history.pop(0)


    def summary(self):

        return {

            "unique_vehicles":
                len(self.seen_ids),

            "class_counts":
                dict(self.class_counts),

            "active_tracks":
                len(self.track_history)
        }


    def get_history(self):

        return self.track_history