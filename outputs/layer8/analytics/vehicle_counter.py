from collections import Counter


class VehicleCounter:

    def __init__(self):

        self.seen_ids = set()

        self.class_counts = Counter()


    def update(self, tracks):

        for track in tracks:

            track_id = track["track_id"]

            class_name = track["class_name"]

            # Count vehicle only once
            if track_id not in self.seen_ids:

                self.seen_ids.add(track_id)

                self.class_counts[
                    class_name
                ] += 1


    def total_unique(self):

        return len(self.seen_ids)


    def get_class_counts(self):

        return dict(self.class_counts)


    def current_count(self, tracks):

        return len(tracks)
