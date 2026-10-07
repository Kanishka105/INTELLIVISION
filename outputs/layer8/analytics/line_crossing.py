class LineCrossingCounter:

    def __init__(
        self,
        line_start,
        line_end
    ):

        self.line_start = line_start
        self.line_end = line_end

        self.previous_sides = {}

        self.total_crossings = 0

        self.direction_counts = {
            "A_to_B": 0,
            "B_to_A": 0
        }


    def point_side(self, point):

        x, y = point

        x1, y1 = self.line_start
        x2, y2 = self.line_end

        value = (
            (x2 - x1) * (y - y1)
            -
            (y2 - y1) * (x - x1)
        )

        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0


    def update(
        self,
        track_id,
        current_center
    ):

        current_side = self.point_side(
            current_center
        )

        previous_side = self.previous_sides.get(
            track_id
        )

        crossed = False
        direction = None

        if (
            previous_side is not None
            and previous_side != 0
            and current_side != 0
            and previous_side != current_side
        ):

            crossed = True

            self.total_crossings += 1

            if previous_side > 0 and current_side < 0:

                direction = "A_to_B"

                self.direction_counts[
                    "A_to_B"
                ] += 1

            elif previous_side < 0 and current_side > 0:

                direction = "B_to_A"

                self.direction_counts[
                    "B_to_A"
                ] += 1


        self.previous_sides[
            track_id
        ] = current_side


        return crossed, direction


    def get_counts(self):

        return {
            "total": self.total_crossings,
            "A_to_B": self.direction_counts[
                "A_to_B"
            ],
            "B_to_A": self.direction_counts[
                "B_to_A"
            ]
        }