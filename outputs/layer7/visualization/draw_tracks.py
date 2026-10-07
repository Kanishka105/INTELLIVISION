import cv2


def draw_tracks(
    frame,
    tracks,
    trajectories
):

    output = frame.copy()

    for track in tracks:

        track_id = track[
            "track_id"
        ]

        x1, y1, x2, y2 = map(
            int,
            track["bbox"]
        )

        class_name = track[
            "class_name"
        ]

        confidence = track[
            "confidence"
        ]

        center = track.get(
            "center"
        )


        # -----------------------------
        # Bounding box
        # -----------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # -----------------------------
        # Label
        # -----------------------------

        label = (
            f"ID {track_id} | "
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


        # -----------------------------
        # Center point
        # -----------------------------

        if center is not None:

            cv2.circle(
                output,
                center,
                4,
                (0, 0, 255),
                -1
            )


        # -----------------------------
        # Trajectory
        # -----------------------------

        points = trajectories.get(
            track_id,
            []
        )

        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                output,
                points[i - 1],
                points[i],
                (255, 0, 0),
                2
            )

    return output