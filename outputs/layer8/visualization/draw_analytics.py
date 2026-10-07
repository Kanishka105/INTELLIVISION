import cv2


def draw_analytics(
    frame,
    tracks,
    summary
):

    output = frame.copy()


    # -----------------------------------------
    # Draw tracked vehicles
    # -----------------------------------------

    for track in tracks:

        x1, y1, x2, y2 = map(
            int,
            track["bbox"]
        )

        track_id = track[
            "track_id"
        ]

        class_name = track[
            "class_name"
        ]

        confidence = track[
            "confidence"
        ]

        speed = track.get(
            "speed_px_s",
            0.0
        )


        label = (
            f"ID {track_id} | "
            f"{class_name} | "
            f"{confidence:.2f}"
        )


        if speed > 0:

            label += (
                f" | {speed:.1f}px/s"
            )


        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        center = track.get(
            "center"
        )

        if center:

            cv2.circle(
                output,
                center,
                4,
                (0, 0, 255),
                -1
            )


    # -----------------------------------------
    # Analytics panel
    # -----------------------------------------

    class_counts = summary[
        "vehicle_classes"
    ]

    density = summary[
        "traffic_density"
    ]

    crossings = summary[
        "line_crossings"
    ]


    y = 30

    lines = [

        f"Unique Vehicles: "
        f"{summary['total_unique_vehicles']}",

        f"Cars: "
        f"{class_counts.get('car', 0)}",

        f"Buses: "
        f"{class_counts.get('bus', 0)}",

        f"Trucks: "
        f"{class_counts.get('truck', 0)}",

        f"Motorcycles: "
        f"{class_counts.get('motorcycle', 0)}",

        f"Crossings: "
        f"{crossings['total']}",

        f"Density: "
        f"{density['level']}"
    ]


    for line in lines:

        cv2.putText(
            output,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y += 28


    return output