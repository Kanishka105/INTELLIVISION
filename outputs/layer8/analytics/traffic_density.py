def calculate_density(
    tracks,
    frame_width,
    frame_height,
    low_threshold=0.10,
    high_threshold=0.25
):

    frame_area = (
        frame_width
        * frame_height
    )

    if frame_area == 0:

        return {
            "vehicle_count": 0,
            "occupancy": 0.0,
            "level": "unknown"
        }


    occupied_area = 0.0


    for track in tracks:

        x1, y1, x2, y2 = (
            track["bbox"]
        )

        box_width = max(
            0,
            x2 - x1
        )

        box_height = max(
            0,
            y2 - y1
        )

        occupied_area += (
            box_width * box_height
        )


    occupancy = (
        occupied_area
        / frame_area
    )


    if occupancy < low_threshold:

        level = "Low"

    elif occupancy < high_threshold:

        level = "Medium"

    else:

        level = "High"


    return {
        "vehicle_count": len(tracks),

        "occupancy": float(
            occupancy
        ),

        "level": level
    }