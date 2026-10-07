from pathlib import Path
import sys
import cv2


# =================================================
# PATHS
# =================================================

CURRENT_FILE = Path(__file__).resolve()

LAYER8_DIR = CURRENT_FILE.parent

OUTPUTS_DIR = LAYER8_DIR.parent

PROJECT_DIR = OUTPUTS_DIR.parent


# =================================================
# LAYER 5 MODEL
# =================================================

LAYER5_DIR = (
    OUTPUTS_DIR / "layer5"
)

MODEL_PATH = (
    LAYER5_DIR
    / "results"
    / "vehicle_detector"
    / "weights"
    / "best.pt"
)


# =================================================
# LAYER 7 TRACKER
# =================================================

LAYER7_DIR = (
    OUTPUTS_DIR / "layer7"
)

TRACKING_DIR = (
    LAYER7_DIR / "tracking"
)

if str(TRACKING_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(TRACKING_DIR)
    )


from tracker import (
    load_tracker,
    track_frame
)


# =================================================
# LAYER 8 IMPORTS
# =================================================

from analytics.vehicle_counter import (
    VehicleCounter
)

from analytics.line_crossing import (
    LineCrossingCounter
)

from analytics.speed_estimator import (
    SpeedEstimator
)

from analytics.traffic_density import (
    calculate_density
)

from results.analytics_results import (
    build_summary
)

from visualization.draw_analytics import (
    draw_analytics
)


# =================================================
# INPUT / OUTPUT
# =================================================

INPUT_VIDEO = (
    PROJECT_DIR
    / "dataset"
    / "video"
    / "traffic.mp4"
)

OUTPUT_VIDEO = (
    LAYER8_DIR
    / "results"
    / "analytics_output.mp4"
)


# =================================================
# MAIN
# =================================================

def run_layer8():

    OUTPUT_VIDEO.parent.mkdir(
        exist_ok=True
    )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"YOLO model not found:\n"
            f"{MODEL_PATH}"
        )


    if not INPUT_VIDEO.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{INPUT_VIDEO}"
        )


    # ---------------------------------------------
    # Open video
    # ---------------------------------------------

    capture = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )


    if not capture.isOpened():

        raise RuntimeError(
            "Could not open input video."
        )


    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 30.0


    # ---------------------------------------------
    # Output writer
    # ---------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )


    # ---------------------------------------------
    # Load YOLO tracker
    # ---------------------------------------------

    print("Loading Layer 5 + Layer 7...")

    model = load_tracker(
        MODEL_PATH
    )

    print("✅ Tracker loaded")


    # ---------------------------------------------
    # Analytics objects
    # ---------------------------------------------

    vehicle_counter = VehicleCounter()


    # Horizontal line at 60% of frame
    line_start = (
        0,
        int(height * 0.60)
    )

    line_end = (
        width,
        int(height * 0.60)
    )


    crossing_counter = (
        LineCrossingCounter(
            line_start,
            line_end
        )
    )


    speed_estimator = (
        SpeedEstimator()
    )


    # ---------------------------------------------
    # Process frames
    # ---------------------------------------------

    frame_number = 0


    while True:

        success, frame = (
            capture.read()
        )


        if not success:

            break


        frame_number += 1


        # -----------------------------------------
        # Layer 7
        # -----------------------------------------

        tracks = track_frame(
            model,
            frame,
            confidence=0.25,
            iou_threshold=0.45
        )


        # -----------------------------------------
        # Layer 8 analytics
        # -----------------------------------------

        vehicle_counter.update(
            tracks
        )


        for track in tracks:

            x1, y1, x2, y2 = (
                track["bbox"]
            )


            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )


            track["center"] = center


            track_id = track[
                "track_id"
            ]


            # Line crossing
            crossed, crossing_direction = (
                crossing_counter.update(
                    track_id,
                    center
                )
            )


            track["crossed"] = crossed

            track["crossing_direction"] = (
                crossing_direction
            )


            # Speed
            pixel_speed, speed_kmh = (
                speed_estimator.update(
                    track_id,
                    center,
                    fps
                )
            )


            track["speed_px_s"] = (
                pixel_speed
            )

            track["speed_kmh"] = (
                speed_kmh
            )


        # -----------------------------------------
        # Density
        # -----------------------------------------

        density = calculate_density(
            tracks,
            width,
            height
        )


        # -----------------------------------------
        # Summary
        # -----------------------------------------

        summary = build_summary(
            vehicle_counter,
            crossing_counter,
            speed_estimator,
            density
        )


        # -----------------------------------------
        # Draw analytics
        # -----------------------------------------

        output_frame = draw_analytics(
            frame,
            tracks,
            summary
        )


        # Draw counting line
        cv2.line(
            output_frame,
            line_start,
            line_end,
            (255, 0, 0),
            2
        )


        # -----------------------------------------
        # Write
        # -----------------------------------------

        writer.write(
            output_frame
        )


        # -----------------------------------------
        # Display
        # -----------------------------------------

        cv2.imshow(
            "INTELLIVISION - Layer 8",
            output_frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


    # ---------------------------------------------
    # Cleanup
    # ---------------------------------------------

    capture.release()

    writer.release()

    cv2.destroyAllWindows()


    print("\n====================================")
    print("✅ LAYER 8 COMPLETE")
    print("====================================")

    print("\nFinal Summary:")

    print(summary)

    print(
        f"\nOutput video:\n"
        f"{OUTPUT_VIDEO}"
    )


if __name__ == "__main__":

    run_layer8()