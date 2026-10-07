from pathlib import Path
import sys
import cv2


# =================================================
# PATHS
# =================================================

CURRENT_FILE = Path(__file__).resolve()

LAYER7_DIR = CURRENT_FILE.parent

OUTPUTS_DIR = LAYER7_DIR.parent

PROJECT_DIR = OUTPUTS_DIR.parent


# =================================================
# LAYER 5
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
# Import Layer 7
# =================================================

TRACKING_DIR = (
    LAYER7_DIR / "tracking"
)

MOVEMENT_DIR = (
    LAYER7_DIR / "movement"
)

VISUALIZATION_DIR = (
    LAYER7_DIR / "visualization"
)


for directory in [
    TRACKING_DIR,
    MOVEMENT_DIR,
    VISUALIZATION_DIR
]:

    if str(directory) not in sys.path:

        sys.path.insert(
            0,
            str(directory)
        )


from tracker import load_tracker, track_frame
from track_manager import TrackManager

from direction import (
    calculate_direction
)

from displacement import (
    calculate_displacement
)

from draw_tracks import (
    draw_tracks
)


# =================================================
# VIDEO
# =================================================

INPUT_VIDEO = (
    PROJECT_DIR
    / "dataset"
    / "video"
    / "traffic.mp4"
)

OUTPUT_VIDEO = (
    LAYER7_DIR
    / "results"
    / "tracking_output.mp4"
)


# =================================================
# MAIN
# =================================================

def run_tracking():

    OUTPUT_VIDEO.parent.mkdir(
        exist_ok=True
    )


    if not INPUT_VIDEO.exists():

        raise FileNotFoundError(
            f"Video not found:\n{INPUT_VIDEO}"
        )


    # ---------------------------------------------
    # Open video
    # ---------------------------------------------

    capture = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not capture.isOpened():

        raise RuntimeError(
            "Could not open video."
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
        fps = 30


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
    # Load YOLO
    # ---------------------------------------------

    print("Loading YOLO...")

    model = load_tracker(
        MODEL_PATH
    )

    print("✅ YOLO loaded")


    # ---------------------------------------------
    # Track manager
    # ---------------------------------------------

    track_manager = TrackManager(
        max_history=50
    )


    previous_centers = {}


    # ---------------------------------------------
    # Process video
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
        # ByteTrack
        # -----------------------------------------

        tracks = track_frame(
            model,
            frame,
            confidence=0.25,
            iou_threshold=0.45
        )


        # -----------------------------------------
        # Update track manager
        # -----------------------------------------

        track_manager.update(
            tracks
        )


        # -----------------------------------------
        # Movement analysis
        # -----------------------------------------

        for track in tracks:

            track_id = track[
                "track_id"
            ]

            current_center = track[
                "center"
            ]

            previous_center = (
                previous_centers.get(
                    track_id
                )
            )


            direction = calculate_direction(
                previous_center,
                current_center
            )


            displacement = (
                calculate_displacement(
                    previous_center,
                    current_center
                )
            )


            track["direction"] = direction

            track["displacement"] = (
                displacement
            )


            previous_centers[
                track_id
            ] = current_center


        # -----------------------------------------
        # Get trajectories
        # -----------------------------------------

        trajectories = (
            track_manager
            .trajectory_manager
            .get_all_trajectories()
        )


        # -----------------------------------------
        # Draw
        # -----------------------------------------

        output_frame = draw_tracks(
            frame,
            tracks,
            trajectories
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
            "INTELLIVISION - Layer 7 Tracking",
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


    print("\n================================")
    print("✅ LAYER 7 TRACKING COMPLETE")
    print("================================")

    print(
        f"Output:\n{OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    run_tracking()