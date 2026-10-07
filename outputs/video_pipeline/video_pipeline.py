from pathlib import Path
import sys
import cv2


# =================================================
# PATHS
# =================================================

CURRENT_FILE = Path(__file__).resolve()

VIDEO_DIR = CURRENT_FILE.parent

OUTPUTS_DIR = VIDEO_DIR.parent

PROJECT_DIR = OUTPUTS_DIR.parent


# =================================================
# DATASET
# =================================================

INPUT_VIDEO = (
    PROJECT_DIR
    / "dataset"
    / "video"
    / "raw"
    / "traffic_01.mp4"
)

OUTPUT_VIDEO = (
    PROJECT_DIR
    / "dataset"
    / "video"
    / "processed"
    / "traffic_output.mp4"
)


# =================================================
# LAYER 5 MODEL
# =================================================

LAYER5_DIR = (
    OUTPUTS_DIR
    / "layer5"
)

YOLO_MODEL = (
    LAYER5_DIR
    / "results"
    / "vehicle_detector"
    / "weights"
    / "best.pt"
)


# =================================================
# LOCAL IMPORTS
# =================================================

if str(VIDEO_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(VIDEO_DIR)
    )


from video_reader import (
    VideoReader
)

from video_writer import (
    VideoWriter
)

from frame_preprocessor import (
    preprocess_frame
)

from yolo_tracker import (
    YOLOTracker
)

from classical_verifier import (
    verify_track
)

from video_analytics import (
    VideoAnalytics
)


# =================================================
# MAIN
# =================================================

def run_video_pipeline():

    print("\n========================================")
    print("      INTELLIVISION VIDEO PIPELINE")
    print("========================================")


    # -------------------------------------------------
    # Check files
    # -------------------------------------------------

    if not INPUT_VIDEO.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{INPUT_VIDEO}"
        )


    if not YOLO_MODEL.exists():

        raise FileNotFoundError(
            f"YOLO model not found:\n"
            f"{YOLO_MODEL}"
        )


    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # -------------------------------------------------
    # Video Reader
    # -------------------------------------------------

    print("\n[VIDEO] Opening video...")

    reader = VideoReader(
        INPUT_VIDEO
    )

    info = reader.get_info()


    print(
        f"Resolution : "
        f"{info['width']} x {info['height']}"
    )

    print(
        f"FPS        : "
        f"{info['fps']:.2f}"
    )

    print(
        f"Frames     : "
        f"{info['frame_count']}"
    )


    # -------------------------------------------------
    # Video Writer
    # -------------------------------------------------

    writer = VideoWriter(

        OUTPUT_VIDEO,

        info["width"],

        info["height"],

        info["fps"]
    )


    # -------------------------------------------------
    # YOLO + ByteTrack
    # -------------------------------------------------

    print(
        "\n[LAYER 5 + LAYER 7]"
    )

    tracker = YOLOTracker(
        YOLO_MODEL
    )

    print(
        "✅ YOLO + ByteTrack loaded"
    )


    # -------------------------------------------------
    # Analytics
    # -------------------------------------------------

    analytics = VideoAnalytics()


    # -------------------------------------------------
    # Classical verification cache
    # -------------------------------------------------

    verified_tracks = {}


    frame_number = 0


    # =================================================
    # FRAME LOOP
    # =================================================

    while True:

        success, frame = (
            reader.read()
        )


        if not success:
            break


        frame_number += 1


        # =================================================
        # LAYER 2
        # =================================================

        processed_frame = (
            preprocess_frame(
                frame,
                enabled=False
            )
        )


        # =================================================
        # LAYER 5 + LAYER 7
        # =================================================

        tracks = tracker.track(
            processed_frame,
            confidence=0.25
        )


        # =================================================
        # LAYER 6
        # =================================================

        for track in tracks:

            track_id = (
                track["track_id"]
            )


            # Verify each new track once
            if track_id not in verified_tracks:

                try:

                    verification = (
                        verify_track(
                            processed_frame,
                            track
                        )
                    )

                    verified_tracks[
                        track_id
                    ] = verification


                except Exception as error:

                    print(
                        f"\nLayer 6 error "
                        f"for ID {track_id}:"
                    )

                    print(error)


                    verified_tracks[
                        track_id
                    ] = {

                        "classical_class":
                            "unknown",

                        "classical_confidence":
                            0.0
                    }


            verification = (
                verified_tracks[
                    track_id
                ]
            )


            track[
                "classical_class"
            ] = verification[
                "classical_class"
            ]


            track[
                "classical_confidence"
            ] = verification[
                "classical_confidence"
            ]


            track["agreement"] = (
                track["class_name"]
                ==
                track["classical_class"]
            )


        # =================================================
        # LAYER 8
        # =================================================

        analytics.update(
            tracks
        )


        # =================================================
        # DRAW TRACKS + ANALYTICS
        # =================================================

        output_frame = (
            processed_frame.copy()
        )


        # ---------------------------------------------
        # Draw trajectories
        # ---------------------------------------------

        history = (
            analytics.get_history()
        )


        for track_id, points in (
            history.items()
        ):

            if len(points) >= 2:

                for i in range(
                    1,
                    len(points)
                ):

                    cv2.line(

                        output_frame,

                        points[i - 1],

                        points[i],

                        (255, 0, 0),

                        2
                    )


        # ---------------------------------------------
        # Draw vehicle boxes
        # ---------------------------------------------

        for track in tracks:

            x1, y1, x2, y2 = map(
                int,
                track["bbox"]
            )


            track_id = (
                track["track_id"]
            )

            yolo_class = (
                track["class_name"]
            )

            yolo_conf = (
                track["confidence"]
            )

            classical_class = (
                track["classical_class"]
            )

            classical_conf = (
                track["classical_confidence"]
            )


            label1 = (
                f"ID {track_id} | "
                f"YOLO: {yolo_class} "
                f"{yolo_conf:.2f}"
            )


            label2 = (
                f"ML: {classical_class} "
                f"{classical_conf:.2f}"
            )


            cv2.rectangle(

                output_frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            cv2.putText(

                output_frame,

                label1,

                (x1, max(y1 - 25, 20)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 255, 0),

                2
            )


            cv2.putText(

                output_frame,

                label2,

                (x1, max(y1 - 5, 40)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 255, 0),

                2
            )


        # =================================================
        # ANALYTICS PANEL
        # =================================================

        summary = analytics.summary()


        lines = [

            f"Unique Vehicles: "
            f"{summary['unique_vehicles']}",

            f"Cars: "
            f"{summary['class_counts'].get('car', 0)}",

            f"Buses: "
            f"{summary['class_counts'].get('bus', 0)}",

            f"Trucks: "
            f"{summary['class_counts'].get('truck', 0)}",

            f"Motorcycles: "
            f"{summary['class_counts'].get('motorcycle', 0)}",

            f"Frame: {frame_number}"
        ]


        y = 30


        for line in lines:

            cv2.putText(

                output_frame,

                line,

                (20, y),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2
            )

            y += 28


        # =================================================
        # WRITE
        # =================================================

        writer.write(
            output_frame
        )


        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            "INTELLIVISION Video Pipeline",
            output_frame
        )


        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):

            break


    # =================================================
    # CLEANUP
    # =================================================

    reader.release()

    writer.release()

    cv2.destroyAllWindows()


    # =================================================
    # FINAL SUMMARY
    # =================================================

    final_summary = (
        analytics.summary()
    )


    print("\n========================================")
    print("✅ VIDEO PIPELINE COMPLETE")
    print("========================================")


    print(
        f"Frames processed: "
        f"{frame_number}"
    )

    print(
        f"Unique vehicles: "
        f"{final_summary['unique_vehicles']}"
    )

    print(
        f"Class counts: "
        f"{final_summary['class_counts']}"
    )

    print(
        f"Output video:\n"
        f"{OUTPUT_VIDEO}"
    )


if __name__ == "__main__":

    run_video_pipeline()