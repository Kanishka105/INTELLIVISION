from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()

VIDEO_DIR = CURRENT_FILE.parent


if str(VIDEO_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(VIDEO_DIR)
    )


def test_imports():

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

    print(
        "✅ All video modules imported"
    )


def test_analytics():

    from video_analytics import (
        VideoAnalytics
    )


    analytics = VideoAnalytics()


    tracks = [

        {
            "track_id": 1,

            "class_name": "car",

            "bbox": [
                10,
                10,
                100,
                100
            ]
        },

        {
            "track_id": 2,

            "class_name": "bus",

            "bbox": [
                200,
                100,
                300,
                200
            ]
        }
    ]


    analytics.update(
        tracks
    )


    summary = analytics.summary()


    assert (
        summary["unique_vehicles"]
        == 2
    )


    assert (
        summary["class_counts"]["car"]
        == 1
    )


    assert (
        summary["class_counts"]["bus"]
        == 1
    )


    print(
        "✅ Video Analytics"
    )


def run_tests():

    print("\n================================")
    print("    VIDEO PIPELINE TEST")
    print("================================\n")


    test_imports()

    test_analytics()


    print("\n================================")
    print("✅ VIDEO PIPELINE TEST PASSED")
    print("================================\n")


if __name__ == "__main__":

    run_tests()