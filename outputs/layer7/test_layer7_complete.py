from pathlib import Path
import sys


# =================================================
# PROJECT PATH
# =================================================

CURRENT_FILE = Path(__file__).resolve()

LAYER7_DIR = CURRENT_FILE.parent

PROJECT_DIR = LAYER7_DIR.parent.parent


# Make INTELLIVISION/outputs available
OUTPUTS_DIR = LAYER7_DIR.parent

if str(OUTPUTS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(OUTPUTS_DIR)
    )


# =================================================
# TEST IMPORTS
# =================================================

def test_imports():

    from layer7.tracking.tracker import (
        load_tracker,
        track_frame
    )

    from layer7.tracking.track_manager import (
        TrackManager
    )

    from layer7.tracking.trajectory import (
        TrajectoryManager
    )

    from layer7.movement.direction import (
        calculate_direction
    )

    from layer7.movement.displacement import (
        calculate_displacement
    )

    print("✅ Tracking Modules")


# =================================================
# TEST TRACK MANAGER
# =================================================

def test_track_manager():

    from layer7.tracking.track_manager import (
        TrackManager
    )

    manager = TrackManager(
        max_history=10
    )

    tracks = [
        {
            "track_id": 1,

            "bbox": [
                10,
                20,
                110,
                120
            ],

            "class_name": "car",

            "confidence": 0.90
        }
    ]

    manager.update(
        tracks
    )

    obj = manager.get_object(
        1
    )

    assert obj is not None

    assert obj["center"] == (
        60,
        70
    )

    trajectory = (
        manager.get_trajectory(
            1
        )
    )

    assert len(trajectory) == 1

    print("✅ Track Manager")


# =================================================
# TEST DIRECTION
# =================================================

def test_direction():

    from layer7.movement.direction import (
        calculate_direction
    )

    result = calculate_direction(
        (100, 200),
        (150, 200)
    )

    assert result == "right"

    print("✅ Direction Calculation")


# =================================================
# TEST DISPLACEMENT
# =================================================

def test_displacement():

    from layer7.movement.displacement import (
        calculate_displacement
    )

    result = calculate_displacement(
        (0, 0),
        (3, 4)
    )

    assert result == 5.0

    print("✅ Displacement Calculation")


# =================================================
# MAIN
# =================================================

def run_all_tests():

    print("\n================================")
    print("     LAYER 7 TEST START")
    print("================================\n")

    test_imports()

    test_track_manager()

    test_direction()

    test_displacement()

    print("\n================================")
    print("✅ LAYER 7 BASIC TESTS PASSED")
    print("================================\n")


if __name__ == "__main__":
    run_all_tests()