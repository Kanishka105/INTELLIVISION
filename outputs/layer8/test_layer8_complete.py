import sys
from pathlib import Path


# =================================================
# PATHS
# =================================================

CURRENT_FILE = Path(__file__).resolve()

LAYER8_DIR = CURRENT_FILE.parent

ANALYTICS_DIR = (
    LAYER8_DIR / "analytics"
)

if str(ANALYTICS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ANALYTICS_DIR)
    )


# =================================================
# TEST 1 — VEHICLE COUNTER
# =================================================

def test_vehicle_counter():

    from vehicle_counter import (
        VehicleCounter
    )

    counter = VehicleCounter()

    tracks = [
        {
            "track_id": 1,
            "class_name": "car"
        },
        {
            "track_id": 2,
            "class_name": "bus"
        },
        {
            "track_id": 3,
            "class_name": "car"
        }
    ]

    counter.update(tracks)

    # Same objects appear again
    counter.update(tracks)

    assert counter.total_unique() == 3

    counts = counter.get_class_counts()

    assert counts["car"] == 2
    assert counts["bus"] == 1

    print("✅ Vehicle Counter")


# =================================================
# TEST 2 — LINE CROSSING
# =================================================

def test_line_crossing():

    from line_crossing import (
        LineCrossingCounter
    )

    counter = LineCrossingCounter(
        (0, 100),
        (200, 100)
    )

    # Above the line
    counter.update(
        1,
        (100, 80)
    )

    # Below the line
    crossed, direction = counter.update(
        1,
        (100, 120)
    )

    assert crossed is True

    assert direction in [
        "A_to_B",
        "B_to_A"
    ]

    print("✅ Line Crossing")


# =================================================
# TEST 3 — SPEED
# =================================================

def test_speed():

    from speed_estimator import (
        SpeedEstimator
    )

    estimator = SpeedEstimator()

    # First position
    estimator.update(
        1,
        (100, 100),
        30
    )

    # Move 10 pixels in one frame
    speed, speed_kmh = estimator.update(
        1,
        (110, 100),
        30
    )

    # 10 pixels/frame × 30 FPS
    assert speed == 300.0

    assert speed_kmh is None

    print("✅ Speed Estimation")


# =================================================
# TEST 4 — TRAFFIC DENSITY
# =================================================

def test_density():

    from traffic_density import (
        calculate_density
    )

    tracks = [
        {
            "bbox": [
                0,
                0,
                100,
                100
            ]
        }
    ]

    result = calculate_density(
        tracks,
        1000,
        1000
    )

    assert result["vehicle_count"] == 1

    assert result["occupancy"] > 0

    assert result["level"] in [
        "Low",
        "Medium",
        "High"
    ]

    print("✅ Traffic Density")


# =================================================
# MAIN
# =================================================

def run_all_tests():

    print("\n========================================")
    print("        LAYER 8 TEST START")
    print("========================================\n")

    test_vehicle_counter()

    test_line_crossing()

    test_speed()

    test_density()

    print("\n========================================")
    print("✅ LAYER 8 BASIC TESTS PASSED")
    print("========================================\n")


if __name__ == "__main__":
    run_all_tests()