import math


# ============================================================
# 1. EUCLIDEAN DISTANCE
# ============================================================

def euclidean_distance(p1, p2):
    """
    Straight-line distance between two pixels.

    Formula:
    D = sqrt((x2-x1)^2 + (y2-y1)^2)
    """

    x1, y1 = p1
    x2, y2 = p2

    distance = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    return distance


# ============================================================
# 2. CITY-BLOCK / MANHATTAN DISTANCE
# ============================================================

def city_block_distance(p1, p2):
    """
    Distance when movement is only horizontal and vertical.

    Diagonal movement is NOT allowed.

    Formula:
    D4 = |x1-x2| + |y1-y2|
    """

    x1, y1 = p1
    x2, y2 = p2

    distance = (
        abs(x1 - x2) +
        abs(y1 - y2)
    )

    return distance


# ============================================================
# 3. CHESSBOARD / CHEBYSHEV DISTANCE
# ============================================================

def chessboard_distance(p1, p2):
    """
    Distance when diagonal movement is allowed.

    Formula:
    D8 = max(|x1-x2|, |y1-y2|)
    """

    x1, y1 = p1
    x2, y2 = p2

    distance = max(
        abs(x1 - x2),
        abs(y1 - y2)
    )

    return distance


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Two pixel coordinates
    pixel1 = (1, 1)
    pixel2 = (4, 5)

    print("Pixel 1:", pixel1)
    print("Pixel 2:", pixel2)

    print("\nDistance Relationships:")

    print(
        "Euclidean Distance:",
        euclidean_distance(pixel1, pixel2)
    )

    print(
        "City-Block Distance:",
        city_block_distance(pixel1, pixel2)
    )

    print(
        "Chessboard Distance:",
        chessboard_distance(pixel1, pixel2)
    )