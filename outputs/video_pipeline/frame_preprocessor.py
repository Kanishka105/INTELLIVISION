from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()

OUTPUTS_DIR = CURRENT_FILE.parents[1]

LAYER2_DIR = OUTPUTS_DIR / "layer2"

GAUSSIAN_DIR = (
    LAYER2_DIR
    / "Smoothing"
    / "Linear"
)

if str(GAUSSIAN_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(GAUSSIAN_DIR)
    )


from gaussian_filter import gaussian_filter


def preprocess_frame(
    frame,
    enabled=False
):

    if not enabled:
        return frame

    return gaussian_filter(
        frame
    )