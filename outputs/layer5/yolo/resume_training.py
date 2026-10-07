from pathlib import Path
from ultralytics import YOLO


CURRENT_FILE = Path(__file__).resolve()

LAYER5_DIR = CURRENT_FILE.parent.parent

LAST_PT = (
    LAYER5_DIR
    / "results"
    / "vehicle_detector"
    / "weights"
    / "last.pt"
)


def resume_training():

    print("======================================")
    print("       RESUME LAYER 5 YOLO TRAINING")
    print("======================================")

    print(f"Checkpoint: {LAST_PT}")

    if not LAST_PT.exists():
        raise FileNotFoundError(
            f"last.pt not found:\n{LAST_PT}"
        )

    model = YOLO(str(LAST_PT))

    print("✅ Checkpoint loaded")
    print("Starting resume...\n")

    model.train(resume=True)

    print("\n======================================")
    print("✅ RESUMED TRAINING FINISHED")
    print("======================================")


if __name__ == "__main__":
    resume_training()