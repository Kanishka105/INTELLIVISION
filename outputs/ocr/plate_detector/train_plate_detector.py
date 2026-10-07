from pathlib import Path
from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

OCR_DIR = CURRENT_FILE.parent.parent

DATA_YAML = (
    CURRENT_FILE.parent
    / "data.yaml"
)

RESULTS_DIR = (
    OCR_DIR
    / "results"
    / "plate_detector"
)


# =========================================================
# TRAIN
# =========================================================

def train():

    print("========================================")
    print("   INTELLIVISION PLATE DETECTOR")
    print("========================================")

    print("\nDataset:")
    print(DATA_YAML)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    # Lightweight YOLO model
    model = YOLO(
        "yolo11n.pt"
    )

    model.train(

        # Dataset
        data=str(DATA_YAML),

        # More training for the small dataset
        epochs=100,

        # Larger resolution for small plates
        imgsz=1024,

        # CPU-friendly batch
        batch=2,

        # CPU
        device="cpu",

        # Save results here
        project=str(RESULTS_DIR.parent),

        # Experiment name
        name="plate_detector",

        # Allow existing folder
        exist_ok=True
    )

    print(
        "\n✅ PLATE DETECTOR TRAINING COMPLETE"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    train()