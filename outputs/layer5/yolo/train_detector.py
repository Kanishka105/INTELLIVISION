from pathlib import Path
import torch
from ultralytics import YOLO


# -------------------------------------------------
# Get project paths automatically
# -------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

# ...\INTELLIVISION\outputs\layer5\yolo
YOLO_DIR = CURRENT_FILE.parent

# ...\INTELLIVISION\outputs\layer5
LAYER5_DIR = YOLO_DIR.parent

# ...\INTELLIVISION
PROJECT_DIR = LAYER5_DIR.parent.parent

# Dataset configuration file
DATA_YAML = YOLO_DIR / "data.yaml"

# YOLO model
MODEL_NAME = "yolo11n.pt"


# -------------------------------------------------
# Training settings
# -------------------------------------------------

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# Training function
# -------------------------------------------------

def train_model():

    print("======================================")
    print("        LAYER 5 YOLO TRAINING")
    print("======================================")

    print(f"Project directory : {PROJECT_DIR}")
    print(f"YOLO directory    : {YOLO_DIR}")
    print(f"Data YAML         : {DATA_YAML}")
    print(f"Device            : {DEVICE}")

    # Check data.yaml
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found:\n{DATA_YAML}"
        )

    print("\n✅ data.yaml found")

    # Load YOLO
    print("\nLoading YOLO model...")

    model = YOLO(MODEL_NAME)

    print("✅ YOLO model loaded")

    # Start training
    print("\nStarting training...\n")

    model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,

        project=str(LAYER5_DIR / "results"),
        name="vehicle_detector",

        exist_ok=True
    )

    print("\n======================================")
    print("       YOLO TRAINING COMPLETE")
    print("======================================")


if __name__ == "__main__":
    train_model()