import joblib
import numpy as np
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()

LAYER6_DIR = CURRENT_FILE.parents[1]

MODEL_PATH = (
    LAYER6_DIR
    / "results"
    / "vehicle_classifier.joblib"
)


def load_classifier():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Classifier model not found:\n"
            f"{MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


def predict_vehicle(
    model,
    feature_vector
):

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    prediction = model.predict(
        feature_vector
    )[0]

    probabilities = (
        model.predict_proba(
            feature_vector
        )[0]
    )

    confidence = float(
        probabilities.max()
    )

    return prediction, confidence