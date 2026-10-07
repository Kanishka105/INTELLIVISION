from pathlib import Path
import sys
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -------------------------------------------------
# Paths
# -------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

LAYER6_DIR = CURRENT_FILE.parents[1]

DATASET_DIR = (
    LAYER6_DIR / "dataset"
)

RESULTS_DIR = (
    LAYER6_DIR / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)

CSV_PATH = (
    DATASET_DIR
    / "vehicle_features.csv"
)

MODEL_PATH = (
    RESULTS_DIR
    / "vehicle_classifier.joblib"
)


# -------------------------------------------------
# Import classifier
# -------------------------------------------------

CLASSIFIER_DIR = CURRENT_FILE.parent

if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CLASSIFIER_DIR)
    )

from classifier import train_classifier


# -------------------------------------------------
# Training
# -------------------------------------------------

def main():

    print("===================================")
    print("LAYER 6 CLASSICAL ML TRAINING")
    print("===================================")

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{CSV_PATH}\n\n"
            f"Run build_feature_dataset.py first."
        )

    dataframe = pd.read_csv(
        CSV_PATH
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith("feature_")
    ]

    if not feature_columns:
        raise RuntimeError(
            "No feature columns found."
        )

    X = dataframe[
        feature_columns
    ].values

    y = dataframe[
        "label"
    ].values

    print(
        f"Samples: {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Classes: {sorted(set(y))}"
    )

    # Train-test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print("\nTraining Random Forest...")

    model = train_classifier(
        X_train,
        y_train
    )

    # Evaluation
    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # Save
    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\n✅ Model saved to:\n"
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()