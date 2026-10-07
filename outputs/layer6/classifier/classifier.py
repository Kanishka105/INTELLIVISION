from sklearn.ensemble import RandomForestClassifier


def create_classifier():

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    return model


def train_classifier(
    X_train,
    y_train
):

    model = create_classifier()

    model.fit(
        X_train,
        y_train
    )

    return model


def predict(
    model,
    features
):

    prediction = model.predict(
        features
    )

    probabilities = model.predict_proba(
        features
    )

    confidence = probabilities.max(
        axis=1
    )

    return (
        prediction,
        confidence
    )