import joblib

from src.evaluate import evaluate_model
from src.prepare import prepare_data
from src.train import train_model


ACCURACY_THRESHOLD = 0.70


def test_model_accuracy_above_threshold(tmp_path):
    X_train, X_test, y_train, y_test = prepare_data(
        "data/raw/churn.csv",
        target_column="churn",
    )
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert metrics["accuracy"] >= ACCURACY_THRESHOLD


def test_model_can_be_serialized(tmp_path):
    X_train, X_test, y_train, y_test = prepare_data(
        "data/raw/churn.csv",
        target_column="churn",
    )
    model = train_model(X_train, y_train)
    model_path = tmp_path / "model.pkl"

    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)

    assert loaded_model is not None
