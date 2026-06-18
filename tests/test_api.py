import numpy as np
from fastapi.testclient import TestClient

import app as api_app


class DummyModel:
    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]])


class OrderAwareDummyModel:
    """Prediction depends on monthly_charges, to check that batch order is preserved."""

    def predict(self, X):
        return [1 if X["monthly_charges"].iloc[0] > 80 else 0]

    def predict_proba(self, X):
        if self.predict(X)[0] == 1:
            return np.array([[0.1, 0.9]])
        return np.array([[0.9, 0.1]])


FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_Month-to-month",
    "contract_One year",
    "contract_Two year",
]


client = TestClient(api_app.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_input(monkeypatch):
    monkeypatch.setattr(api_app, "model", DummyModel())
    monkeypatch.setattr(
        api_app,
        "feature_columns",
        [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "contract_One year",
            "contract_Two year",
        ],
    )

    payload = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 906.0,
        "contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_rejects_missing_field():
    response = client.post("/predict", json={"tenure_months": 12})

    assert response.status_code == 422


def test_predict_batch_valid_input_preserves_order(monkeypatch):
    monkeypatch.setattr(api_app, "model", OrderAwareDummyModel())
    monkeypatch.setattr(api_app, "feature_columns", FEATURE_COLUMNS)

    payload = {
        "inputs": [
            {"tenure_months": 12, "monthly_charges": 75.5, "total_charges": 906.0, "contract": "Month-to-month"},
            {"tenure_months": 24, "monthly_charges": 90.0, "total_charges": 2160.0, "contract": "One year"},
        ]
    }
    response = client.post("/predict_batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["n_inputs"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["prediction"] == 0
    assert body["predictions"][1]["prediction"] == 1


def test_predict_batch_rejects_oversized_batch(monkeypatch):
    monkeypatch.setattr(api_app, "model", DummyModel())
    monkeypatch.setattr(api_app, "feature_columns", FEATURE_COLUMNS)

    single_input = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 906.0,
        "contract": "Month-to-month",
    }
    payload = {"inputs": [single_input] * 101}
    response = client.post("/predict_batch", json=payload)

    assert response.status_code == 413
