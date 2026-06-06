import json
import os
import joblib


def save_artifacts(model, metrics: dict, feature_columns, output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pkl")
    metrics_path = os.path.join(output_dir, "metrics.json")
    columns_path = os.path.join(output_dir, "feature_columns.json")

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(list(feature_columns), f, indent=2)

    if not os.path.exists(model_path):
        raise RuntimeError("Le modèle n'a pas été sauvegardé.")

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "columns_path": columns_path,
    }
