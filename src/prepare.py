import os
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(csv_path: str, target_column: str, test_size: float = 0.2):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise KeyError(f"Colonne cible absente : {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y, test_size=test_size, random_state=42)
