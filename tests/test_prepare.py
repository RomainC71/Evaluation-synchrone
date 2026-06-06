import pandas as pd
import pytest

from src.prepare import prepare_data


def test_prepare_data_returns_four_objects(tmp_path):
    df = pd.DataFrame(
        {
            "tenure_months": [1, 12, 24, 36, 48, 60],
            "monthly_charges": [20.0, 50.0, 70.0, 80.0, 90.0, 100.0],
            "total_charges": [20.0, 600.0, 1680.0, 2880.0, 4320.0, 6000.0],
            "contract": ["Month-to-month", "Month-to-month", "One year", "One year", "Two year", "Two year"],
            "churn": [1, 1, 0, 0, 0, 0],
        }
    )
    csv_path = tmp_path / "mini_churn.csv"
    df.to_csv(csv_path, index=False)

    result = prepare_data(str(csv_path), target_column="churn")

    assert len(result) == 4


def test_prepare_data_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        prepare_data("data/raw/missing.csv", target_column="churn")


def test_prepare_data_raises_on_missing_target(tmp_path):
    df = pd.DataFrame(
        {
            "tenure_months": [1, 2],
            "monthly_charges": [20.0, 30.0],
            "total_charges": [20.0, 60.0],
            "contract": ["Month-to-month", "One year"],
        }
    )
    csv_path = tmp_path / "no_target.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(KeyError):
        prepare_data(str(csv_path), target_column="churn")
