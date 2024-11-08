import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any


def safe_log(x):
    """Match the original transformation exactly"""
    return np.log1p(np.clip(x, 0, None))


def inverse_safe_log(x):
    """Match the original inverse transformation exactly"""
    return np.expm1(x)


def clean_row_dict(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Clean dictionary values for JSON serialization"""
    cleaned_dict = {}
    for k, v in row_dict.items():
        if pd.isna(v):
            cleaned_dict[k] = None
        elif isinstance(v, (np.floating, np.integer)):
            cleaned_dict[k] = float(v)
        else:
            cleaned_dict[k] = v
    return cleaned_dict


def evaluate_model_api(
    df: pd.DataFrame,
    api_url: str = "http://localhost:8000/predict"
) -> None:
    """
    Evaluate model performance by making API requests and matching original script's logic
    """

    y = df['target']
    y_log = safe_log(y)

    predictions_log = []

    for i, row in df.iterrows():
        row_dict = row.to_dict()
        cleaned_dict = clean_row_dict(row_dict)

        try:
            response = requests.post(api_url, json=cleaned_dict)
            response.raise_for_status()
            prediction_log = response.json()['prediction']
            predictions_log.append(prediction_log)

        except Exception as e:
            print(f"Error processing row {i}: {str(e)}")
            return

    predictions_log = np.array(predictions_log)

    mse_log = mean_squared_error(y_log, predictions_log)
    r2_log = r2_score(y_log, predictions_log)

    print('mse (log scale):', mse_log)
    print('r2 (log scale):', r2_log)
    print('Prediction:', inverse_safe_log(predictions_log))


if __name__ == "__main__":
    file_path = "../candidateschallenge/candidates_data.csv"
    df = pd.read_csv(file_path)
    df = df.head(3)

    evaluate_model_api(df)
