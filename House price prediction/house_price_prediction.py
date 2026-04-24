from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
COMMON_TARGETS = [
    "price",
    "saleprice",
    "house_price",
    "selling_price",
    "target",
]


def normalize_name(name: str) -> str:
    return "".join(char.lower() for char in name if char.isalnum())


def detect_target_column(columns: Iterable[str]) -> str:
    normalized_map = {normalize_name(column): column for column in columns}
    for candidate in COMMON_TARGETS:
        match = normalized_map.get(normalize_name(candidate))
        if match:
            return match
    raise ValueError(
        "Could not detect the target price column automatically. "
        f"Expected one of: {', '.join(COMMON_TARGETS)}"
    )


def find_dataset() -> Path:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Create the folder and place your Kaggle house price CSV inside it."
        )

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {DATA_DIR}\n"
            "Place the downloaded Kaggle dataset there, for example: data/house_prices.csv"
        )

    return csv_files[0]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(dataset_path)
    if dataframe.empty:
        raise ValueError(f"The dataset {dataset_path} is empty.")
    return dataframe


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = pipeline.predict(x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    metrics = {
        "model": name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_test, predictions)),
    }
    return metrics


def save_plot(actual: pd.Series, predicted: np.ndarray, model_name: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_path = OUTPUT_DIR / "predicted_vs_actual.png"

    plt.figure(figsize=(9, 6))
    plt.scatter(actual, predicted, alpha=0.7, edgecolors="white", linewidths=0.5)
    reference_min = min(actual.min(), predicted.min())
    reference_max = max(actual.max(), predicted.max())
    plt.plot(
        [reference_min, reference_max],
        [reference_min, reference_max],
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )
    plt.title(f"Predicted vs Actual House Prices ({model_name})")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def main() -> None:
    dataset_path = find_dataset()
    dataframe = load_dataset(dataset_path)
    target_column = detect_target_column(dataframe.columns)

    dataframe = dataframe.dropna(subset=[target_column]).copy()
    x = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    if x.empty:
        raise ValueError("The dataset does not contain any feature columns after removing the target.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(x)
    model_pipelines = {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    }

    results = []
    trained_models = {}
    for model_name, pipeline in model_pipelines.items():
        pipeline.fit(x_train, y_train)
        trained_models[model_name] = pipeline
        results.append(evaluate_model(model_name, pipeline, x_test, y_test))

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    best_model_name = results_df.loc[0, "model"]
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(x_test)

    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics.json"
    results_path = OUTPUT_DIR / "model_results.csv"
    plot_path = save_plot(y_test, best_predictions, best_model_name)

    metrics_payload = {
        "dataset": str(dataset_path.name),
        "target_column": target_column,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(x.shape[1]),
        "models": results,
        "best_model": best_model_name,
        "plot": str(plot_path.name),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    results_df.to_csv(results_path, index=False)

    preview = pd.DataFrame(
        {
            "actual_price": y_test.to_numpy(),
            "predicted_price": best_predictions,
        }
    ).head(10)
    preview.to_csv(OUTPUT_DIR / "prediction_preview.csv", index=False)

    print(f"Dataset loaded: {dataset_path.name}")
    print(f"Target column: {target_column}")
    print(f"Train rows: {len(x_train)} | Test rows: {len(x_test)}")
    print("\nModel performance:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved comparison chart to: {plot_path}")


if __name__ == "__main__":
    main()
