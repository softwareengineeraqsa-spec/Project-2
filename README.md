# Project-2
House Price Prediction
# House Price Prediction

This project trains regression models to predict house prices from property features such as square footage, bedrooms, bathrooms, and location-based fields.

## Setup

1. Download a Kaggle house price dataset as CSV.
2. Create a `data` folder in the project root.
3. Place the CSV file inside `data/`.

Expected structure:

```text
House price prediction/
|-- data/
|   |-- your_dataset.csv
|-- house_price_prediction.py
```

## What the script does

- Loads the first CSV file from `data/`
- Detects the target column automatically from common names like `price` or `SalePrice`
- Preprocesses numeric and categorical features
- Scales numeric values and one-hot encodes location/category columns
- Trains both `Linear Regression` and `Gradient Boosting`
- Evaluates models using `MAE`, `RMSE`, and `R2`
- Saves a predicted-vs-actual plot

## Run

```bash
python house_price_prediction.py
```

## Outputs

Files are written to `outputs/`:

- `metrics.json`
- `model_results.csv`
- `prediction_preview.csv`
- `predicted_vs_actual.png`
