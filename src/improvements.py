import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from data_processing import StockData
from temporal_cnn import TCNNTrainer


def add_classic_features(df):
    """Adds day of the week as a feature."""
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


def apply_sin_cos_transform(df):
    """
    Applies sine and cosine transformations to numerical features.
    """
    df = df.copy()

    # Exclude target & date columns
    exclude_cols = ["Extreme_Event"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    for col in feature_cols:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / 10)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / 10)

    return df


def undersample(df: pd.DataFrame, target: str):
    smote = RandomUnderSampler(sampling_strategy="majority", random_state=42)
    X_train = df.drop(target, axis=1)
    y_train = df[target]
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return pd.concat([X_train_resampled, y_train_resampled], axis=1)


def prepare_data(stock_data, lookback, horizon, target_col):
    """Loads and transforms data for TCN training."""
    train, val, _ = stock_data.partition_data()

    # Apply feature engineering
    train = apply_sin_cos_transform(add_classic_features(train))
    val = apply_sin_cos_transform(add_classic_features(val))
    train = undersample(train, "Extreme_Event")

    # Get windows for TCN
    X_train, y_train = stock_data.get_windows(train, lookback, horizon, target_col)
    X_val, y_val = stock_data.get_windows(val, lookback, horizon, target_col)

    return X_train, y_train, X_val, y_val


def main():
    stock_data = StockData(
        ticker="AAPL", start_date="2015-01-01", end_date="2024-01-01"
    )
    lookback, horizon = 10, 1
    target_col = "Extreme_Event"

    X_train, y_train, X_val, y_val = prepare_data(
        stock_data, lookback, horizon, target_col
    )

    trainer = TCNNTrainer(X_train, y_train, X_val, y_val, "data/tcn_config.yaml")
    trainer.optimize_hyperparameters()


if __name__ == "__main__":
    main()
