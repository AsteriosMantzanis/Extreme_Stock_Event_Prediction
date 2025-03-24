import os
import pickle
from typing import Dict, List

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger as logging
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_processing import StockData


# Config class to load hyperparameters and model details from YAML
class Config(BaseModel):
    model: Dict[str, str]
    hyperparameters: Dict


class RandomForestModel:
    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        config: Config,
        lookback: int,
        horizon: int,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.config = config
        self.lookback = lookback
        self.horizon = horizon

    def scale_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        scaler_name: str,
    ) -> np.ndarray:
        """
        Scales data based on the selected scaler.
        """
        scaler = None
        if scaler_name == "standard":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()

        # Fit and transform on training data
        # reshape from rows,lookback,features to rows,lookback*features
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    def evaluate_model(
        self, model: RandomForestClassifier, X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        """
        Evaluate the model using cross-validation and return the score (F1 score).
        """
        return cross_val_score(
            model,
            X_val,
            y_val,
            cv=self.config.hyperparameters["cv"],
            scoring=self.config.hyperparameters["scoring"][0],
        ).mean()

    def optimize_hyperparameters(self, trials: int) -> RandomForestClassifier:
        """
        Hyperparameter optimization using Optuna.
        """

        def objective(trial):
            # Define the hyperparameters to be optimized
            n_estimators = trial.suggest_categorical(
                "n_estimators", self.config.hyperparameters["n_estimators"]
            )
            max_depth = trial.suggest_categorical(
                "max_depth", self.config.hyperparameters["max_depth"]
            )
            min_samples_split = trial.suggest_categorical(
                "min_samples_split", self.config.hyperparameters["min_samples_split"]
            )
            class_weight = trial.suggest_categorical(
                "class_weight", self.config.hyperparameters["class_weight"]
            )
            scoring = self.config.hyperparameters["scoring"][0]

            # Build the model with suggested hyperparameters
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                class_weight=class_weight,
                random_state=self.config.hyperparameters["random_state"],
            )

            # Generate windows from the train and validation data
            X_train, y_train = StockData.get_windows(
                self.train, self.lookback, self.horizon, "Extreme_Event"
            )
            X_val, y_val = StockData.get_windows(
                self.val, self.lookback, self.horizon, "Extreme_Event"
            )
            X_test, y_test = StockData.get_windows(
                self.test, self.lookback, self.horizon, "Extreme_Event"
            )

            y_train, y_val, y_test = y_train.ravel(), y_val.ravel(), y_test.ravel()

            # Scale data
            X_train_scaled, X_val_scaled, X_test, scaler = self.scale_data(
                X_train, X_val, X_test, self.config.hyperparameters["scaler"][0]
            )

            # Evaluate model
            score = self.evaluate_model(model, X_val_scaled, y_val)
            return score

        # Optuna study setup
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)

        # Train the best model on the full training data
        best_model = RandomForestClassifier(**study.best_params)

        # Generate windows for training data
        X_train, y_train = StockData.get_windows(
            self.train, self.lookback, self.horizon, "Extreme_Event"
        )
        X_train_scaled, _, _, scaler = self.scale_data(
            X_train, None, None, self.config.hyperparameters["scaler"][0]
        )

        # Convert scaled array back to DataFrame with feature names
        X_train_scaled_df = pd.DataFrame(
            X_train_scaled, columns=self.train.drop(columns=["Extreme_Event"]).columns
        )
        best_model.fit(X_train_scaled_df, y_train)

        # Prepare MLflow logging
        input_example = pd.DataFrame(
            X_train_scaled[:5],
            columns=self.train.drop(columns=["Extreme_Event"]).columns,
        )
        signature = mlflow.models.infer_signature(
            input_example, best_model.predict(input_example)
        )

        # Start an MLflow run and log the model and metrics
        with mlflow.start_run():
            # Set the model name and experiment ID
            mlflow.set_tag("model_name", self.config.model["name"])
            mlflow.set_tag("experiment_id", self.config.model["experiment_id"])

            logging.info(f"Best parameters: {study.best_params}")
            mlflow.log_params(study.best_params)
            mlflow.log_metric(
                "f1_score", study.best_value
            )  # Log the best F1 score from Optuna
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="random_forest_model",
                signature=signature,
                input_example=input_example,
            )

        # Create model name with experiment ID
        model_filename = f"{self.config.model['name']}_{self.config.model['experiment_id']}_best_model.pkl"
        pickle_path = os.path.join("data", model_filename)

        # Save the best model as a pickle file
        with open(pickle_path, "wb") as f:
            pickle.dump(best_model, f)

        # Save the scaler
        scaler_filename = f"{self.config.model['name']}_{self.config.model['experiment_id']}_scaler.pkl"
        scaler_path = os.path.join("data", scaler_filename)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        return best_model, scaler


def run():
    # Load config from YAML file
    with open("data/rf_config.yaml") as file:
        config_data = yaml.safe_load(file)
    config = Config(**config_data)

    # Load and partition data
    stock_data = StockData(
        ticker="AAPL", start_date="2015-01-01", end_date="2024-01-01"
    )
    train, val, test = stock_data.partition_data()

    # Initialize and train model with window sizes for lookback and horizon
    lookback = 10
    horizon = 5
    rf_model = RandomForestModel(train, val, test, config, lookback, horizon)
    best_model, scaler = rf_model.optimize_hyperparameters(
        trials=config.hyperparameters["trials"]
    )


if __name__ == "__main__":
    run()
