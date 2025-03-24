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
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from data_processing import StockData


# Config class to load hyperparameters and model details from YAML
class Config(BaseModel):
    model: Dict[str, str]
    hyperparameters: Dict


class RandomForestModel:
    def __init__(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: Config
    ):
        self.train = train
        self.val = val
        self.test = test
        self.config = config

    def scale_data(self, scaler_name: str) -> np.ndarray:
        """
        Scales data based on the selected scaler.
        """
        scaler = None
        if scaler_name == "standard":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()

        # Fit and transform on training data
        X_train_scaled = scaler.fit_transform(
            self.train.drop(columns=["Extreme_Event"])
        )
        X_val_scaled = scaler.transform(self.val.drop(columns=["Extreme_Event"]))
        X_test_scaled = scaler.transform(self.test.drop(columns=["Extreme_Event"]))

        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    def evaluate_model(
        self, model: RandomForestClassifier, X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        """
        Evaluate the model using cross-validation and return the score (F1 score).
        """
        # by default when passing an int argument to cross_val_score it uses StratifiedKFold if y is is binary or multiclass
        return cross_val_score(
            model,
            X_val,
            y_val,
            cv=self.config.hyperparameters["cv"],
            scoring=self.config.hyperparameters["scoring"][0],
        ).mean()

    def optimize_hyperparameters(self, trials: int = 100) -> RandomForestClassifier:
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

            # Scale data
            X_train_scaled, X_val_scaled, _, scaler = self.scale_data(
                self.config.hyperparameters["scaler"][0]
            )
            y_train = self.train["Extreme_Event"]
            y_val = self.val["Extreme_Event"]

            # Evaluate model
            score = self.evaluate_model(model, X_val_scaled, y_val)
            return score

        # Optuna study setup
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)

        # Log the best parameters and model to MLflow
        logging.info(f"Best parameters: {study.best_params}")
        mlflow.log_params(study.best_params)

        # Train the best model on the full training data and save it
        best_model = RandomForestClassifier(**study.best_params)
        X_train_scaled, _, _, scaler = self.scale_data(
            self.config.hyperparameters["scaler"][0]
        )
        y_train = self.train["Extreme_Event"]
        best_model.fit(X_train_scaled, y_train)
        mlflow.sklearn.log_model(best_model, "random_forest_model")

        # Create model name with experiment ID
        model_filename = (
            f"{self.config.model_name}_{self.config.experiment_id}_best_model.pkl"
        )
        pickle_path = os.path.join("data", model_filename)

        # Save the best model as pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(best_model, f)

        # Save the scaler
        scaler_filename = (
            f"{self.config.model_name}_{self.config.experiment_id}_scaler.pkl"
        )
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

    # Initialize and train model
    rf_model = RandomForestModel(train, val, test, config)
    best_model, scaler = rf_model.optimize_hyperparameters(
        trials=config.hyperparameters["trials"][0]
    )


if __name__ == "__main__":
    run()
