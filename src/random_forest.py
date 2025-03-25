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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_processing import StockData


# Config class to load hyperparameters and model details from YAML
class Config(BaseModel):
    model: Dict[str, str]
    hyperparameters: Dict


class RandomForestModel:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, config_path: str):
        # Load config from YAML file
        with open(config_path) as file:
            config_data = yaml.safe_load(file)
        self.config = Config(**config_data)

        self.X_train = X_train
        self.y_train = y_train

        # Perform scaling in the constructor for global access
        self.X_train_scaled, self.scaler = self._scale_data(
            X_train, self.config.hyperparameters["scaler"][0]
        )

    def _scale_data(
        self,
        X_train: np.ndarray,
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
        X_train_scaled = scaler.fit_transform(X_train)

        return X_train_scaled, scaler

    def tune_model(
        self, model: RandomForestClassifier, X_train: np.ndarray, y_train: np.ndarray
    ) -> float:
        """
        Tune the model using StratifiedShuffleSplit and return the F1 score.
        """
        sss = StratifiedShuffleSplit(
            n_splits=self.config.hyperparameters["cv"],
            test_size=0.2,
            random_state=self.config.hyperparameters["random_state"],
        )

        scores = []
        for train_idx, val_idx in sss.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model.fit(X_train_fold, y_train_fold)
            pred = model.predict(X_val_fold)
            score = f1_score(y_val_fold, pred)
            scores.append(score)

        return np.mean(scores)

    def optimize_hyperparameters(self) -> RandomForestClassifier:
        """
        Hyperparameter optimization using Optuna.
        """

        trials = self.config.hyperparameters["trials"]

        def objective(trial):
            return self.objective_fn(trial)

        # Optuna study setup
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)

        # Train the best model on the full training data
        best_model = RandomForestClassifier(**study.best_params)
        best_model.fit(self.X_train_scaled, self.y_train)

        # Log and save model
        self.log_and_save_model(best_model)

        return best_model

    def objective_fn(self, trial):
        """
        Objective function to be used in Optuna optimization.
        """
        # Define the hyperparameters to be optimized
        n_estimators = trial.suggest_int(
            "n_estimators",
            self.config.hyperparameters["num_estimators_range"][0],
            self.config.hyperparameters["num_estimators_range"][1],
        )
        max_depth = trial.suggest_int(
            "max_depth",
            self.config.hyperparameters["max_depth_range"][0],
            self.config.hyperparameters["max_depth_range"][1],
        )
        min_samples_split = trial.suggest_int(
            "min_samples_split",
            self.config.hyperparameters["min_samples_split_range"][0],
            self.config.hyperparameters["min_samples_split_range"][1],
        )
        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            self.config.hyperparameters["min_samples_leaf_range"][0],
            self.config.hyperparameters["min_samples_leaf_range"][1],
        )
        class_weight = trial.suggest_categorical(
            "class_weight", self.config.hyperparameters["class_weight"]
        )

        # Build the model with suggested hyperparameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=self.config.hyperparameters["random_state"],
        )

        # Evaluate model
        score = self.tune_model(model, self.X_train_scaled, self.y_train)
        return score

    def log_and_save_model(self, model: RandomForestClassifier):
        """
        Logs the model and metrics using MLflow and saves the best model as a pickle file.
        """
        # Start an MLflow run and log the model and metrics
        with mlflow.start_run():
            mlflow.set_experiment(
                f"Extreme Stock Event Prediction {self.config.model['experiment_id']}"
            )
            mlflow.set_tag("model_name", self.config.model["name"])
            mlflow.set_tag("experiment_id", self.config.model["experiment_id"])

            logging.info(f"Best parameters: {model.get_params()}")
            mlflow.log_params(model.get_params())
            mlflow.log_metric(
                "f1_score", self.tune_model(model, self.X_train_scaled, self.y_train)
            )
            mlflow.sklearn.log_model(
                sk_model=model, artifact_path="random_forest_model"
            )

        # Save the best model as a pickle file
        model_filename = f"{self.config.model['name']}_{self.config.model['experiment_id']}_best_model.pkl"
        pickle_path = os.path.join("data", model_filename)
        with open(pickle_path, "wb") as f:
            pickle.dump(model, f)

        # Save the scaler
        scaler_filename = f"{self.config.model['name']}_{self.config.model['experiment_id']}_scaler.pkl"
        scaler_path = os.path.join("data", scaler_filename)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)


def run():
    # Load and partition data
    stock_data = StockData(
        ticker="AAPL", start_date="2015-01-01", end_date="2024-01-01"
    )
    train, val, _ = stock_data.partition_data()
    train = pd.concat([train, val])

    # Initialize and train model with window sizes for lookback and horizon
    lookback = 10
    horizon = 1

    # get windows
    X_train, y_train = stock_data.get_windows(train, lookback, horizon, "Extreme_Event")

    # reshape to rows,lookback*features
    X_train = X_train.reshape(X_train.shape[0], -1)

    # turn to 1d array
    y_train = y_train.ravel()

    # instantiate rf
    rf_model = RandomForestModel(X_train, y_train, "data/rf_config.yaml")

    # tune hyperparams and get best model
    rf_model.optimize_hyperparameters()


if __name__ == "__main__":
    run()
