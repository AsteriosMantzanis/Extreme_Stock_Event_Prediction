import argparse
import os
import pickle

import mlflow
import numpy as np
import torch
from loguru import logger as logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from data_processing import StockData
from improvements import add_classic_features, apply_sin_cos_transform
from temporal_cnn import TemporalCNN


class ModelEvaluator:
    def __init__(self, trial: int, improved: False):
        self.trial = trial
        self.improved = improved
        self.model_dispatch = {"rf": self.rf_evaluation, "tcn": self.tcn_evaluate}

        # Load stock data once in the constructor
        self.stock_data = StockData(
            ticker="AAPL", start_date="2015-01-01", end_date="2024-01-01"
        )
        _, _, self.test = self.stock_data.partition_data()

        # Compute and store test set windows
        self.lookback = 10
        self.horizon = 1
        if self.improved:
            self.test = add_classic_features(self.test)
            self.test = apply_sin_cos_transform(self.test)

        self.X_test, self.y_test = self.stock_data.get_windows(
            self.test, self.lookback, self.horizon, "Extreme_Event"
        )

        # Store shape variables
        self.num_test_samples, self.lookback, self.num_features = self.X_test.shape

    def log_metrics(self, model_name, metrics):
        with mlflow.start_run(run_name=f"{model_name}_trial_{self.trial}"):
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.float32, np.float64)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(
                        key, str(value)
                    )  # Log non-scalars as string parameters

    def load_artifact(self, path: str):
        """Load an artifact from storage (pickle format)."""
        try:
            with open(path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            logging.error(f"Failed to load pickle artifact from {path}: {e}")
            return None

    def load_torch_model(self, path: str, params: dict):
        """Load a PyTorch model from a .pth file using stored parameters."""
        try:
            model = TemporalCNN(
                input_channels=self.num_features,  # Use precomputed value
                lookback=self.lookback,
                out_channels=params["conv1_out_channels"],
                kernel_size=params["kernel_size"],
                dropout=params["dropout"],
            )
            model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Failed to load PyTorch model from {path}: {e}")
            return None

    def locate_artifact(self, model_type: str, cache: str):
        """Locate and load model and scaler artifacts based on model type and trial."""
        model, scaler, params = None, None, None

        # Collect files first pass: load params and scaler
        params_files = []
        scaler_files = []
        model_files = []

        for filename in os.listdir(cache):
            file_path = os.path.join(cache, filename)

            parts = filename.split("_")

            if len(parts) < 3:
                continue  # Skip unexpected filenames

            try:
                trial_number = int(parts[-2])  # Extract trial number
            except ValueError:
                continue  # Skip if trial number isn't an integer

            if trial_number != self.trial:
                continue  # Skip files that don't match the current trial

            # Categorize files
            if model_type == "rf":
                if filename.startswith("random_forest"):
                    if "model" in filename:
                        model_files.append(file_path)
                    elif "scaler" in filename:
                        scaler_files.append(file_path)
            elif model_type == "tcn":
                if filename.startswith("TemporalCNN"):
                    if "params" in filename:
                        params_files.append(file_path)
                    elif "scaler" in filename:
                        scaler_files.append(file_path)
                    elif "model" in filename and filename.endswith(".pth"):
                        model_files.append(file_path)

        # First pass: Load params and scaler
        if model_type == "tcn" and params_files:
            params = self.load_artifact(params_files[0])  # Load params first

        if scaler_files:
            scaler = self.load_artifact(scaler_files[0])  # Load scaler

        # Second pass: Load model (now that params and scaler are available)
        if model_type == "tcn" and params and model_files:
            model = self.load_torch_model(
                model_files[0], params
            )  # Now load the model with params
        else:
            # load rf model
            model = self.load_artifact(model_files[0])

        return model, scaler, params

    def rf_evaluation(self):
        logging.info(f"Evaluating Random Forest Model - Trial {self.trial}")
        model, scaler, _ = self.locate_artifact("rf", "data")

        if model is None or scaler is None:
            logging.error("Model or scaler not found. Skipping RF evaluation.")
            return

        X_test = self.X_test.reshape(self.X_test.shape[0], -1)
        X_test = scaler.transform(X_test)
        y_test = self.y_test.ravel()

        y_pred = model.predict(X_test)

        metrics = {
            "ROC AUC Score": roc_auc_score(y_test, y_pred),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred),
        }
        self.log_metrics("tcn", metrics)

        [logging.info(f"{i}: {j}") for i, j in zip(metrics.keys(), metrics.values())]

    def tcn_evaluate(self):
        logging.info(f"Evaluating TCN Model - Trial {self.trial}")
        model, scaler, params = self.locate_artifact("tcn", "data")

        if model is None or scaler is None or params is None:
            logging.error(
                "Model, scaler, or params not found. Skipping TCN evaluation."
            )
            return

        # Reshape & scale test data
        X_test = self.X_test.reshape(-1, self.num_features)
        X_test_scaled = scaler.transform(X_test)

        X_test_scaled = X_test_scaled.reshape(
            self.num_test_samples, self.lookback, self.num_features
        )
        X_test_scaled = (
            torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1).to("cpu")
        )
        y_test = torch.tensor(self.y_test.ravel(), dtype=torch.long).to("cpu")

        test_dataset = TensorDataset(X_test_scaled, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_dataloader:
                outputs = model(X_batch)
                preds = torch.softmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy()[:, 1])
                all_labels.extend(y_batch.cpu().numpy())

        y_pred_binary = [1 if p > 0.5 else 0 for p in all_preds]

        metrics = {
            "ROC AUC Score": roc_auc_score(all_labels, all_preds),
            "Accuracy": accuracy_score(all_labels, y_pred_binary),
            "Precision": precision_score(all_labels, y_pred_binary),
            "Recall": recall_score(all_labels, y_pred_binary),
            "F1 Score": f1_score(all_labels, y_pred_binary),
            "Confusion Matrix": confusion_matrix(all_labels, y_pred_binary),
            "Classification Report": classification_report(all_labels, y_pred_binary),
        }
        self.log_metrics("tcn", metrics)

        [logging.info(f"{i}: {j}") for i, j in zip(metrics.keys(), metrics.values())]

    def evaluate(self, model_name):
        """Dispatches the evaluation based on model type."""
        self.model_dispatch.get(model_name, self.unknown_model)()

    def unknown_model(self):
        logging.error(f"Unknown model type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate machine learning models.")
    parser.add_argument(
        "model", choices=["rf", "tcn"], help="Model type to evaluate (rf or tcn)"
    )
    parser.add_argument("trial", type=int, help="Trial number")
    parser.add_argument(
        "--improved", action="store_true", help="Use improved feature engineering"
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator(trial=args.trial, improved=args.improved)
    evaluator.evaluate(args.model)
