import os
import pickle
import random
from typing import Dict

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger as logging
from pydantic import BaseModel
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_processing import StockData


class Config(BaseModel):
    model: Dict[str, str]
    hyperparameters: Dict


class TemporalCNN(nn.Module):
    def __init__(
        self,
        input_channels,
        lookback,
        out_channels,
        kernel_size,
        dropout,
        num_classes=2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv1d(out_channels, out_channels * 3, kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        new_length = lookback - 2 * (
            kernel_size - 1
        )  # Corrected length after convolution
        self.fc = nn.Linear(new_length * out_channels * 3, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        return self.fc(x)


class TCNNTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, config_path):
        with open(config_path) as file:
            config_data = yaml.safe_load(file)
        self.config = Config(**config_data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_seed(self.config.hyperparameters["random_seed"])

        X_train_scaled, X_val_scaled, self.scaler = self._scale_data(
            X_train, X_val, self.config.hyperparameters["scaler"]
        )

        # Balance class weights
        class_counts = np.bincount(y_train.ravel())
        self.class_weights = torch.tensor(
            len(y_train) / (len(class_counts) * np.maximum(class_counts, 1)),
            dtype=torch.float32,
        ).to(self.device)

        self.num_features = X_train.shape[2]  # Set input_channels dynamically

        # Convert to PyTorch tensors and permute for Conv1d
        self.X_train_scaled = (
            torch.tensor(X_train_scaled, dtype=torch.float32)
            .permute(0, 2, 1)
            .to(self.device)
        )
        self.y_train = torch.tensor(y_train.ravel(), dtype=torch.long).to(self.device)

        self.X_val_scaled = (
            torch.tensor(X_val_scaled, dtype=torch.float32)
            .permute(0, 2, 1)
            .to(self.device)
        )
        self.y_val = torch.tensor(y_val.ravel(), dtype=torch.long).to(self.device)

        self.train_dataset = TensorDataset(self.X_train_scaled, self.y_train)
        self.val_dataset = TensorDataset(self.X_val_scaled, self.y_val)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.hyperparameters["batch_size"],
            shuffle=False,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.hyperparameters["batch_size"],
            shuffle=False,
        )

    def set_seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _scale_data(self, X_train: np.ndarray, X_val: np.ndarray, scaler_name: str):
        """
        Scales data based on the selected scaler.
        """
        if scaler_name == "standard":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Scaler must be 'standard' or 'minmax'.")

        num_train_samples, lookback, num_features = X_train.shape
        num_val_samples, _, _ = X_val.shape

        X_train = X_train.reshape(-1, num_features)  # Flatten lookback dimension
        X_val = X_val.reshape(-1, num_features)

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Reshape back
        X_train_scaled = X_train_scaled.reshape(
            num_train_samples, lookback, num_features
        )
        X_val_scaled = X_val_scaled.reshape(num_val_samples, lookback, num_features)

        return X_train_scaled, X_val_scaled, scaler

    def train_and_evaluate(self, model, optimizer, criterion):
        best_roc = 0
        patience = 30
        counter = 0
        best_model_state = None  # Ensure it's defined before training

        for epoch in range(self.config.hyperparameters["epochs"]):
            model.train()
            total_train_loss = 0.0

            for X_batch, y_batch in self.train_dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_dataloader)

            # Validation
            model.eval()
            total_val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for X_batch, y_batch in self.val_dataloader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_val_loss += loss.item()

                    preds = torch.softmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy()[:, 1])
                    all_labels.extend(y_batch.cpu().numpy())

            avg_val_loss = total_val_loss / len(self.val_dataloader)
            roc = roc_auc_score(all_labels, all_preds)

            logging.info(
                f"Epoch [{epoch+1}/{self.config.hyperparameters['epochs']}] "
                f"- Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val ROC-AUC: {roc:.4f}"
            )

            # Early stopping logic
            if roc > best_roc:
                best_roc = roc
                counter = 0
                best_model_state = model.state_dict()  # Save best model state
            else:
                counter += 1
                if counter >= patience:
                    logging.info(
                        f"Early stopping triggered. Best ROC-AUC: {best_roc:.4f}"
                    )
                    break

        # Load the best model state (handle case where early stopping never triggers)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            logging.warning(
                "No improvement during training. Returning last model state."
            )

        return best_roc, model

    def objective(self, trial):
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", *self.config.hyperparameters["learning_rate_range"]
            ),
            "dropout": trial.suggest_float(
                "dropout", *self.config.hyperparameters["dropout_range"]
            ),
            "conv1_out_channels": trial.suggest_int(
                "conv1_out_channels",
                *self.config.hyperparameters["conv1_out_channels_range"],
            ),
            "kernel_size": trial.suggest_int(
                "kernel_size", *self.config.hyperparameters["kernel_size_range"]
            ),
        }

        model = TemporalCNN(
            self.num_features,
            10,
            params["conv1_out_channels"],
            params["kernel_size"],
            params["dropout"],
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        return self.train_and_evaluate(model, optimizer, criterion)[0]

    def optimize_hyperparameters(self):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=self.config.hyperparameters["random_seed"]
            ),
        )
        study.optimize(self.objective, n_trials=self.config.hyperparameters["trials"])

        best_params = study.best_params
        mlflow.log_params(best_params)

        best_model = TemporalCNN(
            input_channels=self.num_features,
            lookback=10,
            out_channels=best_params["conv1_out_channels"],
            kernel_size=best_params["kernel_size"],
            dropout=best_params["dropout"],
        ).to(self.device)

        optimizer = optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        logging.info("Training of best model started.")

        _, best_model = self.train_and_evaluate(best_model, optimizer, criterion)

        logging.info("Training of best model ended.")

        self.log_and_save(best_model, self.scaler, best_params)

        return best_model

    def log_and_save(self, model, scaler, params):
        os.makedirs("data", exist_ok=True)

        params_filename = f"{self.config.model['name']}_{self.config.model['experiment_id']}_params.pkl"
        params_path = os.path.join("data", params_filename)
        with open(params_path, "wb") as f:
            pickle.dump(params, f)
        mlflow.log_artifact(params_path)

        model_filename = f"{self.config.model['name']}_{self.config.model['experiment_id']}_model.pth"
        model_path = os.path.join("data", model_filename)
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        scaler_filename = f"{self.config.model['name']}_{self.config.model['experiment_id']}_scaler.pkl"
        scaler_path = os.path.join("data", scaler_filename)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path)


def run():
    mlflow.set_experiment("TemporalCNN_Optimization")
    with mlflow.start_run():
        stock_data = StockData(
            ticker="AAPL", start_date="2015-01-01", end_date="2024-01-01"
        )
        train, val, _ = stock_data.partition_data()

        # Initialize lookback and horizon
        lookback = 10
        horizon = 1

        # get windows
        X_train, y_train = stock_data.get_windows(
            train, lookback, horizon, "Extreme_Event"
        )
        X_val, y_val = stock_data.get_windows(val, lookback, horizon, "Extreme_Event")

        trainer = TCNNTrainer(X_train, y_train, X_val, y_val, "data/tcn_config.yaml")
        trainer.optimize_hyperparameters()


if __name__ == "__main__":
    run()
