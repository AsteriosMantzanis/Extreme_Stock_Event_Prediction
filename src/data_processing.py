import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, model_validator
from datetime import date
import yfinance as yf
from loguru import logger as logging
from typing import Tuple

class StockData(BaseModel):
    """
    A class for fetching, processing, and partitioning stock data.
    """
    
    ticker: str
    start_date: date = Field(..., description="Date format must be YYYY-MM-DD")
    end_date: date = Field(..., description="Date format must be YYYY-MM-DD")

    @model_validator(mode='before')
    def check_end_date(cls, values):
        """
        Validates that the end date is after the start date.
        
        Args:
            values (dict): Dictionary containing the start_date and end_date.
        
        Raises:
            ValueError: If end_date is not after start_date.
        
        Returns:
            dict: The validated values.
        """
        start_date = values.get('start_date')
        end_date = values.get('end_date')

        if start_date and end_date and end_date <= start_date:
            raise ValueError("end_date must be after start_date")
        return values

    def load_transform(self) -> pd.DataFrame:
        """
        Fetches stock data from Yahoo Finance, calculates daily returns, identifies extreme events,
        and saves the processed data to a CSV file.
        
        Returns:
            pd.DataFrame: Processed stock data with daily returns and extreme event labels.
        """
        logging.info(f"Initializing dataset for ${self.ticker} stock from {self.start_date} to {self.end_date}.")
        
        # Download stock data
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
        if stock_data.empty:
            logging.error(f"No data found for {self.ticker} from {self.start_date} to {self.end_date}.")
            return pd.DataFrame()  # Return empty DataFrame if no data

        # Process Data
        data = pd.DataFrame(stock_data)
        data['Daily_Return'] = data['Adj Close'].pct_change() * 100
        data.columns = data.columns.levels[0].tolist() if isinstance(data.columns, pd.MultiIndex) else data.columns.tolist()
        data = data.dropna(subset=['Daily_Return'])  # Remove missing values (weekends/holidays)
        data['Extreme_Event'] = np.where(data['Daily_Return'].abs() > 2, 1, 0)
        data['Extreme_Event'] = data['Extreme_Event'].shift(-1)  # Shift by one day
        data = data.dropna(subset=['Extreme_Event'])  # Remove rows with missing target values
        data['Extreme_Event'] = data['Extreme_Event'].astype(int)

        # Save Data
        save_path = 'data'
        os.makedirs(save_path, exist_ok=True)
        filename = f"{self.ticker}_{self.start_date.isoformat()}_{self.end_date.isoformat()}.csv"
        file_path = os.path.join(save_path, filename)
        logging.info(f"Saving {filename} to {save_path} folder.")
        data.to_csv(file_path, index=True)

        return data  # Return DataFrame for further use if needed
    
    def partition_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the processed stock data into training, validation, and test sets (70%, 15%, 15%).
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test datasets.
        """
        data = self.load_transform()
        if data.empty:
            logging.error("Dataset is an empty df.")
            return None, None, None  # If no data, return empty sets

        train_size, val_size = int(0.7 * data.shape[0]), int(0.85 * data.shape[0])
        train = data[:train_size]
        val = data[train_size:val_size]
        test = data[val_size:]

        # Log target distribution across splits
        logging.info(f"Train split has {train.Extreme_Event.value_counts()[0]} normal events and {train.Extreme_Event.value_counts()[1]} extreme events.")
        logging.info(f"Validation split has {val.Extreme_Event.value_counts()[0]} normal events and {val.Extreme_Event.value_counts()[1]} extreme events.")
        logging.info(f"Test split has {test.Extreme_Event.value_counts()[0]} normal events and {test.Extreme_Event.value_counts()[1]} extreme events.")
        
        return train, val, test
    
    @staticmethod
    def get_windows(df: pd.DataFrame, lookback: int, horizon: int, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates time-series windows for model training.
        
        Args:
            df (pd.DataFrame): Dataframe containing stock data.
            lookback (int): Number of past time steps to consider.
            horizon (int): Number of future time steps to predict.
            target (str): Target column name.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature windows and corresponding target values.
        """
        if df is None or target is None:
            raise ValueError("Dataframe and target label is required.")

        X, y = [], []
        features = [i for i in df.columns if target not in i]
        for i in range(df.shape[0] - lookback - horizon + 1): # here by including a step we could have overlapping windows
            window_data = df[i:i + lookback + horizon]
            X.append(window_data[features].iloc[:lookback].to_numpy())
            y.append(window_data[target].iloc[lookback:lookback + 1].to_numpy())

        return np.array(X), np.array(y)

    @classmethod
    def save_default(cls):
        """
        Saves a default dataset for Apple (AAPL) stock from 2015-01-01 to 2024-01-01.
        """
        stock = cls(ticker="AAPL", start_date="2015-01-01", end_date="2024-01-01")
        stock.load_transform()

if __name__ == '__main__':
    StockData.save_default()
