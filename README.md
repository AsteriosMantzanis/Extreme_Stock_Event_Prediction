# Stock Price Movement Prediction

## Overview

This repository contains a stock price movement prediction pipeline using Random Forest and Temporal CNN models. The goal is to preprocess stock data, train models, evaluate their performance, and explore potential improvements.


## Installation & Setup

### Prerequisites

Ensure you have the following installed:

- Python (>= 3.8)
- [Poetry](https://python-poetry.org/docs/#installation)

### Setting up the environment

#### Clone the repository:

```bash
git clone https://github.com/AsteriosMantzanis/Extreme_Stock_Event_Prediction.git
```
####  cd into project:
```bash
cd Extreme_Stock_Event_Prediction
```
#### Install dependecies using Poetry:
```bash
poetry install
```
This will create a virtual environment and install all required packages.

#### Activate the Poetry virtual environment:
```bash
poetry shell
```
## Running the Scripts

Each script in the src directory is modular and can be executed independently.
### Data Processing
Running the data processing script will save a static csv to the data folder:

```bash
poetry run python .\src\data_processing.py
```
### Random Forest Training

```bash
poetry run python .\src\random_forest.py
```
This will train a random forest model and save its artifacts in the data folder along with the trial number from the rf_config.yaml.

You need to specify the trial number and the model type for the evaluation.

### Random Forest Evaluation

```bash
poetry run python .\src\model_evaluation rf <trial_number>
```
### Temporal CNN Training

```bash
poetry run python .\src\temporal_cnn.py
```
This will train a random forest model and save its artifacts in the data folder along with the trial number from the rf_config.yaml.

You need to specify the trial number and the model type for the evaluation.

### Temporal CNN Evaluation

```bash
poetry run python .\src\model_evaluation tcn <trial_number>
```
### Improvements

```bash
poetry run python .\src\improvements.py
```
This will run a TCN with the proposed improvements.

To evaluate it you must do the following.

### Temporal CNN Improvements Evaluation

Run the same command as the previous TCN evaluation but also add the following argument:

```bash
poetry run python .\src\model_evaluation.py tcn <trial_number> --improved
```
This will get you through training and evaluation of the models.
