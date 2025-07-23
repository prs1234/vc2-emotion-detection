import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Tuple, Any
from sklearn.ensemble import RandomForestClassifier

import os
os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    filename='logs/modelling.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(filepath: str) -> dict:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {filepath}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def load_train_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Train data loaded from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading train data: {e}")
        raise

def get_features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        x_train = df.drop(columns=['label']).values
        y_train = df['label'].values
        return x_train, y_train
    except Exception as e:
        logging.error(f"Error extracting features and labels: {e}")
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> Any:
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("Model training completed")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: Any, filepath: str) -> None:
    try:
        pickle.dump(model, open(filepath, "wb"))
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main() -> None:
    params = load_params("params.yaml")
    n_estimators = params['modelling']['n_estimators']
    max_depth = params['modelling']['max_depth']
    # Use TF-IDF features instead of BOW
    train_data = load_train_data("data/interim/train_tfidf.csv")
    x_train, y_train = get_features_and_labels(train_data)
    model = train_model(x_train, y_train, n_estimators, max_depth)
    save_model(model, "models/random_forest_model.pkl")

if __name__ == "__main__":
    main()