import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    filename='logs/features.log',
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

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath).dropna(subset=['content'])
        logging.info(f"Loaded data from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def extract_features_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X = df['content'].values
        y = df['sentiment'].values
        return X, y
    except Exception as e:
        logging.error(f"Error extracting features and labels: {e}")
        raise

def vectorize_data(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Data vectorization completed")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Error vectorizing data: {e}")
        raise

def save_features(X_bow, y, filepath: str) -> None:
    try:
        df = pd.DataFrame(X_bow.toarray())
        df['label'] = y
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logging.info(f"Saved features to {filepath}")
    except Exception as e:
        logging.error(f"Error saving features to {filepath}: {e}")
        raise

def main() -> None:
    params = load_params("params.yaml")
    max_features = params['feature_engineering']['max_features']
    train_data = load_data("data/processed/train.csv")
    test_data = load_data("data/processed/test.csv")
    X_train, y_train = extract_features_and_labels(train_data)
    X_test, y_test = extract_features_and_labels(test_data)
    X_train_bow, X_test_bow, _ = vectorize_data(X_train, X_test, max_features)
    save_features(X_train_bow, y_train, "data/interim/train_bow.csv")
    save_features(X_test_bow, y_test, "data/interim/test_bow.csv")

if __name__ == "__main__":
    main()