import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    filename='logs/data_ingestion.log',  # <--- Save logs to file
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

def load_dataset(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logging.info(f"Dataset loaded from {url}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Data preprocessing completed")
        return final_df
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train and test sets with test_size={test_size}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, dir_path: str) -> None:
    try:
        os.makedirs(dir_path, exist_ok=True)
        train_data.to_csv(os.path.join(dir_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(dir_path, "test.csv"), index=False)
        logging.info(f"Train and test data saved to {dir_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main() -> None:
    params = load_params("params.yaml")
    test_size = params['data_ingestion']['test_size']
    df = load_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = preprocess_data(df)
    train_data, test_data = split_data(final_df, test_size)
    save_data(train_data, test_data, "data/raw")

if __name__ == "__main__":
    main()
    main()
