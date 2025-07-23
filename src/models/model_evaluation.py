import logging
import pandas as pd
import pickle
import json
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os

os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    filename='logs/model_evaluation.log',  # <--- Save logs to file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(filepath: str) -> Any:
    try:
        model = pickle.load(open(filepath, "rb"))
        logging.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_test_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Test data loaded from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

def evaluate_model(model: Any, X_test, y_test) -> Dict[str, float]:
    try:
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation completed")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], filepath: str) -> None:
    try:
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def main() -> None:
    model = load_model("models/random_forest_model.pkl")
    # Use TF-IDF features instead of BOW
    test_data = load_test_data("data/interim/test_tfidf.csv")
    X_test = test_data.drop(columns=['label']).values
    y_test = test_data['label'].values
    metrics_dict = evaluate_model(model, X_test, y_test)
    save_metrics(metrics_dict, "reports/metrics.json")

if __name__ == "__main__":
    main()