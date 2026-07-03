import json
import logging
import pickle

import pandas as pd
      

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import logging
import os

# Create logs folder
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "project.log")

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),   # Save logs in file
        logging.StreamHandler()          # Show logs in terminal
    ]
)

logger = logging.getLogger(__name__)

# Load Model
def load_model(model_path: str):
    """
    Load the trained model.
    """

    try:
        logger.info("Loading trained model...")

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded successfully.")

        return model

    except Exception:
        logger.exception("Failed to load model.")
        raise


# Load Test Data
def load_data() -> pd.DataFrame:
    """
    Load feature-engineered test dataset.
    """

    try:
        logger.info("Loading test dataset...")

        test_data = pd.read_csv("./data/interim/test_tfidf.csv")

        logger.info(f"Test dataset loaded. Shape: {test_data.shape}")

        return test_data

    except Exception:
        logger.exception("Failed to load test dataset.")
        raise


# Split Features and Target
def split_features_target(test_df: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split test dataset into features and target.
    """

    try:
        logger.info("Splitting features and target...")

        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        logger.info("Features and target separated successfully.")

        return X_test, y_test

    except Exception:
        logger.exception("Failed while splitting features and target.")
        raise


# Evaluate Model
def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the trained model.
    """

    try:

        logger.info("Evaluating model...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.info("Model evaluation completed successfully.")

        return metrics_dict

    except Exception:
        logger.exception("Model evaluation failed.")
        raise


# Save Metrics
def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Save evaluation metrics to JSON.
    """

    try:

        logger.info("Saving metrics...")

        with open(output_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.info(f"Metrics saved successfully at {output_path}")

    except Exception:
        logger.exception("Failed to save metrics.")
        raise


# Main Function
def main():

    try:

        logger.info("=" * 60)
        logger.info("Model Evaluation Pipeline Started")

        model = load_model("./models/model.pkl")

        test_data = load_data()

        X_test, y_test = split_features_target(test_data)

        metrics = evaluate_model(model, X_test, y_test)

        save_metrics(metrics, "./reports/metrics.json")

        logger.info(f"Evaluation Metrics: {metrics}")

        logger.info("Model Evaluation Pipeline Completed Successfully.")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Pipeline execution failed.")
        raise


if __name__ == "__main__":
    main()