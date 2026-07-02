import os
import pickle
import yaml
import logging
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier


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

# -------------------- Load Parameters -------------------- #
def load_params(params_path: str) -> tuple[int, float]:
    """
    Load model hyperparameters from params.yaml.
    """

    try:
        logger.info("Loading parameters from params.yaml...")

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        n_estimators = params["model_building"]["n_estimators"]
        learning_rate = params["model_building"]["learning_rate"]

        logger.info(
            f"Parameters loaded successfully. "
            f"n_estimators={n_estimators}, learning_rate={learning_rate}"
        )

        return n_estimators, learning_rate

    except Exception:
        logger.exception("Failed to load parameters.")
        raise


# -------------------- Load Data -------------------- #
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test feature datasets.
    """

    try:
        logger.info("Loading feature datasets...")

        train_data = pd.read_csv("./data/interim/train_bow.csv")
        test_data = pd.read_csv("./data/interim/test_bow.csv")

        logger.info(f"Train Shape : {train_data.shape}")
        logger.info(f"Test Shape : {test_data.shape}")

        return train_data, test_data

    except Exception:
        logger.exception("Failed to load feature datasets.")
        raise


# -------------------- Split Features -------------------- #
def split_features_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple:
    """
    Split features and target.
    """

    try:
        logger.info("Splitting features and target...")

        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        logger.info("Feature-target split completed.")

        return X_train, X_test, y_train, y_test

    except Exception:
        logger.exception("Error while splitting features and target.")
        raise


# -------------------- Train Model -------------------- #
def train_model(
    X_train,
    y_train,
    n_estimators: int,
    learning_rate: float
):
    """
    Train Gradient Boosting Classifier.
    """

    try:
        logger.info("Training Gradient Boosting model...")

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )

        clf.fit(X_train, y_train)

        logger.info("Model training completed successfully.")

        return clf

    except Exception:
        logger.exception("Error while training the model.")
        raise


# -------------------- Save Model -------------------- #
def save_model(model) -> None:
    """
    Save trained model.
    """

    try:
        logger.info("Saving trained model...")

        os.makedirs("models", exist_ok=True)

        with open("./models/model.pkl", "wb") as file:
            pickle.dump(model, file)

        logger.info("Model saved successfully.")

    except Exception:
        logger.exception("Error while saving model.")
        raise


# -------------------- Main -------------------- #
def main():

    try:

        logger.info("=" * 60)
        logger.info("Model Training Pipeline Started")

        n_estimators, learning_rate = load_params("params.yaml")

        train_data, test_data = load_data()

        X_train, X_test, y_train, y_test = split_features_target(
            train_data,
            test_data
        )

        model = train_model(
            X_train,
            y_train,
            n_estimators,
            learning_rate
        )

        save_model(model)

        logger.info("Model Training Pipeline Completed Successfully.")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Pipeline execution failed.")
        raise


if __name__ == "__main__":
    main()