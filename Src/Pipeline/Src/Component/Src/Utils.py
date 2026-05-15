# src/utils.py
import os
import sys
import dill
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator

from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj: Any) -> None:
    """
    Save a Python object to a file using dill serialization.

    Args:
        file_path: Path where the object will be saved.
        obj: Object to serialize.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save object to {file_path}: {str(e)}")
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Load a Python object from a file using dill serialization.

    Args:
        file_path: Path to the serialized object.

    Returns:
        Deserialized object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load object from {file_path}: {str(e)}")
        raise CustomException(e, sys)


def save_model_metadata(
    model_name: str,
    best_params: Dict,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    file_path: str,
) -> None:
    """
    Save model training metadata to a JSON file.

    Args:
        model_name: Name of the model.
        best_params: Best hyperparameters found.
        train_metrics: Dictionary of training metrics (mae, rmse, r2).
        test_metrics: Dictionary of testing metrics (mae, rmse, r2).
        file_path: Path to save the metadata JSON.
    """
    metadata = {
        "model_name": model_name,
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Model metadata saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model metadata: {str(e)}")
        raise CustomException(e, sys)


def evaluate_regression_model(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary containing MAE, RMSE, and R² score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def tune_and_evaluate_model(
    model: BaseEstimator,
    param_distributions: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[BaseEstimator, Dict[str, float], Dict[str, float], Dict]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV and evaluate the best model.

    Args:
        model: Scikit-learn compatible model.
        param_distributions: Hyperparameter distributions for RandomizedSearchCV.
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        n_iter: Number of parameter settings sampled.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing:
            - Best fitted model
            - Training metrics dict
            - Testing metrics dict
            - Best hyperparameters dict
    """
    try:
        logger.info(f"Starting hyperparameter tuning for {model.__class__.__name__}")

        # If no hyperparameters to tune, just fit and evaluate
        if not param_distributions:
            logger.info(f"No hyperparameters provided. Fitting model as-is.")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_metrics = evaluate_regression_model(y_train, y_train_pred)
            test_metrics = evaluate_regression_model(y_test, y_test_pred)
            return model, train_metrics, test_metrics, {}

        # Randomized search for hyperparameters
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            random_state=random_state,
            verbose=0,  # Set >0 for debugging
        )
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        logger.info(f"Best params for {model.__class__.__name__}: {best_params}")

        # Evaluate on train and test sets
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metrics = evaluate_regression_model(y_train, y_train_pred)
        test_metrics = evaluate_regression_model(y_test, y_test_pred)

        return best_model, train_metrics, test_metrics, best_params

    except Exception as e:
        logger.error(f"Error during model tuning/evaluation: {str(e)}")
        raise CustomException(e, sys)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, BaseEstimator],
    params: Dict[str, Dict],
    n_iter: int = 20,
) -> Tuple[Dict[str, float], Dict[str, BaseEstimator], Dict[str, Dict]]:
    """
    Tune and evaluate multiple models, returning performance scores and best models.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
        models: Dictionary mapping model names to model instances.
        params: Dictionary mapping model names to hyperparameter distributions.
        n_iter: Number of iterations for RandomizedSearchCV.

    Returns:
        Tuple containing:
            - model_report: Dict mapping model name to test R² score.
            - trained_models: Dict mapping model name to fitted best model.
            - best_params_dict: Dict mapping model name to best hyperparameters.
    """
    try:
        model_report: Dict[str, float] = {}
        trained_models: Dict[str, BaseEstimator] = {}
        best_params_dict: Dict[str, Dict] = {}

        for model_name, model in models.items():
            logger.info(f"Processing model: {model_name}")

            param_dist = params.get(model_name, {})

            best_model, train_metrics, test_metrics, best_params = tune_and_evaluate_model(
                model=model,
                param_distributions=param_dist,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_iter=n_iter,
            )

            # Store results
            model_report[model_name] = test_metrics["r2"]
            trained_models[model_name] = best_model
            best_params_dict[model_name] = best_params

            logger.info(
                f"{model_name} - Train R²: {train_metrics['r2']:.4f}, Test R²: {test_metrics['r2']:.4f}"
            )

        return model_report, trained_models, best_params_dict

    except Exception as e:
        logger.error(f"Error in evaluate_models: {str(e)}")
        raise CustomException(e, sys)
