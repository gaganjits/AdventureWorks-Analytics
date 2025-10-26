"""
Model Training Module
Handles training and hyperparameter tuning for various ML models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Class for training machine learning models."""

    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models = {}

    def prepare_data(self, df, target_col, feature_cols=None, test_size=0.2, random_state=42):
        """
        Prepare data for training.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target variable column name
        feature_cols : list
            List of feature column names (if None, uses all except target)
        test_size : float
            Proportion of test set
        random_state : int
            Random seed

        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def train_regression_model(self, X_train, y_train, model_type='random_forest', **kwargs):
        """
        Train a regression model.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        model_type : str
            'linear', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'
        **kwargs : dict
            Model-specific parameters
        """
        if model_type == 'linear':
            model = LinearRegression(**kwargs)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'lightgbm':
            model = lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', -1),
                random_state=kwargs.get('random_state', 42),
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        self.trained_models[model_type] = model

        return model

    def train_classification_model(self, X_train, y_train, model_type='random_forest', **kwargs):
        """
        Train a classification model.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        model_type : str
            'logistic', 'random_forest', 'xgboost', 'lightgbm'
        **kwargs : dict
            Model-specific parameters
        """
        if model_type == 'logistic':
            model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', -1),
                random_state=kwargs.get('random_state', 42),
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        self.trained_models[model_type] = model

        return model

    def evaluate_regression(self, model, X_test, y_test):
        """
        Evaluate a regression model.

        Returns:
        --------
        dict with RMSE, MAE, R2 score
        """
        y_pred = model.predict(X_test)

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        return metrics

    def evaluate_classification(self, model, X_test, y_test):
        """
        Evaluate a classification model.

        Returns:
        --------
        dict with accuracy, precision, recall, F1, ROC-AUC
        """
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # ROC-AUC for binary classification
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            except:
                pass

        return metrics

    def cross_validate(self, model, X, y, cv=5, scoring='r2'):
        """
        Perform cross-validation.

        Parameters:
        -----------
        model : estimator
            Model to evaluate
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        cv : int
            Number of folds
        scoring : str
            Scoring metric

        Returns:
        --------
        dict with mean and std of scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, cv=5, scoring='r2'):
        """
        Perform grid search for hyperparameter tuning.

        Parameters:
        -----------
        model : estimator
            Base model
        param_grid : dict
            Parameter grid for search
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        cv : int
            Number of folds
        scoring : str
            Scoring metric

        Returns:
        --------
        Best model and parameters
        """
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def save_model(self, model, filename, subfolder=''):
        """Save a trained model to disk."""
        save_dir = self.model_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath = save_dir / filename
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")

        return filepath

    def load_model(self, filename, subfolder=''):
        """Load a trained model from disk."""
        filepath = self.model_dir / subfolder / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

        return model

    def get_feature_importance(self, model, feature_names, top_n=10):
        """
        Get feature importance from tree-based models.

        Parameters:
        -----------
        model : estimator
            Trained model with feature_importances_ attribute
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to return

        Returns:
        --------
        pd.DataFrame with features and importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return None

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    print("Model Training Module initialized successfully!")
