"""
Model Evaluation Module
Provides comprehensive model evaluation and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Class for comprehensive model evaluation and visualization."""

    def __init__(self, output_dir='outputs/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        """
        Plot confusion matrix for classification models.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list
            Class labels
        save_path : str
            Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {self.output_dir / save_path}")

        plt.show()

    def plot_roc_curve(self, y_true, y_proba, save_path=None):
        """
        Plot ROC curve for binary classification.

        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_proba : array-like
            Predicted probabilities for positive class
        save_path : str
            Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {self.output_dir / save_path}")

        plt.show()

        return roc_auc

    def plot_precision_recall_curve(self, y_true, y_proba, save_path=None):
        """
        Plot Precision-Recall curve.

        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_proba : array-like
            Predicted probabilities for positive class
        save_path : str
            Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {self.output_dir / save_path}")

        plt.show()

    def plot_feature_importance(self, importance_df, top_n=15, save_path=None):
        """
        Plot feature importance.

        Parameters:
        -----------
        importance_df : pd.DataFrame
            DataFrame with 'feature' and 'importance' columns
        top_n : int
            Number of top features to plot
        save_path : str
            Path to save the plot
        """
        plot_df = importance_df.head(top_n).sort_values('importance')

        plt.figure(figsize=(10, 8))
        plt.barh(plot_df['feature'], plot_df['importance'], color='steelblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {self.output_dir / save_path}")

        plt.show()

    def plot_actual_vs_predicted(self, y_true, y_pred, save_path=None):
        """
        Plot actual vs predicted values for regression.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        save_path : str
            Path to save the plot
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Actual vs Predicted plot saved to {self.output_dir / save_path}")

        plt.show()

    def plot_residuals(self, y_true, y_pred, save_path=None):
        """
        Plot residuals for regression models.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        save_path : str
            Path to save the plot
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')

        # Residual distribution
        axes[1].hist(residuals, bins=30, edgecolor='k', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Residual plot saved to {self.output_dir / save_path}")

        plt.show()

    def plot_learning_curve(self, train_scores, val_scores, train_sizes=None, save_path=None):
        """
        Plot learning curve.

        Parameters:
        -----------
        train_scores : array-like
            Training scores
        val_scores : array-like
            Validation scores
        train_sizes : array-like
            Training set sizes
        save_path : str
            Path to save the plot
        """
        if train_sizes is None:
            train_sizes = range(1, len(train_scores) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', color='r', label='Training Score')
        plt.plot(train_sizes, val_scores, 'o-', color='g', label='Validation Score')
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curve saved to {self.output_dir / save_path}")

        plt.show()

    def print_regression_metrics(self, y_true, y_pred, model_name='Model'):
        """
        Print comprehensive regression metrics.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print(f"\n{'='*50}")
        print(f"{model_name} - Regression Metrics")
        print(f"{'='*50}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"MAE:   {mae:.4f}")
        print(f"R²:    {r2:.4f}")
        print(f"MAPE:  {mape:.2f}%")
        print(f"{'='*50}\n")

        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    def print_classification_metrics(self, y_true, y_pred, model_name='Model', labels=None):
        """
        Print comprehensive classification metrics.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name of the model
        labels : list
            Class labels
        """
        print(f"\n{'='*50}")
        print(f"{model_name} - Classification Metrics")
        print(f"{'='*50}")
        print(classification_report(y_true, y_pred, target_names=labels))
        print(f"{'='*50}\n")

    def compare_models(self, results_dict, metric='r2', save_path=None):
        """
        Compare multiple models.

        Parameters:
        -----------
        results_dict : dict
            Dictionary with model names as keys and metrics dict as values
        metric : str
            Metric to compare
        save_path : str
            Path to save the plot
        """
        models = list(results_dict.keys())
        scores = [results_dict[model][metric] for model in models]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(models, scores, color='steelblue')
        plt.xlabel(metric.upper())
        plt.ylabel('Model')
        plt.title(f'Model Comparison - {metric.upper()}')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{scores[i]:.4f}',
                    ha='left', va='center', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {self.output_dir / save_path}")

        plt.show()

    def create_prediction_report(self, y_true, y_pred, model_name, task_type='regression'):
        """
        Create a comprehensive prediction report.

        Parameters:
        -----------
        y_true : array-like
            True values/labels
        y_pred : array-like
            Predicted values/labels
        model_name : str
            Name of the model
        task_type : str
            'regression' or 'classification'

        Returns:
        --------
        dict with all metrics and report
        """
        report = {'model_name': model_name, 'task_type': task_type}

        if task_type == 'regression':
            metrics = self.print_regression_metrics(y_true, y_pred, model_name)
            report['metrics'] = metrics

            # Create visualizations
            self.plot_actual_vs_predicted(y_true, y_pred,
                                         save_path=f'{model_name}_actual_vs_pred.png')
            self.plot_residuals(y_true, y_pred,
                               save_path=f'{model_name}_residuals.png')

        elif task_type == 'classification':
            self.print_classification_metrics(y_true, y_pred, model_name)

            # Create visualizations
            self.plot_confusion_matrix(y_true, y_pred,
                                      save_path=f'{model_name}_confusion_matrix.png')

        return report


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    print("Model Evaluation Module initialized successfully!")
