"""
Data Preprocessing Module
Handles loading, cleaning, and initial processing of raw data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataPreprocessor:
    """Class for preprocessing AdventureWorks data."""

    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path('data/processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename):
        """Load a CSV file from the raw data directory."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath)

    def handle_missing_values(self, df, strategy='drop', threshold=0.5):
        """
        Handle missing values in the dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            'drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_forward'
        threshold : float
            Proportion of missing values above which to drop column
        """
        # Drop columns with too many missing values
        missing_prop = df.isnull().sum() / len(df)
        cols_to_drop = missing_prop[missing_prop > threshold].index
        df = df.drop(columns=cols_to_drop)

        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'fill_mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'fill_median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'fill_mode':
            df = df.fillna(df.mode().iloc[0])
        elif strategy == 'fill_forward':
            df = df.fillna(method='ffill')

        return df

    def convert_date_columns(self, df, date_columns):
        """Convert specified columns to datetime format."""
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def remove_duplicates(self, df, subset=None):
        """Remove duplicate rows."""
        return df.drop_duplicates(subset=subset)

    def save_processed_data(self, df, filename):
        """Save processed data to the processed directory."""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved processed data to {filepath}")
        return filepath


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    print("Data Preprocessing Module initialized successfully!")
