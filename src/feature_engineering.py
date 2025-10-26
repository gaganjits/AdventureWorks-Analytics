"""
Feature Engineering Module
Creates new features and transforms existing ones for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from datetime import datetime


class FeatureEngineer:
    """Class for feature engineering on AdventureWorks data."""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def create_time_features(self, df, date_column):
        """
        Extract time-based features from a datetime column.

        Features created:
        - Year, Month, Day, DayOfWeek, Quarter
        - Is Weekend, Is Month Start/End
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        df[f'{date_column}_quarter'] = df[date_column].dt.quarter
        df[f'{date_column}_is_weekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)
        df[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start.astype(int)
        df[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end.astype(int)

        return df

    def create_revenue_features(self, df, quantity_col='OrderQty', price_col='UnitPrice'):
        """Create revenue-related features."""
        df = df.copy()

        if quantity_col in df.columns and price_col in df.columns:
            df['total_revenue'] = df[quantity_col] * df[price_col]

            # If discount exists
            if 'UnitPriceDiscount' in df.columns:
                df['discount_amount'] = df['total_revenue'] * df['UnitPriceDiscount']
                df['net_revenue'] = df['total_revenue'] - df['discount_amount']

        return df

    def create_customer_aggregates(self, df, customer_col='CustomerID', agg_features=None):
        """
        Create customer-level aggregate features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        customer_col : str
            Column containing customer ID
        agg_features : dict
            Dictionary mapping column names to aggregation functions
        """
        if agg_features is None:
            agg_features = {
                'total_revenue': ['sum', 'mean', 'std', 'count'],
                'OrderQty': ['sum', 'mean']
            }

        customer_agg = df.groupby(customer_col).agg(agg_features)
        customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns.values]
        customer_agg = customer_agg.reset_index()

        return customer_agg

    def create_recency_features(self, df, date_col, customer_col='CustomerID', reference_date=None):
        """
        Create RFM (Recency, Frequency, Monetary) features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        date_col : str
            Date column name
        customer_col : str
            Customer ID column
        reference_date : datetime
            Reference date for recency calculation (defaults to max date in data)
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        if reference_date is None:
            reference_date = df[date_col].max()

        # Recency: days since last purchase
        recency = df.groupby(customer_col)[date_col].max().reset_index()
        recency['recency_days'] = (reference_date - recency[date_col]).dt.days

        # Frequency: number of purchases
        frequency = df.groupby(customer_col).size().reset_index(name='frequency')

        # Monetary: total spending
        if 'total_revenue' in df.columns:
            monetary = df.groupby(customer_col)['total_revenue'].sum().reset_index()
            monetary.columns = [customer_col, 'monetary_value']
        else:
            monetary = None

        # Merge features
        rfm = recency.merge(frequency, on=customer_col)
        if monetary is not None:
            rfm = rfm.merge(monetary, on=customer_col)

        return rfm

    def encode_categorical(self, df, columns, method='onehot'):
        """
        Encode categorical variables.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            List of column names to encode
        method : str
            'onehot' or 'label'
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'label':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))

            elif method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

        return df

    def scale_features(self, df, columns, method='standard'):
        """
        Scale numerical features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            List of column names to scale
        method : str
            'standard' for StandardScaler
        """
        df = df.copy()

        if method == 'standard':
            for col in columns:
                if col not in df.columns:
                    continue

                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[[col]])
                else:
                    df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])

        return df

    def create_lag_features(self, df, columns, lags=[1, 7, 30], group_col=None):
        """
        Create lag features for time series analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe (should be sorted by date)
        columns : list
            Columns to create lags for
        lags : list
            List of lag periods
        group_col : str
            Column to group by (e.g., CustomerID)
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for lag in lags:
                if group_col:
                    df[f'{col}_lag_{lag}'] = df.groupby(group_col)[col].shift(lag)
                else:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return df

    def create_rolling_features(self, df, columns, windows=[7, 30, 90], group_col=None):
        """
        Create rolling window statistics.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe (should be sorted by date)
        columns : list
            Columns to create rolling features for
        windows : list
            List of window sizes
        group_col : str
            Column to group by
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for window in windows:
                if group_col:
                    df[f'{col}_rolling_mean_{window}'] = df.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    df[f'{col}_rolling_std_{window}'] = df.groupby(group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                else:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window, min_periods=1).std()

        return df


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    print("Feature Engineering Module initialized successfully!")
