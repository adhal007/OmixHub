
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import src.CustomLogger.custom_logger
logger = src.CustomLogger.custom_logger.CustomLogger()

class DataQualityChecker:
    def __init__(self, data:pd.DataFrame) -> None:
        """
        Initialize the DataQualityChecker with a DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.
        """
        self.data = data
        self.logger = logger.custlogger(loglevel='DEBUG')
        self.logger.debug("Initialized Quality Checker class")
        if self.data is None:
            raise ValueError("No data provided")
        elif self.data.shape[0] == 0:
            raise ValueError("No data rows provided")
        elif self.data.shape[1] == 0:
            raise ValueError("No data columns provided")

    def check_class_imbalance(self, target_column):
        """
        Check if there is class imbalance in the data.

        Args:
            target_column (str): The name of the target column.

        Returns:
            bool: True if there is class imbalance, False otherwise.
        """
        class_counts = self.data[target_column].value_counts()
        imbalance = (class_counts.max() / class_counts.min()) > 10
        return imbalance

    def check_missing_data(self):
        """
        Check if there is any missing data in the DataFrame.

        Returns:
            bool: True if there is missing data, False otherwise.
        """
        missing_data = self.data.isnull().sum().sum() > 0
        return missing_data

    def check_duplicate_data(self):
        """
        Check if there is any duplicate data in the DataFrame.

        Returns:
            bool: True if there is duplicate data, False otherwise.
        """
        duplicates = self.data.duplicated().any()
        return duplicates

    def check_outlier_data(self, numerical_columns):
        """
        Check if there is any outlier data in the numerical columns of the DataFrame.

        Args:
            numerical_columns (list): List of numerical column names.

        Returns:
            bool: True if there are outliers, False otherwise.
        """
        outliers = False
        for col in numerical_columns:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers |= ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).any()
        return outliers

    def split_data(self, data, target_column, test_size=0.2, validation_size=0.2, random_state=None):
        """
        Split the data into training, test, and validation cohorts.

        Args:
            target_column (str): The name of the target column.
            test_size (float): The proportion of data to include in the test split.
            validation_size (float): The proportion of data to include in the validation split.
            random_state (int or None): Seed for random number generation.

        Returns:
            dict: A dictionary containing the splits - 'train', 'test', and 'validation'.
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + validation_size), random_state=random_state)
        X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=validation_size/(test_size + validation_size), random_state=random_state)
        splits = {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'validation': (X_validation, y_validation)
        }
        return splits

    def check_data_leakage(self, train, test, validation):
        """
        Check for data leakage between training, test, and validation cohorts.

        Args:
            train (pd.DataFrame): The training data.
            test (pd.DataFrame): The test data.
            validation (pd.DataFrame): The validation data.

        Returns:
            bool: True if there is data leakage, False otherwise.
        """
        train_ids = set(train.index)
        test_ids = set(test.index)
        validation_ids = set(validation.index)

        leakage = (len(train_ids.intersection(test_ids)) > 0) or (len(train_ids.intersection(validation_ids)) > 0)
        return leakage

    def check_data_drift(self, train, test, validation, numerical_columns):
        """
        Check for data drift between training, test, and validation cohorts.

        Args:
            train (pd.DataFrame): The training data.
            test (pd.DataFrame): The test data.
            validation (pd.DataFrame): The validation data.
            numerical_columns (list): List of numerical column names.

        Returns:
            bool: True if there is data drift, False otherwise.
        """
        train_stats = train[numerical_columns].describe()
        test_stats = test[numerical_columns].describe()
        validation_stats = validation[numerical_columns].describe()

        drift = False
        for col in numerical_columns:
            train_mean = train_stats[col]['mean']
            test_mean = test_stats[col]['mean']
            validation_mean = validation_stats[col]['mean']

            train_std = train_stats[col]['std']
            test_std = test_stats[col]['std']
            validation_std = validation_stats[col]['std']

            drift |= ((np.abs(train_mean - test_mean) > 2 * (train_std + test_std)) or
                      (np.abs(train_mean - validation_mean) > 2 * (train_std + validation_std)))
        return drift

    def check_data_skew(self, train, test, validation, numerical_columns):
        """
        Check for data skew between training, test, and validation cohorts.

        Args:
            train (pd.DataFrame): The training data.
            test (pd.DataFrame): The test data.
            validation (pd.DataFrame): The validation data.
            numerical_columns (list): List of numerical column names.

        Returns:
            bool: True if there is data skew, False otherwise.
        """
        train_skew = train[numerical_columns].skew()
        test_skew = test[numerical_columns].skew()
        validation_skew = validation[numerical_columns].skew()

        skew = (train_skew - test_skew).abs().max() > 0.5 or (train_skew - validation_skew).abs().max() > 0.5
        return skew

    def check_data_bias(self, train, test, validation, target_column):
        """
        Check for data bias between training, test, and validation cohorts.

        Args:
            train (pd.DataFrame): The training data.
            test (pd.DataFrame): The test data.
            validation (pd.DataFrame): The validation data.
            target_column (str): The name of the target column.

        Returns:
            bool: True if there is data bias, False otherwise.
        """
        train_bias = train[target_column].value_counts(normalize=True)
        test_bias = test[target_column].value_counts(normalize=True)
        validation_bias = validation[target_column].value_counts(normalize=True)

        bias = (train_bias - test_bias).abs().max() > 0.1 or (train_bias - validation_bias).abs().max() > 0.1
        return bias

    def check_data_noise(self, train, test, validation, numerical_columns):
        """
        Check for data noise between training, test, and validation cohorts.

        Args:
            train (pd.DataFrame): The training data.
            test (pd.DataFrame): The test data.
            validation (pd.DataFrame): The validation data.
            numerical_columns (list): List of numerical column names.

        Returns:
            bool: True if there is data noise, False otherwise.
        """
        noise = False
        for col in numerical_columns:
            train_std = train[col].std()
            test_std = test[col].std()
            validation_std = validation[col].std()

            noise |= ((np.abs(train_std - test_std) > 0.5 * train_std) or
                      (np.abs(train_std - validation_std) > 0.5 * train_std))
        return noise
    

# Example usage:
# df = pd.read_csv('your_data.csv')
# checker = DataQualityChecker(df)
# imbalance = checker.check_class_imbalance('target_column')
# missing_data = checker.check_missing_data()
# duplicate_data = checker.check_duplicate_data()
# outlier_data = checker.check_outlier_data(['numerical_column1', 'numerical_column2'])
# splits = checker.split_data('target_column')
# leakage = checker.check_data_leakage(*splits.values())
# drift = checker.check_data_drift(*splits.values(), numerical_columns=['numerical_column1', 'numerical_column2'])
# skew = checker.check_data_skew(*splits.values(), numerical_columns=['numerical_column1', 'numerical_column2'])
# bias = checker.check_data_bias(*splits.values(), target_column='target_column')
# noise = checker.check_data_noise(*splits.values(), numerical_columns=['numerical_column1', 'numerical_column2'])