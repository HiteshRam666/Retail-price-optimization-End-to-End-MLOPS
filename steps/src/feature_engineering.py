from abc import ABC, abstractmethod
from typing import List
import pandas as pd 

class FeatureEngineer(ABC):
    """ 
    Abstract Base Class representing the feature Engineer.
    """
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
                """
        Fit and transform the DataFrame.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            columns (List[str]): List of column names to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
                pass

class DateFeatureEngineer(FeatureEngineer):
        
    def __init__(self, date_format: str = "%m-%d-%Y"):
        self.date_format = date_format

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for column in columns:
            df = self.split_data(df, column)
        return df
    
    def _split_date(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df[column] = pd.to_datetime(df[column], format=self.date_format)
        df[f'{column}_year'] = df[column].dt.year
        df[f'{column}_month'] = df[column].dt.month
        return df 