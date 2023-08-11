import pandas as pd 
from sqlalchemy import create_engine, exc

class DataLoader:
    """ Class to load data from a sql database."""

    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.data = None
    
    def load_data(self, table_name: str) -> pd.DataFrame:
        """
        Loads the data from the specified table into DF, which is stored as an instance variable self.data.

        Args:
            table_name: Name of the table to read from.
        
        Returns: 
            pd.DataFrame: Data from the table
        """
        query = "SELECT * FROM" + table_name
        try:
            self.data = pd.read_sql(query, self.engine)
            return self.data
        except exc.SQLAlchemyError as e:
            raise e

    def get_data(self) -> pd.DataFrame:
        """
        Returns the data that was loaded into the class instance

        Returns:
            pd.DataFrame: Data from the table
        """ 
        if self.data is not None:
            return self.data
        else:
            raise ValueError("No data loaded yet. Please run load_data() first.")