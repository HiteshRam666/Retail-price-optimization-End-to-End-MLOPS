import logging
import pandas as pd 
from zenml import step 
from steps.src.data_loader import DataLoader

@step(enable_cache=False)
def ingest_data(
    table_name: str,
) -> pd.DataFrame:
    """
    Read data from sql database and return a pandas dataframe.

    Args:
        table_name: Name of the table to read from.
    """
    try:
        data_loader = DataLoader('postgresql://postgres:hiteshram@localhost:5432/test1')
        data_loader.load_data(table_name)
        df = data_loader.get_data()
        logging.info(f"Successfully read data from {table_name}.")
    except Exception as e:
        logging.error(f'Error while reading data from {table_name}.')
        raise e