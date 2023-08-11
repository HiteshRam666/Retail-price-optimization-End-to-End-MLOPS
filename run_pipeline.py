# from pipelines.training_pipeline import training_retail
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

from pipelines.training_pipeline import training_retail
from steps.ingest_data import ingest_data
from steps.process_data import categorical_encoding
# def main():
#     training_retail()

if __name__ == "__main__":
    training_retail()