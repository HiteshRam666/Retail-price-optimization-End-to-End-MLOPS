from typing import List, Tuple
import logging
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml import logger
from zenml.logger import get_logger

from materializers.custom_materializer import(
    ListMaterializer,
    SKLearnModelMaterializer,
    StatsModelMaterializer,
)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for"
        "this example to work."
    )

@step(experiment_tracker="mlflow_tracker",
      settings={"experiment_tracker.mlflow":{"experiment_name": "test_name"}},  
      enable_cache=False, output_materializers=[SKLearnModelMaterializer, ListMaterializer])

def sklearn_train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"]
) -> Tuple[
    Annotated[LinearRegression, "model"],
    Annotated[List[str], "predictors"],
]:
    """"
    Trains a linear Regression model and outputs the summary.
    """
    try:
        mlflow.end_run() # End any exiting run
        with mlflow.start_run() as run:
            mlflow.sklearn.autolog() # Automatically logs all sklearn parameters, metrics and models
            model = LinearRegression()
            model.fit(X_train, y_train) # Train the model 
            predictors = X_train.columns.tolist() # Considering all columns in X_train as predictors
            return model, predictors
    except Exception as e:
        logger.error(e)
        raise e
