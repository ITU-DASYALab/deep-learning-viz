import os
from time import time

start = time()

# Retrieve authentication envvars
with open("../keys") as f:
    os.environ["MLFLOW_TRACKING_USERNAME"] = f.readline().strip()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = f.readline().strip()
    os.environ["AWS_ACCESS_KEY_ID"] = f.readline().strip()
    os.environ["AWS_SECRET_ACCESS_KEY"] = f.readline().strip()
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f.readline().strip()


import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from pathlib import Path

# Establish connection
mlflow.set_tracking_uri("SPECIFY_URL")
client = MlflowClient()

# print(client.get_experiment(6).artifact_location)
