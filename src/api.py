
from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel
from typing import List

class SensorData(BaseModel):
    data: List[List[float]]

app = FastAPI()

# Load the model
model_name = "predictive_maintenance_model_tuned"
model_version = "latest"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: SensorData):
    """
    Accepts sensor data and returns RUL predictions.
    """
    df = pd.DataFrame(data.data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
