
import mlflow.sklearn
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Create FastAPI application with only one API endpoint
app = FastAPI(
    title="Digit Prediction API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Load the model included in the Docker image
model = mlflow.sklearn.load_model("/app/model")


# Define the expected input format
class PredictionInput(BaseModel):
    pixels: list[float] = Field(min_length=64, max_length=64)


# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):

    # Convert input into a NumPy array
    input_data = np.array(data.pixels).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Return prediction as JSON
    return {"prediction": int(prediction[0])}
