from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="MLOps Starter API")

model = joblib.load("model.pkl")


class Input(BaseModel):
    feature1: float
    feature2: float


@app.post("/predict")
def predict(data: Input):
    x = [[data.feature1, data.feature2]]
    y = model.predict(x)
    return {"prediction": float(y[0])}
