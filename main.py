import sys
import os

# Dynamically add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI
from pydantic import BaseModel
from src.infer import predict
from tensorflow.keras.models import load_model

model = load_model("models/diabetes_mlp.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
class Patient(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API"}

@app.post("/predict")
def predict_diabetes(p: Patient):
    features = [
        p.Pregnancies, p.Glucose, p.BloodPressure, p.SkinThickness,
        p.Insulin, p.BMI, p.DiabetesPedigreeFunction, p.Age
    ]
    return predict(features)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
