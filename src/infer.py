import joblib
import numpy as np
from tensorflow import keras

load_model = keras.models.load_model

scaler = joblib.load("models/scaler.joblib")
model  = load_model("models/diabetes_mlp.h5")

def predict(features: list):
    x = np.array(features).reshape(1, -1)
    x = scaler.transform(x)
    prob = float(model.predict(x)[0][0])
    return {"probability": prob, "prediction": int(prob > 0.5)}

print ('Inference ccode completed')