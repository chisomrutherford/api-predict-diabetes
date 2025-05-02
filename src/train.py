import sys
import os

# Dynamically add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Only one '..'
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_prep import load_data
from src.model import build_model
import joblib

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_data()
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train,
              epochs=50, batch_size=32,
              validation_split=0.1)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")
    model.save("models/diabetes_mlp.h5")
    joblib.dump(scaler, "models/scaler.joblib")

print("Model and scaler saved.")