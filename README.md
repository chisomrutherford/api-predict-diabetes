# Diabetes Prediction API

This repository contains a FastAPI-based web application for predicting the likelihood of diabetes in patients using a machine learning model. The application uses a TensorFlow-based neural network model trained on the Pima Indians Diabetes Dataset.

## Features

- **API Endpoints**:
  - `GET /`: Welcome message.
  - `POST /predict`: Predicts the likelihood of diabetes based on patient data.
- **Machine Learning**:
  - A neural network model built with TensorFlow/Keras.
  - Data preprocessing using pandas and scikit-learn.
- **Dockerized**: Easily deployable using Docker.

## Project Structure

```
Dockerfile
README.md
requirements.txt
app/
    main.py
    __pycache__/
data/
    diabetes.csv
models/
    diabetes_mlp.h5
    scaler.joblib
src/
    __init__.py
    data_prep.py
    infer.py
    model.py
    train.py
    __pycache__/
```

### Key Files

- **`app/main.py`**: FastAPI application with endpoints for predictions.
- **`src/data_prep.py`**: Prepares and preprocesses the dataset.
- **`src/model.py`**: Defines the neural network architecture.
- **`src/train.py`**: Trains the model and saves it.
- **`src/infer.py`**: Loads the trained model and performs predictions.
- **`data/diabetes.csv`**: Dataset used for training and testing.
- **`models/diabetes_mlp.h5`**: Trained neural network model.
- **`models/scaler.joblib`**: Scaler for data normalization.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. Access the API at `http://localhost:8000`.

## Using Docker

1. Build the Docker image:
   ```bash
   docker build -t diabetes-prediction-api .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 diabetes-prediction-api
   ```

## API Endpoints

### `GET /`
Returns a welcome message.

### `POST /predict`
Predicts the likelihood of diabetes based on patient data.

#### Request Body
```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 80,
  "SkinThickness": 25,
  "Insulin": 100,
  "BMI": 24.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 30
}
```

#### Response
```json
{
  "probability": 0.85,
  "prediction": 1
}
```

## Dataset

The dataset used is the Pima Indians Diabetes Dataset, which contains the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target variable)

