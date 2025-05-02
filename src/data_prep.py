import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="data/diabetes.csv"):
    df = pd.read_csv(path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        X[col] = X[col].replace(0, X[col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_test),
        y_train, y_test,
        scaler
    )

print ('Code completed')