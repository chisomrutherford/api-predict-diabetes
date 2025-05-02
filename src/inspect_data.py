import pandas as pd

# Load the data
def inspect_data():
    df = pd.read_csv("data/diabetes.csv")

    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check the shape of the DataFrame (rows, columns)
    print("\nShape of the dataset:")
    print(df.shape)

    # Display column names and data types
    print("\nColumn information:")
    print(df.info())

    # Show basic statistics for numerical columns
    print("\nSummary statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Check for zero values in specific columns
    print("\nZero values in each column:")
    print((df == 0).sum())

if __name__ == "__main__":
    inspect_data()