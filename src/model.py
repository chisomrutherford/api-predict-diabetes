from tensorflow import keras

# Replace the problematic imports
Sequential = keras.Sequential
Dense = keras.layers.Dense

def build_model(input_dim):
    model = Sequential([
        Dense(16, activation="relu", input_shape=(input_dim,)),
        Dense(8,  activation="relu"),
        Dense(1,  activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

print ('Code completed')