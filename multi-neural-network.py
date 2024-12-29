structure = [
    [".\\training-data-resting.json", ".\\training-data-hello.json", ".\\training-data-nice to meet you.json"], 
    [".\\training-data-nice to meet you.json", ".\\training-data-his.json", ".\\training-data-my.json"], 
    [".\\training-data-hello.json", ".\\training-data-my.json", ".\\training-data-name is.json"], 
    [".\\training-data-resting.json", ".\\training-data-his.json", ".\\training-data-name is.json"]
]
model_dir = ".\\mnn-test-1\\"

import json
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Dropout
import time, pathlib
jsn = {}
i = 1
for eachnn in structure:
    training_data = {}
    for l in eachnn:
        with open(l, "r") as fw:
            w = json.load(fw)
            for u in w.keys():
                training_data[u] = w[u]
            fw.close()
    
    X = []
    y = []
    labels = list(training_data.keys())
    label_to_index = {label: index for index, label in enumerate(labels)}

    #Convert the training data dictionary into X and y arrays
    for label, data_sequences in training_data.items():
        for sequence in data_sequences:
            if isinstance(sequence, list) and len(sequence) == 40 and all(len(subseq) == 12 for subseq in sequence):
                X.append(sequence)
                y.append(label_to_index[label])
            else:
                print(f"Skipping invalid sequence with shape {np.shape(sequence)} for label {label}")

                # Convert lists to NumPy arrays
    X = np.array(X)  # Shape should be (num_samples, 40, 6)
    y = np.array(y)  # Shape should be (num_samples,)

    # Check shapes of X and y
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Normalize the data
    X = X / np.max(X)

    # Define the input shape
    input_shape = (40, 12)

    # Create a Sequential model
    model = Sequential()

    # Add layers to the model
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))  # Adding dropout to prevent overfitting
    model.add(LSTM(32))
    model.add(Dropout(0.2))  # Adding dropout to prevent overfitting
    model.add(Dense(16, activation='relu'))
    model.add(Dense(len(labels), activation='softmax'))  # Number of classes corresponds to unique labels

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Train the model
    model.fit(X, y, epochs=20, batch_size=10, validation_split=0.2)

    # Evaluate the model on the same data (for simplicity; ideally, use a separate validation/test set)
    loss, accuracy = model.evaluate(X, y)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save(model_dir+f"{i}.h5")
    jsn[f"{i}"] = eachnn
    i += 1

with open(model_dir+"info.json", "w") as infofile:
    json.dump(jsn, infofile)