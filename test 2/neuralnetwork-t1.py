import json
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D
import time

training_data = {}
fname = input("File name: ")
with open(fname, "r") as file:
    training_data: dict = json.load(file)
    file.close()

X = []
y = []
labels = list(training_data.keys())
label_to_index = {label: index for index, label in enumerate(labels)}

# Convert the training data dictionary into X and y arrays
for label, data_sequences in training_data.items():
    for sequence in data_sequences:
        if isinstance(sequence, list) and len(sequence) == 40 and all(len(subseq) == 10 for subseq in sequence):
            X.append(sequence)
            y.append(label_to_index[label])
        else:
            print("skip")
            # print(f"Skipping invalid sequence with shape {np.shape(sequence)} for label {label}")

# Convert lists to NumPy arrays
X = np.array(X)  # Shape should be (num_samples, 40, 6)
y = np.array(y)  # Shape should be (num_samples,)

# Check shapes of X and y
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Normalize the data
X = X / np.max(X)

# Define the input shape
input_shape = (40, 10)

# Create a Sequential model
model = Sequential()

# Add layers to the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(40, 10)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128))
model.add(Dense(len(labels), activation='softmax'))  # Assuming 5 classes for the sig

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X, y, epochs=20, batch_size=10, validation_split=0.2)

# Evaluate the model on the same data (for simplicity; ideally, use a separate validation/test set)
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# # Measure prediction time
# X_test = np.array([[-0.09,-0.03,0.79,2.33,-1.18,-12.05],[-0.06,-0.05,0.77,2.31,-1.18,-12.01],[0.04,-0.05,0.77,2.37,-1.0,-12.01],[0.6,0.12,0.79,2.38,-0.74,-12.0],[1.59,0.18,0.87,2.17,-1.34,-12.83],[2.9,0.33,1.05,2.47,-3.25,-13.68],[3.76,-0.06,1.3,2.73,-4.74,-13.01],[4.32,-0.16,1.48,2.59,-6.58,-10.93],[4.14,-0.4,1.59,2.21,-8.51,-8.36],[3.4,-0.88,1.61,1.27,-8.7,-6.55],[2.55,-0.62,1.23,1.03,-11.05,-4.95],[1.33,-0.38,0.83,0.69,-10.45,-4.55],[0.62,0.33,0.72,0.54,-11.09,-4.65],[0.77,0.1,0.19,0.75,-11.37,-4.74],[1.38,0.24,1.33,2.39,-11.27,-4.45],[1.84,0.14,1.49,0.79,-10.92,-3.19],[1.84,-0.29,1.23,0.39,-11.61,-2.04],[1.98,-0.63,1.03,0.06,-11.45,-1.24],[2.24,-0.29,0.93,-0.3,-11.14,-0.21],[2.54,-0.26,0.88,-0.5,-10.0,1.11],[2.38,-0.19,0.85,-0.07,-10.35,3.6],[2.23,-0.58,0.74,0.02,-9.48,3.81],[2.86,-1.0,0.52,0.44,-9.03,4.94],[2.84,-0.55,0.56,0.63,-8.04,5.66],[2.11,-0.36,0.61,0.64,-6.51,6.29],[2.05,-0.07,0.72,1.14,-5.67,7.24],[2.26,0.23,0.82,0.86,-4.34,7.5],[2.07,0.46,0.94,1.0,-2.86,9.01],[1.83,0.16,0.88,1.13,-3.06,8.86],[1.72,0.31,0.77,0.78,-1.62,7.72],[1.26,0.28,0.68,0.38,-1.25,7.66],[1.18,0.71,0.65,0.31,-0.45,7.61],[0.84,0.24,0.68,-0.34,0.58,8.11],[0.71,0.47,0.58,-0.57,1.1,7.71],[0.33,-0.2,0.71,-0.88,0.95,7.38],[0.15,-0.03,0.71,-0.96,1.62,7.43],[-0.11,-0.19,0.82,-0.71,1.64,7.42],[-0.11,-0.0,0.82,-0.33,1.46,7.57],[-0.25,-0.17,0.88,-0.28,1.12,7.28],[-0.31,-0.14,0.87,-0.18,1.05,7.47]])  # Generate a single test sample
# X_test = X_test / np.max(X_test)  # Normalize the test data
# X_test = np.expand_dims(X_test, axis=0)
# print(X_test.shape)
# import time
# start_time = time.time()
# prediction = model.predict(X_test)
# end_time = time.time()
# prediction_time = end_time - start_time
# print(f"Prediction: {prediction}")
# print(f"Time taken to predict: {prediction_time} seconds")
model.save("./test-model.h5")
print("================================================================")
# Create a Sequential model
model2 = Sequential()

# Add layers to the model
model2.add(LSTM(64, input_shape=input_shape, return_sequences=True))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())

model2.add(LSTM(64))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())

model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(Dense(len(labels), activation='softmax'))  # Number of classes corresponds to unique labels
# Compile the model
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model2.summary()

# Train the model
model2.fit(X, y, epochs=20, batch_size=10, validation_split=0.2)

# Evaluate the model on the same data (for simplicity; ideally, use a separate validation/test set)
loss, accuracy = model2.evaluate(X, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# # Measure prediction time
# X_test = np.array([[-0.09,-0.03,0.79,2.33,-1.18,-12.05],[-0.06,-0.05,0.77,2.31,-1.18,-12.01],[0.04,-0.05,0.77,2.37,-1.0,-12.01],[0.6,0.12,0.79,2.38,-0.74,-12.0],[1.59,0.18,0.87,2.17,-1.34,-12.83],[2.9,0.33,1.05,2.47,-3.25,-13.68],[3.76,-0.06,1.3,2.73,-4.74,-13.01],[4.32,-0.16,1.48,2.59,-6.58,-10.93],[4.14,-0.4,1.59,2.21,-8.51,-8.36],[3.4,-0.88,1.61,1.27,-8.7,-6.55],[2.55,-0.62,1.23,1.03,-11.05,-4.95],[1.33,-0.38,0.83,0.69,-10.45,-4.55],[0.62,0.33,0.72,0.54,-11.09,-4.65],[0.77,0.1,0.19,0.75,-11.37,-4.74],[1.38,0.24,1.33,2.39,-11.27,-4.45],[1.84,0.14,1.49,0.79,-10.92,-3.19],[1.84,-0.29,1.23,0.39,-11.61,-2.04],[1.98,-0.63,1.03,0.06,-11.45,-1.24],[2.24,-0.29,0.93,-0.3,-11.14,-0.21],[2.54,-0.26,0.88,-0.5,-10.0,1.11],[2.38,-0.19,0.85,-0.07,-10.35,3.6],[2.23,-0.58,0.74,0.02,-9.48,3.81],[2.86,-1.0,0.52,0.44,-9.03,4.94],[2.84,-0.55,0.56,0.63,-8.04,5.66],[2.11,-0.36,0.61,0.64,-6.51,6.29],[2.05,-0.07,0.72,1.14,-5.67,7.24],[2.26,0.23,0.82,0.86,-4.34,7.5],[2.07,0.46,0.94,1.0,-2.86,9.01],[1.83,0.16,0.88,1.13,-3.06,8.86],[1.72,0.31,0.77,0.78,-1.62,7.72],[1.26,0.28,0.68,0.38,-1.25,7.66],[1.18,0.71,0.65,0.31,-0.45,7.61],[0.84,0.24,0.68,-0.34,0.58,8.11],[0.71,0.47,0.58,-0.57,1.1,7.71],[0.33,-0.2,0.71,-0.88,0.95,7.38],[0.15,-0.03,0.71,-0.96,1.62,7.43],[-0.11,-0.19,0.82,-0.71,1.64,7.42],[-0.11,-0.0,0.82,-0.33,1.46,7.57],[-0.25,-0.17,0.88,-0.28,1.12,7.28],[-0.31,-0.14,0.87,-0.18,1.05,7.47]])  # Generate a single test sample
# X_test = X_test / np.max(X_test)  # Normalize the test data
# X_test = np.expand_dims(X_test, axis=0)
# print(X_test.shape)
# import time
# start_time = time.time()
# prediction = model.predict(X_test)
# end_time = time.time()
# prediction_time = end_time - start_time
# print(f"Prediction: {prediction}")
# print(f"Time taken to predict: {prediction_time} seconds")
model2.save("./test-model-2.h5")