import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import time

# Define the input shape
input_shape = (40, 6)

# Create a Sequential model
model = Sequential()

# Add layers to the model
model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Assuming 10 different ASL gestures

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Dummy data for training
# Generate random data for 100 samples
X_train = np.random.rand(100, 40, 6)
y_train = np.random.randint(10, size=(100,))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=10)

# Evaluate the model on the same dummy data (normally, you would use a separate validation/test set)
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Measure prediction time
X_test = np.random.rand(1, 40, 6)  # Generate a single test sample
start_time = time.time()
prediction = model.predict(X_test)
end_time = time.time()
prediction_time = end_time - start_time
print(f"Prediction: {prediction}")
print(f"Time taken to predict: {prediction_time} seconds")
