import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import joblib

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

# Flatten the data: (100, 40, 10) -> (100, 400)
X_flattened = X.reshape(X.shape[0], -1)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flattened)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = make_pipeline(StandardScaler(), SVC(probability=True))
svm_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')