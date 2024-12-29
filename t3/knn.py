import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
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


for label, data_sequences in training_data.items():
    for sequence in data_sequences:
        if isinstance(sequence, list) and len(sequence) == 40 and all(len(subseq) == 10 for subseq in sequence):
            X.append(sequence)
            y.append(label_to_index[label])
        else:
            print("skip")
            # print(f"Skipping invalid sequence with shape {np.shape(sequence)} for label {label}")


X = np.array(X)  # Shape should be (num_samples, 40, 6)
y = np.array(y)  # Shape should be (num_samples,)


X_flattened = X.reshape(X.shape[0], -1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flattened)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print(predictions)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, predictions))

joblib.dump(knn, 'knn.pkl')
joblib.dump(scaler, 'scaler_knn.pkl')