
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create required folders
os.makedirs("model", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/student_data.csv")

# Input features
X = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score",
        "internal_marks"
    ]
]

# Target column
y = df["performance"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_test, predictions))

# Save model and scaler
joblib.dump(model, "model/student_performance_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully!")
