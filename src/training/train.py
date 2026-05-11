import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import joblib
df = pd.read_csv("data/raw/telco.csv")

# Remove customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove rows with missing values
df.dropna(inplace=True)
# Convert target column to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)


# Features
X = df.drop("Churn", axis=1)

# Target
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1 Score:", f1)

print("Confusion Matrix:")
print(cm)

# Save trained model
joblib.dump(model, "models/churn_model.pkl")

print("Model saved successfully!")
#python src/training/train.py