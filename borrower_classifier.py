import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Define columns to load
usecols = ["annual_inc", "dti", "fico_range_low", "loan_amnt", "loan_status"]
dtypes = {
    "annual_inc": "float32",
    "dti": "float32",
    "fico_range_low": "Int32",  # Nullable integer type
    "loan_amnt": "float32",
    "loan_status": "object"
}

# Load dataset
df = pd.read_csv("loan_data.csv", usecols=usecols, dtype=dtypes)

# Clean dataset
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].dropna()

# Map target: 0 = good (Fully Paid), 1 = bad (Charged Off)
df["loan_status"] = df["loan_status"].map({"Fully Paid": 0, "Charged Off": 1})

# Split features and target
features = ["annual_inc", "dti", "fico_range_low", "loan_amnt"]
X = df[features]
y = df["loan_status"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and scaler
joblib.dump(model, "borrower_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Prediction function for new data
def predict_borrower(annual_inc, dti, fico_score, loan_amnt):
    # Create a DataFrame with feature names to match training data
    input_data = pd.DataFrame(
        [[annual_inc, dti, fico_score, loan_amnt]],
        columns=["annual_inc", "dti", "fico_range_low", "loan_amnt"]
    )
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of "bad"
    return "Bad" if prediction == 1 else "Good", probability

# Test prediction
if __name__ == "__main__":
    result, prob = predict_borrower(50000, 20, 700, 10000)
    print(f"Prediction: {result}, Probability of Default: {prob:.2f}")