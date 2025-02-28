import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("borrower_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names (consistent with training)
FEATURES = ["annual_inc", "dti", "fico_range_low", "loan_amnt"]

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract input values
        annual_inc = float(data["annual_inc"])
        dti = float(data["dti"])
        fico_score = int(data["fico_score"])
        loan_amnt = float(data["loan_amnt"])
        
        # Create DataFrame with feature names
        input_data = pd.DataFrame(
            [[annual_inc, dti, fico_score, loan_amnt]],
            columns=FEATURES
        )
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # Probability of "bad"
        
        # Return result
        result = "Bad" if prediction == 1 else "Good"
        return jsonify({
            "prediction": result,
            "probability_of_default": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Health check endpoint
@app.route("/")
def home():
    return "Borrower Classifier API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)