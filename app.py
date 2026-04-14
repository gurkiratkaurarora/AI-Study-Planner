from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Study plan logic
def generate_plan(pred):
    if pred == 0:
        return {
            "level": "High Risk",
            "study_hours": "5-6 hours/day",
            "focus": "Weak subjects + revision",
            "tip": "Study daily in small sessions"
        }
    elif pred == 1:
        return {
            "level": "Medium",
            "study_hours": "3-4 hours/day",
            "focus": "Balanced study",
            "tip": "Practice weak areas"
        }
    else:
        return {
            "level": "Low Risk",
            "study_hours": "2-3 hours/day",
            "focus": "Consistency",
            "tip": "Maintain performance"
        }

from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = [
            data["sem1_approved"],
            data["sem2_approved"],
            data["sem1_enrolled"],
            data["sem2_enrolled"],
            data["sem1_evaluations"],
            data["sem2_evaluations"]
        ]

        final = np.array(features).reshape(1, -1)
        final_scaled = scaler.transform(final)

        pred = model.predict(final_scaled)[0]

        plan = generate_plan(pred)

        return jsonify({
            "prediction": int(pred),
            "study_plan": plan
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
