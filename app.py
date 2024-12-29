from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model, scaler and mapping
try:
    model = load_model("student_recommendation_model.h5")
    scaler = joblib.load("scaler.pkl")

    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)
except Exception as e:
    print(f"Error loading models: {str(e)}")

# Define subject features
features = [
    'Arabic', 'English', 'French', 'Mathematics', 'Science',
    'Religious Studies', 'Computer Science', 'Philosophy', 'Art', 'Social Studies'
]

def validate_scores(scores):
    """Validate that all scores are between 0 and 100"""
    return all(0 <= score <= 100 for score in scores)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        data = request.form
        user_input = [float(data[subject]) for subject in features]

        # Validate scores
        if not validate_scores(user_input):
            return jsonify({
                "error": "All scores must be between 0 and 100"
            })

        # Scale the data
        user_data_scaled = scaler.transform([user_input])

        # Make prediction
        prediction = model.predict(user_data_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Get recommendation and confidence score
        recommendation = label_mapping[str(predicted_class)]
        confidence = float(prediction[0][predicted_class] * 100)

        return jsonify({
            "recommendation": recommendation,
            "confidence": f"{confidence:.2f}%"
        })

    except ValueError as ve:
        return jsonify({
            "error": "Please ensure all fields contain valid numbers"
        })
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(debug=True)
