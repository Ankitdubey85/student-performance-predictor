
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("model/student_performance_model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    suggestions = []

    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            previous_score = float(request.form["previous_score"])
            assignment_score = float(request.form["assignment_score"])
            internal_marks = float(request.form["internal_marks"])

            input_data = np.array([[
                study_hours,
                attendance,
                previous_score,
                assignment_score,
                internal_marks
            ]])

            scaled_data = scaler.transform(input_data)
            result = model.predict(scaled_data)[0]

            prediction = str(result)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled_data)[0]
                confidence = round(float(max(probabilities)) * 100, 2)

            if attendance < 75:
                suggestions.append("Improve your class attendance.")

            if study_hours < 3:
                suggestions.append("Try to study at least 3 hours daily.")

            if assignment_score < 60:
                suggestions.append("Focus more on assignment practice.")

            if not suggestions:
                suggestions.append("Keep maintaining your academic performance.")

        except Exception as error:
            prediction = "Error: " + str(error)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        suggestions=suggestions
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
