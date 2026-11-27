


from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from supabase import create_client

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model/student_performance_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Connect to Supabase
supabase = None

try:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        print("Supabase connected successfully!")
    else:
        print("Supabase environment variables missing!")

except Exception as error:
    print("Supabase connection error:", error)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    suggestions = []
    database_message = None

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
                confidence = round(
                    float(max(probabilities)) * 100, 2
                )

            # Academic improvement suggestions
            if attendance < 75:
                suggestions.append(
                    "Improve your class attendance."
                )

            if study_hours < 3:
                suggestions.append(
                    "Try to study at least 3 hours daily."
                )

            if assignment_score < 60:
                suggestions.append(
                    "Focus more on assignment practice."
                )

            if not suggestions:
                suggestions.append(
                    "Keep maintaining your academic performance."
                )

            # Save prediction in Supabase
            if supabase:

                record = {
                    "study_hours": study_hours,
                    "attendance": attendance,
                    "previous_score": previous_score,
                    "assignment_score": assignment_score,
                    "internal_marks": internal_marks,
                    "predicted_performance": prediction
                }

                try:
                    supabase.table(
                        "student_submissions"
                    ).insert(record).execute()

                    database_message = (
                        "Prediction saved successfully!"
                    )

                except Exception as error:
                    print("Database save error:", error)
                    database_message = (
                        "Database Error: " + str(error)
                    )

            else:
                database_message = "Database is not connected."

        except Exception as error:
            prediction = "Error: " + str(error)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        suggestions=suggestions,
        database_message=database_message
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

