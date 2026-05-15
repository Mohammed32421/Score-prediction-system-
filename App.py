
# app.py
import os
from pathlib import Path

from flask import Flask, request, render_template, jsonify

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logger

# Configuration from environment variables
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/model.pkl"))
PREPROCESSOR_PATH = Path(os.environ.get("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl"))

# Initialize Flask app
app = Flask(__name__)

# Global prediction pipeline (loads once at startup)
try:
    predict_pipeline = PredictPipeline(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH
    )
    logger.info("Prediction pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize prediction pipeline: {str(e)}")
    predict_pipeline = None


@app.route("/")
def index() -> str:
    """Landing page."""
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint() -> str:
    """Handle prediction form."""
    if predict_pipeline is None:
        return render_template(
            "home.html",
            results="System not ready. Please check server logs."
        )

    if request.method == "GET":
        return render_template("home.html")

    try:
        # Extract and validate form data
        gender = request.form.get("gender")
        ethnicity = request.form.get("ethnicity")
        parental_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_course = request.form.get("test_preparation_course")
        reading_score = request.form.get("reading_score")
        writing_score = request.form.get("writing_score")

        # Basic presence check
        if not all([gender, ethnicity, parental_education, lunch, test_course,
                    reading_score, writing_score]):
            raise ValueError("All form fields are required")

        # Convert scores to float
        reading_score = float(reading_score)
        writing_score = float(writing_score)

        # Create custom data object
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        logger.info(f"Prediction request: {pred_df.to_dict(orient='records')}")

        # Predict
        results = predict_pipeline.predict(pred_df)
        predicted_score = round(results[0], 2)

        logger.info(f"Prediction result: {predicted_score}")

        return render_template(
            "home.html",
            results=f"Predicted Math Score: {predicted_score}"
        )

    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return render_template("home.html", results=f"Error: {str(ve)}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return render_template(
            "home.html",
            results="An error occurred while processing your request."
        )


# Optional: API endpoint for JSON requests
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for predictions."""
    if predict_pipeline is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Map API fields to expected column names
        required_fields = [
            "gender", "race_ethnicity", "parental_level_of_education",
            "lunch", "test_preparation_course", "reading_score", "writing_score"
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        custom_data = CustomData(
            gender=data["gender"],
            race_ethnicity=data["race_ethnicity"],
            parental_level_of_education=data["parental_level_of_education"],
            lunch=data["lunch"],
            test_preparation_course=data["test_preparation_course"],
            reading_score=float(data["reading_score"]),
            writing_score=float(data["writing_score"]),
        )
        pred_df = custom_data.get_data_as_data_frame()
        result = predict_pipeline.predict(pred_df)
        return jsonify({"predicted_math_score": round(float(result[0]), 2)})

    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
