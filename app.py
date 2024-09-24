from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
import logging
from glob import glob


# Import your model and any other necessary modules
from src.models.predict_model import (
    preprocess_data,
    process_clean_data,
    adding_features,
    test_model,
)

# Initialize the Flask application
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/csv_files"
app.config["ALLOWED_EXTENSIONS"] = {"csv"}

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    """Render the index page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and prediction."""

    if "files[]" not in request.files:
        return redirect(request.url)

    files = request.files.getlist("files[]")
    if files.count == 0:
        return redirect(request.url)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    # Create a list of all files in the data path.
    files = glob(os.path.join(app.config["UPLOAD_FOLDER"], "*.csv"))
    # Remove the path from the filename
    # files = [os.path.basename(file) for file in files]

    # Preprocess the data
    preprocess_data(files)

    # Clean the data
    process_clean_data()

    # Add features
    adding_features()

    # Make predictions
    predictions = test_model()

    # afficher les predictions
    result = pd.DataFrame(predictions)

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
