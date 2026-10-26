import json
import os
import uuid

import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "food_freshness_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the existing model and its class labels once when the application starts.
model = load_model(MODEL_PATH, compile=False)
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as class_file:
    class_names = json.load(class_file)

if not isinstance(class_names, list) or not all(
    isinstance(name, str) for name in class_names
):
    raise ValueError("class_names.json must contain a list of class names.")

CLASS_DETAILS = {
    "freshapples": ("Apple", "Fresh"),
    "freshbanana": ("Banana", "Fresh"),
    "freshoranges": ("Orange", "Fresh"),
    "rottenapples": ("Apple", "Rotten"),
    "rottenbanana": ("Banana", "Rotten"),
    "rottenoranges": ("Orange", "Rotten"),
}


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if image_file is None or not image_file.filename:
            error = "Please choose an image to upload."
        else:
            safe_original_name = secure_filename(image_file.filename)
            extension = os.path.splitext(safe_original_name)[1].lower()

            if not safe_original_name or extension not in ALLOWED_EXTENSIONS:
                error = "Please upload a JPG, JPEG, or PNG image."
            else:
                saved_filename = f"{uuid.uuid4().hex}{extension}"
                saved_path = os.path.join(UPLOAD_FOLDER, saved_filename)

                try:
                    # The model includes MobileNetV2 preprocessing, so only resize
                    # and convert to RGB here.
                    with Image.open(image_file.stream) as uploaded_image:
                        image = uploaded_image.convert("RGB").resize((224, 224))
                    image_array = np.asarray(image, dtype=np.float32)
                    image_batch = np.expand_dims(image_array, axis=0)
                    image_file.stream.seek(0)
                    image_file.save(saved_path)
                except (UnidentifiedImageError, OSError, ValueError):
                    error = "The selected file is not a valid image. Please try another."

                if error is None:
                    try:
                        probabilities = np.asarray(
                            model.predict(image_batch, verbose=0)
                        )
                        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
                            raise ValueError("The model returned an unexpected output.")

                        class_index = int(np.argmax(probabilities[0]))
                        if class_index >= len(class_names):
                            raise ValueError("The model output does not match its labels.")

                        predicted_class = class_names[class_index]
                        if predicted_class not in CLASS_DETAILS:
                            raise ValueError("The predicted class is not recognized.")

                        food, freshness = CLASS_DETAILS[predicted_class]
                        result = {
                            "food": food,
                            "freshness": freshness,
                            "confidence": float(probabilities[0][class_index]) * 100,
                            "image_url": url_for(
                                "uploaded_file", filename=saved_filename
                            ),
                        }
                    except Exception:
                        app.logger.exception("Food freshness prediction failed.")
                        os.remove(saved_path)
                        error = (
                            "We couldn't analyze that image. "
                            "Please try again with a different image."
                        )

    return render_template("index.html", error=error, result=result)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
