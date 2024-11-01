from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("yolov8x.pt")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/classify', methods=['POST'])
def classify():
    if 'photo' not in request.files:
        return jsonify({"error": "No file part"}), 400

    photo = request.files['photo']

    if photo.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if photo and allowed_file(photo.filename):
        filename = secure_filename(photo.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(filepath)

        try:
            results = model.predict(filepath)
            result = results[0]

            if len(result.boxes) > 0:
                box = result.boxes[0]
                classification = result.names[int(box.cls[0].item())]
            else:
                classification = "No object detected"

            os.remove(filepath)

            return jsonify({"classification": classification})

        except Exception as e:
            # Clean up the file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Invalid file type"}), 400


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=5500, debug=True)