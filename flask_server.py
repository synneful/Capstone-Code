"""
===================================================================
Food Preservation System - Flask Image Upload Server
===================================================================
Purpose:
    Bridges the phone camera to the AI inference + Blynk pipeline.

    Workflow:
      1. User opens a simple webpage on their phone (served by this
         script on your laptop's local IP, e.g. http://192.168.1.x:5000)
      2. User taps "Take Photo" or "Choose Photo" and submits
      3. Server receives the image, runs TFLite inference,
         looks up storage parameters, pushes to Blynk
      4. Page shows result: food label, confidence, temp/humidity targets

    Both devices (phone + laptop) must be on the same WiFi network.

Requirements:
    pip install flask

Files needed in same folder as this script:
    - food_classifier.tflite
    - class_indices.json
    - food_storage_db.json

Usage:
    python flask_server.py

Then on your phone, open a browser and go to:
    http://<your-laptop-ip>:5000

To find your laptop's local IP:
    Windows: run `ipconfig` in terminal, look for IPv4 address
             (e.g. 192.168.1.105)
===================================================================
"""

import json
import os
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ---------------------- Configuration ----------------------
TFLITE_MODEL_PATH = "./food_classifier.tflite"
CLASS_INDICES_PATH = "./class_indices.json"
FOOD_DB_PATH = "./food_storage_db.json"

BLYNK_AUTH_TOKEN = "Your Token Here"   # Replace with your real token
BLYNK_SERVER = "https://blynk.cloud/external/api/update"

IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# ---------------------- Load model once at startup ----------------------
print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded.")

with open(CLASS_INDICES_PATH, "r") as f:
    class_map = {int(k): v for k, v in json.load(f).items()}

with open(FOOD_DB_PATH, "r") as f:
    food_db = json.load(f)


# ---------------------- Inference helpers ----------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_inference(image_path):
    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    predicted_index = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))
    return class_map.get(predicted_index, "unknown"), confidence


def lookup_storage_params(food_label):
    return food_db.get(food_label, food_db.get(
        "default", {"target_temp": 4.0, "target_humidity": 85.0}
    ))


def send_to_blynk(target_temp, target_humidity, food_label):
    results = {}
    for pin, value in {"V2": target_temp, "V3": target_humidity, "V4": food_label}.items():
        resp = requests.get(BLYNK_SERVER, params={
            "token": BLYNK_AUTH_TOKEN,
            "pin": pin,
            "value": value
        })
        results[pin] = "OK" if resp.status_code == 200 else f"Error: {resp.text}"
    return results


# ---------------------- Web UI ----------------------
# Single-file HTML page served directly from Python — no separate template files needed.
PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Smart Food Preservation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 480px;
            margin: 0 auto;
            padding: 20px;
            background: #f0f4f8;
        }
        h2 { color: #2d3748; text-align: center; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        input[type=file] {
            width: 100%;
            padding: 10px 0;
            font-size: 16px;
        }
        button {
            width: 100%;
            padding: 14px;
            background: #3182ce;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            margin-top: 12px;
        }
        button:active { background: #2c5282; }
        .result-label { font-size: 22px; font-weight: bold; color: #2d3748; }
        .result-conf { color: #718096; font-size: 14px; margin-bottom: 12px; }
        .result-storage { font-size: 18px; color: #276749; }
        .blynk-status { font-size: 13px; color: #718096; margin-top: 10px; }
        #preview { width: 100%; border-radius: 8px; margin-top: 10px; display: none; }
        .loading { text-align: center; color: #718096; display: none; }
    </style>
</head>
<body>
    <h2>🥗 Food Preservation System</h2>

    <div class="card">
        <b>Step 1:</b> Take or choose a food photo<br><br>
        <input type="file" id="photoInput" accept="image/*" capture="environment">
        <img id="preview" alt="preview">
    </div>

    <div class="card">
        <b>Step 2:</b> Identify food and set storage targets
        <button onclick="uploadPhoto()">📷 Identify Food</button>
    </div>

    <div class="loading" id="loading">⏳ Analysing image...</div>

    <div class="card" id="resultCard" style="display:none">
        <div class="result-label" id="foodLabel">—</div>
        <div class="result-conf" id="confidence">—</div>
        <div class="result-storage" id="storageParams">—</div>
        <div class="blynk-status" id="blynkStatus">—</div>
    </div>

    <script>
        document.getElementById('photoInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        async function uploadPhoto() {
            const fileInput = document.getElementById('photoInput');
            if (!fileInput.files[0]) {
                alert('Please select or take a photo first.');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultCard').style.display = 'none';

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            try {
                const response = await fetch('/identify', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';
                document.getElementById('resultCard').style.display = 'block';

                if (data.error) {
                    document.getElementById('foodLabel').textContent = 'Error';
                    document.getElementById('confidence').textContent = data.error;
                } else {
                    document.getElementById('foodLabel').textContent =
                        data.food_label.replace(/_/g, ' ').toUpperCase();
                    document.getElementById('confidence').textContent =
                        `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                    document.getElementById('storageParams').textContent =
                        `🌡 Target: ${data.target_temp}°C  💧 Humidity: ${data.target_humidity}%`;
                    document.getElementById('blynkStatus').textContent =
                        `Blynk: V2=${data.blynk_status.V2}  V3=${data.blynk_status.V3}  V4=${data.blynk_status.V4}`;
                }
            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                alert('Server error: ' + err.message);
            }
        }
    </script>
</body>
</html>
"""


# ---------------------- Routes ----------------------
@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image temporarily
    save_path = os.path.join(UPLOAD_FOLDER, "latest_upload.jpg")
    file.save(save_path)

    try:
        food_label, confidence = run_inference(save_path)
        params = lookup_storage_params(food_label)
        blynk_status = send_to_blynk(
            params["target_temp"], params["target_humidity"], food_label
        )

        print(f"Identified: {food_label} ({confidence:.2%}) -> "
              f"{params['target_temp']}C / {params['target_humidity']}% RH")

        return jsonify({
            "food_label": food_label,
            "confidence": confidence,
            "target_temp": params["target_temp"],
            "target_humidity": params["target_humidity"],
            "blynk_status": blynk_status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------- Run ----------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Smart Food Preservation Server running.")
    print("Find your laptop IP with: ipconfig (Windows)")
    print("Then open on your phone: http://<your-ip>:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
