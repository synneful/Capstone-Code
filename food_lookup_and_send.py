"""
===================================================================
Food Storage Parameter Lookup + Blynk Bridge
===================================================================
Purpose:
    Step 3 & 4 in the workflow:
      - "System retrieves optimal storage parameters from food database"
      - "Temperature and humidity targets are sent to ESP32"

    Runs TFLite inference on a captured food image, looks up the
    optimal temp/humidity for that food, and pushes those targets
    to the ESP32 via the Blynk Cloud REST API (virtual pin write).

    This script can run on a phone (via a Python backend / Kivy /
    Chaquopy bridge), on a small companion server, or be ported to
    the mobile app's native HTTP client calling the same Blynk
    REST endpoints shown below.

Usage:
    python food_lookup_and_send.py --image captured_food.jpg
===================================================================
"""

import argparse
import json
import numpy as np
import requests
from PIL import Image

# tflite_runtime has no Windows wheels on PyPI, so we fall back to
# TensorFlow's built-in interpreter (same API, works everywhere TF does).
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ---------------------- Configuration ----------------------
TFLITE_MODEL_PATH = "./food_classifier.tflite"      # from model_output_finetuned/
CLASS_INDICES_PATH = "./class_indices.json"          # from model_output_finetuned/
FOOD_DB_PATH = "./food_storage_db.json"

BLYNK_AUTH_TOKEN = "Your Token Here"   # Same token as in the ESP32 sketch
BLYNK_SERVER = "https://blynk.cloud/external/api/update"

IMG_SIZE = (224, 224)


def load_class_map(path):
    with open(path, "r") as f:
        class_indices = json.load(f)
    # class_indices.json is already {"0": "apple_pie", "1": ...} (index -> name).
    # JSON keys are always strings, so just convert the key back to int.
    return {int(k): v for k, v in class_indices.items()}


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def run_inference(image_path):
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    predicted_index = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))

    class_map = load_class_map(CLASS_INDICES_PATH)
    predicted_label = class_map.get(predicted_index, "unknown")

    return predicted_label, confidence


def lookup_storage_params(food_label):
    with open(FOOD_DB_PATH, "r") as f:
        db = json.load(f)

    if food_label in db:
        return db[food_label]
    else:
        print(f"Warning: '{food_label}' not found in database. Using default safe values.")
        return db.get("default", {"target_temp": 4.0, "target_humidity": 85.0})


def send_to_blynk(target_temp, target_humidity, food_label):
    """
    Pushes target temp/humidity and food label to ESP32 via Blynk virtual pins.
    V2 = target temp, V3 = target humidity, V4 = food label (matches ESP32 sketch)
    """
    params_to_send = {
        "V2": target_temp,
        "V3": target_humidity,
        "V4": food_label
    }

    for pin, value in params_to_send.items():
        resp = requests.get(BLYNK_SERVER, params={
            "token": BLYNK_AUTH_TOKEN,
            "pin": pin,
            "value": value
        })
        if resp.status_code != 200:
            print(f"Failed to update {pin}: {resp.text}")
        else:
            print(f"Sent {pin} = {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to captured food image")
    args = parser.parse_args()

    food_label, confidence = run_inference(args.image)
    print(f"Identified food: {food_label} (confidence: {confidence:.2%})")

    params = lookup_storage_params(food_label)
    print(f"Optimal storage: {params['target_temp']}C / {params['target_humidity']}% RH")

    send_to_blynk(params["target_temp"], params["target_humidity"], food_label)


if __name__ == "__main__":
    main()
