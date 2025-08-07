from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

DATA_FILE = "patients.txt"

@app.route("/add-patient", methods=["POST", "OPTIONS"])
def add_patient():
    if request.method == "OPTIONS":
        # Handle preflight request
        return '', 204

    data = request.get_json()
    name = data.get("name")
    age = data.get("age")
    condition = data.get("condition")

    if not name or not age or not condition:
        return jsonify({"success": False, "error": "Missing fields"}), 400

    with open("patients.txt", "a", encoding="utf-8") as f:
        f.write(f"Name: {name}, Age: {age}, Condition: {condition}\n")

    return jsonify({"success": True})

@app.route("/patients", methods=["GET"])
def get_patients():
    if not os.path.exists("patients.txt"):
        return jsonify([])

    patients = []
    with open("patients.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(", ")
                patient = {}
                for part in parts:
                    key, value = part.split(": ", 1)
                    patient[key.strip()] = value.strip()
                patients.append(patient)

    return jsonify(patients)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    app.run(debug=True, port=5000)