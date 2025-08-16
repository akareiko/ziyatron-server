from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
from google.cloud import firestore 
from openai import OpenAI

app = Flask(__name__)
CORS(app)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./firebase-key.json"
db = firestore.Client()

client = OpenAI(api_key="apikey")

@app.route("/add-patient", methods=["POST", "OPTIONS"])
def add_patient():
    if request.method == "OPTIONS":
        return '', 204

    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    name = data.get("name")
    age = data.get("age")
    condition = data.get("condition")

    if not name or not age or not condition:
        return jsonify({"success": False, "error": "Missing fields"}), 400

    try:
        db.collection("patients").add({
            "name": name,
            "age": age,
            "condition": condition
        })
        return jsonify({"success": True})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/patients", methods=["GET"])
def get_patients():
    try:
        patients_ref = db.collection("patients")
        docs = patients_ref.stream()

        patients = []
        for doc in docs:
            patient = doc.to_dict()
            patient["id"] = doc.id
            patients.append(patient)

        return jsonify(patients)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"error": "Empty message"}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for patient-related chat."},
                {"role": "user", "content": user_message},
            ],
        )
        reply = response.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)