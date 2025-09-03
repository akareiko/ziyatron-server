from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
from google.cloud import firestore 
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Firestore credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./firebase-key.json"
db = firestore.Client()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------
# Add a new patient
# ----------------------
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

# ----------------------
# Get all patients
# ----------------------
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

# ----------------------
# Chat endpoint (per patient)
# ----------------------
@app.route("/chat/<patient_id>", methods=["POST"])
def chat(patient_id):
    data = request.json
    user_message = data.get("message", "")
    
    file_path = data.get("file_path", "")
    eeg_data = preprocess_eeg_data(file_path) # (N, 18, 2500)


    if not user_message.strip():
        return jsonify({"error": "Empty message"}), 400

    try:
        # Reference to patient's chat subcollection
        chat_ref = db.collection("patients").document(patient_id).collection("chat_history")

        # Save user message
        chat_ref.add({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow()
        })

        # Generate assistant reply
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are little cutie patootie."},
                {"role": "user", "content": user_message},
            ],
        )
        reply = response.choices[0].message.content

        # Save assistant message
        chat_ref.add({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.utcnow()
        })

        return jsonify({"response": reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------
# Fetch chat history (per patient)
# ----------------------
@app.route("/chat-history/<patient_id>", methods=["GET"])
def get_chat_history(patient_id):
    try:
        chat_ref = (
            db.collection("patients")
              .document(patient_id)
              .collection("chat_history")
              .order_by("timestamp")
        )
        docs = chat_ref.stream()

        messages = []
        for doc in docs:
            msg = doc.to_dict()
            messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })

        return jsonify(messages)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/search", methods=["GET"])
def search_messages():
    keyword = request.args.get("q", "").lower().strip()
    if not keyword:
        return jsonify({"error": "Missing search keyword"}), 400

    try:
        chats_ref = db.collection_group("chat_history")
        docs = chats_ref.stream()

        results = []
        for doc in docs:
            data = doc.to_dict()
            if keyword in data.get("content", "").lower():
                patient_ref = doc.reference.parent.parent
                patient_id = patient_ref.id

                results.append({
                    "patient_id": patient_id,
                    "message_id": doc.id,
                    "role": data.get("role"),
                    "content": data.get("content"),
                    "timestamp": data.get("timestamp").isoformat()
                })

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ----------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)