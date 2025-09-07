from flask import Flask, request, jsonify, g
from flask_cors import CORS
import os, traceback
from google.cloud import firestore
from openai import OpenAI
from datetime import datetime, timedelta, timezone
import numpy as np
import bcrypt
import jwt
import yaml
import json
from dotenv import load_dotenv
from functools import wraps
from collections import defaultdict

load_dotenv()

SECRET_KEY = "super-secret-key"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./firebase-key.json"

app = Flask(__name__)
CORS(app)

db = firestore.Client()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------
# Auth decorator
# ----------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == "OPTIONS":
            return '', 204

        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            return jsonify({"error": "Token missing"}), 401
        try:
            token = token.split()[1]
            g.user = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception as e:
            return jsonify({"error": "Invalid token", "details": str(e)}), 401
        return f(*args, **kwargs)
    return decorated


# ----------------------
# User endpoints
# ----------------------
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")
        name = data.get("name", "")
        if not email or not password:
            return jsonify({"error": "Missing email or password"}), 400

        users_ref = db.collection("users")
        existing = users_ref.where("email", "==", email).get()
        if existing:
            return jsonify({"error": "User exists"}), 400

        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        user_doc = users_ref.document()
        user_doc.set({"email": email, "password": hashed_pw.decode(), "name": name})
        return jsonify({"success": True, "user_id": user_doc.id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")
        if not email or not password:
            return jsonify({"error": "Missing email/password"}), 400

        users_ref = db.collection("users")
        docs = users_ref.where("email", "==", email).get()
        if not docs:
            return jsonify({"error": "User not found"}), 404

        user = docs[0].to_dict()
        if not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return jsonify({"error": "Invalid password"}), 401

        token = jwt.encode({
            "user_id": docs[0].id,
            "email": email,
            "exp": datetime.now(timezone.utc) + timedelta(days=1)
        }, SECRET_KEY, algorithm="HS256")

        return jsonify({"token": token, "user": {"id": docs[0].id, "email": email, "name": user.get("name")}})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------
# Patient endpoints
# ----------------------
@app.route("/add-patient", methods=["POST", "OPTIONS"])
@token_required
def add_patient():
    if request.method == "OPTIONS":
        return '', 204
    try:
        data = request.get_json()
        name, age, condition = data.get("name"), data.get("age"), data.get("condition")
        if not name or not age or not condition:
            return jsonify({"success": False, "error": "Missing fields"}), 400

        patients_ref = db.collection("users").document(g.user["user_id"]).collection("patients")
        patients_ref.add({"name": name, "age": age, "condition": condition})
        return jsonify({"success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/patients", methods=["GET"])
@token_required
def get_patients():
    try:
        patients_ref = db.collection("users").document(g.user["user_id"]).collection("patients")
        patients = [{"id": doc.id, **doc.to_dict()} for doc in patients_ref.stream()]
        return jsonify(patients)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ----------------------
# Chat endpoints
# ----------------------
@app.route("/chat/<patient_id>", methods=["POST", "OPTIONS"])
@token_required
def chat(patient_id):
    if request.method == "OPTIONS":
        return '', 204

    data = request.json
    user_message = data.get("message", "").strip()
    file_url = data.get("file_url")
    file_name = data.get("file_name", "uploaded_file")

    if not user_message and not file_url:
        return jsonify({"error": "Empty message and no file"}), 400

    try:
        # ✅ Keep everything scoped under the authenticated user
        patient_ref = (
            db.collection("users")
            .document(g.user["user_id"])
            .collection("patients")
            .document(patient_id)
        )
        if not patient_ref.get().exists:
            return jsonify({"error": "Unauthorized"}), 401

        chat_ref = patient_ref.collection("chat_history")

        # Save user message
        msg_doc = {"role": "user", "content": user_message, "timestamp": datetime.utcnow()}
        if file_url:
            msg_doc["file"] = {"url": file_url, "name": file_name}
        chat_ref.add(msg_doc)

        # Optional EEG file processing
        eeg_summary = None
        if file_url:
            from script import run_eeg_inference
            bucket, blob = file_url.replace("gs://", "").split("/", 1)
            eeg_output = run_eeg_inference(bucket, blob)
            eeg_summary = f"EEG model output: {np.array(eeg_output).tolist()}"
            chat_ref.add({
                "role": "system",
                "content": f"Attached EEG file analyzed from {file_url}",
                "timestamp": datetime.utcnow()
            })

        # Prepare GPT messages
        with open("prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "developer", "content": prompts["developer"]}
        ]
        if eeg_summary:
            messages.append({
                "role": "system", 
                "name": "eeg_data",
                "content": json.dumps(eeg_summary, indent=2)
            })
        if user_message:
            messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content

        chat_ref.add({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.utcnow()
        })

        return jsonify({"response": reply, "eeg_summary": eeg_summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route("/chat-history/<patient_id>", methods=["GET"])
@token_required
def get_chat_history(patient_id):
    try:
        patient_ref = db.collection("users").document(g.user["user_id"]).collection("patients").document(patient_id)
        if not patient_ref.get().exists:
            return jsonify({"error": "Unauthorized"}), 401

        chat_ref = patient_ref.collection("chat_history").order_by("timestamp")
        messages = [doc.to_dict() for doc in chat_ref.stream()]
        for m in messages:
            if "timestamp" in m:
                m["timestamp"] = m["timestamp"].isoformat()
        return jsonify(messages)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/search", methods=["GET", "OPTIONS"])
@token_required
def search_messages():
    if request.method == "OPTIONS":
        return '', 204

    keyword = request.args.get("q", "").lower().strip()
    if not keyword:
        return jsonify([])  # Return empty array if no keyword

    try:
        # Search all chat_history subcollections across all patients
        chats_ref = db.collection_group("chat_history")
        docs = chats_ref.stream()

        results = []

        for doc in docs:
            data = doc.to_dict()
            content = data.get("content", "").lower()
            if keyword in content:
                # Find patient_id from document reference
                patient_ref = doc.reference.parent.parent
                if not patient_ref:
                    continue
                patient_id = patient_ref.id

                results.append({
                    "patient_id": patient_id,
                    "message_id": doc.id,
                    "role": data.get("role"),
                    "content": data.get("content"),
                    "timestamp": data.get("timestamp").isoformat() if data.get("timestamp") else None
                })

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    # In-memory storage per session (keyed by a temporary ID, e.g., generated on frontend)
ephemeral_chats = defaultdict(list)

@app.route("/anon-chat", methods=["POST", "OPTIONS"])
def anon_chat():
    if request.method == "OPTIONS":
        return '', 204

    data = request.json
    session_id = data.get("session_id")  # frontend generates a random UUID per page load
    user_message = data.get("message", "").strip()

    if not session_id or not user_message:
        return jsonify({"error": "Missing session_id or message"}), 400

    try:
        # Save user message
        ephemeral_chats[session_id].append({"role": "user", "content": user_message})

        # GPT call
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + ephemeral_chats[session_id]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content

        # Save assistant message
        ephemeral_chats[session_id].append({"role": "assistant", "content": reply})

        return jsonify({"response": reply, "history": ephemeral_chats[session_id]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ----------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)