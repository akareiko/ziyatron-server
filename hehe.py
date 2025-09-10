from flask import Flask, request, jsonify, g, Response, stream_with_context
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import os, traceback
from google.cloud import firestore, storage
from openai import OpenAI
from datetime import datetime, timedelta, timezone
import numpy as np
import bcrypt
import jwt
from functools import wraps
from collections import defaultdict
import json
from dotenv import load_dotenv
import yaml
from script import run_eeg_inference
import uuid


load_dotenv()

SECRET_KEY = "super-secret-key"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./firebase-key.json"

app = Flask(__name__)
CORS(app, origins="http://localhost:3000", supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000", async_mode="eventlet")

db = firestore.Client()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

storage_client = storage.Client()

def generate_signed_url(bucket_name, blob_name, expiration_minutes=15):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(expiration=timedelta(minutes=expiration_minutes))
    return url


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
def post_chat(patient_id):
    if request.method == "OPTIONS":
        return '', 204

    data = request.json
    user_message = data.get("message", "").strip()
    file_url = data.get("file_url")
    file_name = data.get("file_name", "uploaded_file")

    if not user_message and not file_url:
        return jsonify({"error": "Empty message and no file"}), 400

    try:
        # Firestore refs
        patient_ref = db.collection("users").document(g.user["user_id"]).collection("patients").document(patient_id)
        if not patient_ref.get().exists:
            return jsonify({"error": "Unauthorized"}), 401
        chat_ref = patient_ref.collection("chat_history")

        # Save user message
        msg_doc = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow()
        }

        file_metadata = None
        bucket, blob = None, None
        if file_url:
            bucket, blob = file_url.replace("gs://", "").split("/", 1)
            signed_url = generate_signed_url(bucket, blob)
            file_metadata = {"url": signed_url, "name": file_name}
            msg_doc["file"] = file_metadata

        chat_ref.add(msg_doc)

        # EEG processing
        eeg_summary = None
        if file_url:
            try:
                output = run_eeg_inference(bucket, blob)
                eeg_summary = f"EEG model output: {np.array(output).tolist()}"
                chat_ref.add({
                    "role": "system",
                    "content": f"Attached EEG file analyzed from {file_name}",
                    "timestamp": datetime.utcnow()
                })
            except Exception as e:
                print(f"[ERROR] EEG inference failed: {e}")
                eeg_summary = "EEG analysis failed. Please try again later."

        # Create GPT session ID
        session_id = str(uuid.uuid4())

        # Store session metadata in memory or DB (optional)
        # e.g., pending_messages[session_id] = {...}

        return jsonify({"session_id": session_id, "eeg_summary": eeg_summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------
# WebSocket: stream assistant responses
# ----------------------
@socketio.on("join")
def on_join(data):
    session_id = data.get("session_id")
    join_room(session_id)
    emit("status", {"msg": f"Joined session {session_id}"}, room=session_id)


@socketio.on("start_assistant")
def start_assistant(data):
    token = data.get("token")
    patient_id = data.get("patient_id")
    session_id = data.get("session_id")
    user_message = data.get("message")
    eeg_summary = data.get("eeg_summary")  # optional

    # manually decode JWT (same as token_required)
    if token.startswith("Bearer "):
        token = token.split()[1]

    try:
        user = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        emit("assistant_update", {"text_delta": "Auth error: Token expired"}, room=session_id)
        return
    except Exception as e:
        emit("assistant_update", {"text_delta": f"Auth error: {e}"}, room=session_id)
        return

    try:
        # Firestore refs
        patient_ref = db.collection("users").document(user["user_id"]).collection("patients").document(patient_id)
        chat_ref = patient_ref.collection("chat_history")

        # Prepare GPT messages
        with open("prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)

        messages = [
            {"role": "system", "content": prompts.get("system", "")},
            # {"role": "user", "content": prompts.get("developer", "")}
        ]

        if eeg_summary:
            messages.append({"role": "system", "content": eeg_summary})

        partial_text = ""

        with client.responses.stream(model="gpt-4o-mini", input=messages) as stream:
            for event in stream:
                # Only process text deltas
                if event.type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta and isinstance(delta, str):
                        # normalize whitespace/newlines
                        clean_delta = delta.replace("\r", "")
                        partial_text += clean_delta

                        # emit to frontend
                        emit("assistant_update", {"text_delta": clean_delta}, room=session_id)

        # Save final message to Firestore
        chat_ref.add({
            "role": "assistant",
            "content": {
                "text": partial_text
            },
            "timestamp": datetime.utcnow()
        })

    except Exception as e:
        emit("assistant_update", {"text_delta": f"Error: {e}"}, room=session_id)
        traceback.print_exc()



@app.route("/chat-history/<patient_id>", methods=["GET"])
@token_required
def get_chat_history(patient_id):
    try:
        patient_ref = db.collection("users").document(g.user["user_id"]).collection("patients").document(patient_id)
        if not patient_ref.get().exists:
            return jsonify({"error": "Unauthorized"}), 401

        chat_ref = patient_ref.collection("chat_history").order_by("timestamp")
        messages = []
        for doc in chat_ref.stream():
            data = doc.to_dict()

            # Convert Firestore timestamp to iso
            if "timestamp" in data:
                data["timestamp"] = data["timestamp"].isoformat()

            # If file exists, replace gs:// with signed URL
            if "file" in data and "url" in data["file"]:
                gs_path = data["file"]["url"]
                if gs_path.startswith("gs://"):
                    bucket_name, blob_name = gs_path.replace("gs://", "").split("/", 1)
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    signed_url = blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(hours=1),
                        method="GET"
                    )
                    data["file"]["url"] = signed_url

            messages.append(data)

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
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)