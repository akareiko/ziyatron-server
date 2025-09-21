from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import traceback
from google.cloud import firestore, storage
from openai import OpenAI
from datetime import datetime, timedelta, timezone
import numpy as np
import bcrypt
import jwt
from collections import defaultdict
import json
from dotenv import load_dotenv
import yaml
from script import run_eeg_inference
import uuid
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
import re
import asyncio

load_dotenv()

SECRET_KEY = "super-secret-key"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./firebase-key.json"
FRONTEND_URL = os.getenv("FRONTEND_URL")

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db = firestore.Client()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
storage_client = storage.Client()
security = HTTPBearer()

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()

# Pydantic models
class UserRegister(BaseModel):
    email: str
    password: str
    name: Optional[str] = ""

class UserLogin(BaseModel):
    email: str
    password: str

class PatientCreate(BaseModel):
    name: str
    age: int
    condition: str

class ChatMessage(BaseModel):
    message: Optional[str] = ""
    file_url: Optional[str] = None
    file_name: Optional[str] = "uploaded_file"

class AnonChatMessage(BaseModel):
    session_id: str
    message: str

class WebSocketData(BaseModel):
    token: str
    patient_id: str
    session_id: str
    message: str
    eeg_summary: Optional[str] = None

# Utility functions
def generate_signed_url(bucket_name: str, blob_name: str, expiration_minutes: int = 15) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(expiration=timedelta(minutes=expiration_minutes))
    return url

# Auth dependency
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        token = credentials.credentials
        user = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# User endpoints
@app.post("/register")
async def register(user_data: UserRegister):
    try:
        users_ref = db.collection("users")
        existing = users_ref.where("email", "==", user_data.email).get()
        if existing:
            raise HTTPException(status_code=400, detail="User exists")

        hashed_pw = bcrypt.hashpw(user_data.password.encode("utf-8"), bcrypt.gensalt())
        user_doc = users_ref.document()
        user_doc.set({
            "email": user_data.email, 
            "password": hashed_pw.decode(), 
            "name": user_data.name
        })
        return {"success": True, "user_id": user_doc.id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
async def login(user_data: UserLogin):
    try:
        users_ref = db.collection("users")
        docs = users_ref.where("email", "==", user_data.email).get()
        if not docs:
            raise HTTPException(status_code=404, detail="User not found")

        user = docs[0].to_dict()
        if not bcrypt.checkpw(user_data.password.encode(), user["password"].encode()):
            raise HTTPException(status_code=401, detail="Invalid password")

        token = jwt.encode({
            "user_id": docs[0].id,
            "email": user_data.email,
            "exp": datetime.now(timezone.utc) + timedelta(days=1)
        }, SECRET_KEY, algorithm="HS256")

        return {
            "token": token, 
            "user": {
                "id": docs[0].id, 
                "email": user_data.email, 
                "name": user.get("name")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Patient endpoints
@app.post("/add-patient")
async def add_patient(patient_data: PatientCreate, current_user: dict = Depends(get_current_user)):
    try:
        patients_ref = db.collection("users").document(current_user["user_id"]).collection("patients")
        doc_ref = patients_ref.add({
            "name": patient_data.name,
            "age": patient_data.age,
            "condition": patient_data.condition,
            "createdAt": SERVER_TIMESTAMP
        })
        
        new_doc_id = doc_ref[1].id if isinstance(doc_ref, tuple) else doc_ref.id

        return {
            "success": True,
            "patient": {
                "id": new_doc_id,
                "name": patient_data.name,
                "age": patient_data.age,
                "condition": patient_data.condition,
                "createdAt": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients")
async def get_patients(current_user: dict = Depends(get_current_user)):
    try:
        patients_ref = db.collection("users").document(current_user["user_id"]).collection("patients").order_by("createdAt", direction=firestore.Query.DESCENDING)
        patients = [{"id": doc.id, **doc.to_dict()} for doc in patients_ref.stream()]
        return patients
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@app.post("/chat/{patient_id}")
async def post_chat(patient_id: str, chat_data: ChatMessage, current_user: dict = Depends(get_current_user)):
    user_message = chat_data.message.strip() if chat_data.message else ""
    file_url = chat_data.file_url
    file_name = chat_data.file_name

    if not user_message and not file_url:
        raise HTTPException(status_code=400, detail="Empty message and no file")

    try:
        # Firestore refs
        patient_ref = db.collection("users").document(current_user["user_id"]).collection("patients").document(patient_id)
        if not patient_ref.get().exists:
            raise HTTPException(status_code=401, detail="Unauthorized")
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

        return {"session_id": session_id, "eeg_summary": eeg_summary}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{patient_id}")
async def get_chat_history(patient_id: str, current_user: dict = Depends(get_current_user)):
    try:
        patient_ref = db.collection("users").document(current_user["user_id"]).collection("patients").document(patient_id)
        if not patient_ref.get().exists:
            raise HTTPException(status_code=401, detail="Unauthorized")

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

        return messages
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_messages(q: str = "", current_user: dict = Depends(get_current_user)):
    keyword = q.lower().strip()
    if not keyword:
        return []

    try:
        results = []

        # 1. Get the current user's patients
        patients_ref = db.collection("users").document(current_user["user_id"]).collection("patients")
        for patient_doc in patients_ref.stream():
            patient_id = patient_doc.id

            # 2. Get chat_history for this patient
            chat_ref = patients_ref.document(patient_id).collection("chat_history")
            for chat_doc in chat_ref.stream():
                data = chat_doc.to_dict()
                content_value = data.get("content", "")

                # Ensure content is a string
                if isinstance(content_value, dict):
                    content_str = " ".join(str(v) for v in content_value.values()).lower()
                else:
                    content_str = str(content_value).lower()

                if keyword in content_str:
                    results.append({
                        "patient_id": patient_id,
                        "message_id": chat_doc.id,
                        "role": data.get("role"),
                        "content": data.get("content"),
                        "timestamp": data.get("timestamp").isoformat() if data.get("timestamp") else None
                    })

        return results

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Anonymous chat
ephemeral_chats = defaultdict(list)

@app.post("/anon-chat")
async def anon_chat(chat_data: AnonChatMessage):
    session_id = chat_data.session_id
    user_message = chat_data.message.strip()

    if not session_id or not user_message:
        raise HTTPException(status_code=400, detail="Missing session_id or message")

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

        return {"response": reply, "history": ephemeral_chats[session_id]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Wait for incoming message
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "join":
                await manager.send_message({"status": f"Joined session {session_id}"}, session_id)
            
            elif data.get("type") == "start_assistant":
                await handle_assistant_stream(data, session_id)
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)

async def handle_assistant_stream(data: dict, session_id: str):
    token = data.get("token", "")
    patient_id = data.get("patient_id")
    user_message = data.get("message")
    eeg_summary = data.get("eeg_summary")

    # Remove Bearer prefix if present
    if token.startswith("Bearer "):
        token = token.split()[1]

    try:
        user = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        await manager.send_message({"assistant_update": {"text_delta": "Auth error: Token expired"}}, session_id)
        return
    except Exception as e:
        await manager.send_message({"assistant_update": {"text_delta": f"Auth error: {e}"}}, session_id)
        return

    try:
        # Firestore refs
        patient_ref = db.collection("users").document(user["user_id"]).collection("patients").document(patient_id)
        chat_ref = patient_ref.collection("chat_history")

        # Prepare GPT messages
        with open("prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)

        messages = []
        if eeg_summary:
            messages.append({"role": "system", "content": prompts.get("system_eeg_summary", "")})
            messages.append({"role": "user", "content": eeg_summary})
        else:
            messages.append({"role": "system", "content": prompts.get("system_general", "")})

        messages.append({"role": "user", "content": user_message})
        partial_text = ""
        
        with client.responses.stream(model="gpt-4o-mini", input=messages) as stream:
            for event in stream:
                # Only process text deltas
                if event.type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta and isinstance(delta, str):
                        clean_delta = delta.replace("\r", "")

                        # remove zero-width / BOM / NBSP that break "start of line"
                        clean_delta = clean_delta.replace("\u200b", "")   # ZERO WIDTH SPACE
                        clean_delta = clean_delta.replace("\ufeff", "")   # BOM
                        clean_delta = clean_delta.replace("\u00a0", " ")  # NO-BREAK SPACE -> normal

                        # If the incoming chunk *starts* with a heading/list token, ensure it's on its own line.
                        # (if partial_text doesn't already end with a newline, prefix one)
                        if re.match(r'^\s*(#{1,6}\s+|- |\* |\d+\.\s+|> )', clean_delta) and not partial_text.endswith('\n'):
                            clean_delta = '\n' + clean_delta.lstrip()

                        # Also normalize internal cases where a space + '#' appears mid-chunk -> replace with newline + '#'
                        clean_delta = re.sub(r'(?<!\n)\s+(?=(#{1,6}\s+|- |\* |\d+\.\s+|> ))', '\n', clean_delta)

                        partial_text += clean_delta
                        await manager.send_message({"assistant_update": {"text_delta": partial_text}}, session_id)

        # Save final message to Firestore
        chat_ref.add({
            "role": "assistant",
            "content": {
                "text": partial_text
            },
            "timestamp": datetime.utcnow()
        })

    except Exception as e:
        await manager.send_message({"assistant_update": {"text_delta": f"Error: {e}"}}, session_id)
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=True)