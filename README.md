# Ziyatron Server

Backend for [Ziyatron](https://github.com/akareiko/ziyatron): a REST + WebSocket API that serves patient data, runs EEG recordings through a seizure-detection model, and lets clinicians chat with an LLM about the results.

## What it does

- **Auth & patients** — register/login (JWT), add and search patients.
- **EEG inference** — `eeg_inference/` preprocesses raw EEG (filtering, z-normalization, channel selection via `mne`) and runs it through an ONNX seizure-classification model (`onnx_infer.py`).
- **Chat** — per-patient chat backed by OpenAI, prompted with clinical-assistant instructions (`prompts.yaml`) and grounded in that patient's EEG results; responses are structured Markdown reports (seizure activity, possible types, clinical relevance, next steps).
- **Realtime** — a WebSocket channel streams inference/chat progress.
- Firebase (Firestore + Cloud Storage) for data and file storage.

## Two server implementations

- **`server_fastapi.py`** — FastAPI + async, includes the WebSocket endpoint. This is the current one.
- **`server_flask.py`** — an earlier Flask + Flask-SocketIO version with the same REST routes, kept for reference.

(Renamed from `haha.py` / `hehe.py`.)

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# needed, none of these are committed:
#  - firebase-key.json (Firebase service account)
#  - models/kaz_data_v2.onnx (ONNX model)
#  - .env with OPENAI_API_KEY, FRONTEND_URL, etc.

uvicorn server_fastapi:app --reload
```

## Status

Active development — inference pipeline and chat are functional against the team's own model/Firebase project; not packaged for a fresh deploy yet.

> **Note:** JWT signing currently uses a hardcoded secret in source. Swap this for an environment variable before any real deployment.
