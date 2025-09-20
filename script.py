import os
import tempfile
from contextlib import contextmanager
from google.cloud import storage
from eeg_inference.onnx_infer import (
    create_session,
    run_inference_sync,
    run_inference_async,
    inspect_onnx_model,
)

# -------------------------------
# Config
# -------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./firebase-key.json"
ONNX_MODEL_PATH = "./models/kaz_data.onnx"
CONFIG_PATH = "./eeg_inference/kaz_configs.yaml"

# Load ONNX model session once (global)
try:
    SESSION = create_session(ONNX_MODEL_PATH)
    print("[INFO] ONNX model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load ONNX model: {e}")
    SESSION = None


# -------------------------------
# Helpers
# -------------------------------
@contextmanager
def download_from_firebase(bucket_name: str, blob_name: str):
    """
    Download a file from Firebase Storage (GCS bucket).
    Yields a local temporary file path (auto-cleaned).
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, os.path.basename(blob_name))
        blob.download_to_filename(local_path)
        print(f"[INFO] File downloaded to {local_path}")
        yield local_path
    # cleanup handled automatically


# -------------------------------
# Main functions
# -------------------------------
def run_eeg_inference(bucket_name: str, blob_name: str):
    """
    Full pipeline: download → ONNX inference.
    """
    if SESSION is None:
        raise RuntimeError("ONNX session not initialized. Check model path.")

    try:
        with download_from_firebase(bucket_name, blob_name) as file_path:
            output = run_inference_sync(SESSION, file_path, CONFIG_PATH)
            return output
    except Exception as e:
        print(f"[ERROR] run_eeg_inference failed: {e}")
        raise


async def run_eeg_inference_async(bucket_name: str, blob_name: str):
    """
    Async pipeline: download → ONNX inference.
    """
    if SESSION is None:
        raise RuntimeError("ONNX session not initialized. Check model path.")

    try:
        with download_from_firebase(bucket_name, blob_name) as file_path:
            output = await run_inference_async(SESSION, file_path, CONFIG_PATH)
            print(f"[INFO] Async inference output shape: {output.shape}")
            return output
    except Exception as e:
        print(f"[ERROR] run_eeg_inference_async failed: {e}")
        raise


# -------------------------------
# Script entry (manual testing)
# -------------------------------
if __name__ == "__main__":
    BUCKET_NAME = "ziya-57d19.firebasestorage.app"
    BLOB_NAME = "uploads/1756889356689_chb01_02.edf"

    # Inspect model (optional)
    inspect_onnx_model(ONNX_MODEL_PATH)

    # Run sync inference
    try:
        output = run_eeg_inference(BUCKET_NAME, BLOB_NAME)
        print("[FINAL OUTPUT]", output)
    except Exception as e:
        print(f"[FATAL] Inference failed: {e}")