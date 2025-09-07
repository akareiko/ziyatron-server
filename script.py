import os
import tempfile
import asyncio
from google.cloud import storage
from eeg_inference.onnx_infer import create_session, run_inference_sync, run_inference_async, inspect_onnx_model

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./firebase-key.json"
ONNX_MODEL_PATH = "./models/best_model.onnx"

def download_from_firebase(bucket_name: str, blob_name: str) -> str:
    """
    Download a file from Firebase Storage (GCS bucket).
    Returns local file path.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, os.path.basename(blob_name))
    blob.download_to_filename(local_path)

    print(f"[INFO] File downloaded to {local_path}")
    return local_path


def run_eeg_inference(bucket_name: str, blob_name: str):
    """
    Full pipeline: download → preprocess → ONNX inference.
    """
    # 1. Download EEG file
    file_path = download_from_firebase(bucket_name, blob_name)

    # 2. Load ONNX model
    session = create_session(ONNX_MODEL_PATH)

    # 3. Run inference
    output = run_inference_sync(session, file_path)
    return output


async def run_eeg_inference_async(bucket_name: str, blob_name: str):
    file_path = download_from_firebase(bucket_name, blob_name)
    session = create_session(ONNX_MODEL_PATH)
    output = await run_inference_async(session, file_path)
    return output


if __name__ == "__main__":
    BUCKET_NAME = "ziya-57d19.firebasestorage.app"
    BLOB_NAME = "uploads/1756889356689_chb01_02.edf"

    # Just inspect model structure (optional)
    inspect_onnx_model(ONNX_MODEL_PATH)

    # Run sync inference
    output = run_eeg_inference(BUCKET_NAME, BLOB_NAME)
    print("[FINAL OUTPUT]", output)