import onnx
import asyncio 
import platform
import numpy as np
import onnxruntime as ort

from eeg_inference import preprocess
from eeg_inference import utils

def inspect_onnx_model(onnx_file_path):
    model = onnx.load(onnx_file_path)

    print("Model Metadata:")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")
    print(f"Doc String: {model.doc_string}")

    print("\nInputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        print(f"Type: {input.type}")
        if input.type.tensor_type.HasField("shape"):
            shape = []
            for dim in input.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            print(f"Shape: {shape}")
        else:
            print("Shape: Not specified")

    print("\nOutputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}")
        print(f"Type: {output.type}")

        if output.type.tensor_type.HasField("shape"):
            shape = []
            for dim in output.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            print(f"Shape: {shape}")
        else:
            print("Shape: Not specified")

    print("\nInitializers (weights/biases):")
    for initializer in model.graph.initializer:
        print(f"Name: {initializer.name}")
        print(f"Shape: {list(initializer.dims)}")

def get_available_providers():
    """
    Detect best execution providers depending on the device and OS.
    Priority order:
        1. CUDA (NVIDIA GPU)
        2. CoreML / MPS (Apple Silicon)
        3. CPU
    """
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    if platform.system() == "Darwin":
        if "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]        
    return ["CPUExecutionProvider"]

def create_session(onnx_path: str) -> ort.InferenceSession:
    """
    Initialize ONNX Runtime session with the best available provider.
    """
    providers = get_available_providers()
    print(f"[INFO] Using providers: {providers}")

    return ort.InferenceSession(onnx_path, providers=providers)

def get_io_names(session: ort.InferenceSession) -> tuple:
    """
    Retrieve model input and output names.
    """
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    return input_names, output_names

def run_inference_sync(session: ort.InferenceSession, file_path: str, config_file: str) -> np.ndarray:
    """
    Synchronous inference call.
    """
    inputs, config = preprocess.preprocess_eeg_data(file_path, config_file)
    input_names, output_names = get_io_names(session)
    eeg_results = []
    for input in inputs:
        result = session.run(output_names, {input_names[0]: input})
        if isinstance(result, list) and len(result) == 1:
            result = result[0]
        eeg_results.append(result)
    eeg_results = np.concatenate(eeg_results, axis=0) # gather from all batches
    eeg_results = utils.softmax(eeg_results) # apply softmax to raw logits
    class_indices = np.argmax(eeg_results, axis=1)
    confidence_levels = np.max(eeg_results, axis=1)

    events = []
    left, right = 0, -1

    event_annotation = config['event_annotation']

    def save_event(index: int, start: int, end: int):
        events.append({
            "event": event_annotation[class_indices[index]],
            "confidence_level": confidence_levels[index],
            "start_time": start,
            "end_time": end, 
        })

    window = config['window']

    for i in range(len(class_indices)):
        if not (0 <= class_indices[i] < len(event_annotation)):
            continue
        if i == 0 or class_indices[i - 1] != class_indices[i]:
            if i != 0:
                save_event(i - 1, left, right)
            left, right = right + 1, right + window
        elif class_indices[i - 1] == class_indices[i]:
            right += window
        if i == len(class_indices) - 1:
            save_event(i, left, right)
            left, right = right, right
    config['events'] = events
    return config

async def run_inference_async(session: ort.InferenceSession, file_path: str, config_file: str) -> np.ndarray:
    """
    Async wrapper for inference (useful in FastAPI or async workflows).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_inference_sync, session, file_path, config_file)

if __name__ == "__main__":
    onnx_model_path = "./models/kaz_data.onnx"
    sample_path = "./samples/kaz_sample1.edf"
    config_path = "./eeg_inference/kaz_configs.yaml"
    
    # inspect_onnx_model(onnx_model_path)
    session = create_session(onnx_model_path)

    output = run_inference_sync(session, sample_path, config_path)
    print(output)
    # async def main():
    #     output = await run_inference_async(session, "path/to/file/eeg.edf")
    # asyncio.run(main())