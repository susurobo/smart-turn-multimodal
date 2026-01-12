import numpy as np
import onnxruntime as ort
import av
import os
from transformers import WhisperFeatureExtractor
from audio_utils import truncate_audio_to_last_n_seconds

# --- CONFIG ---
ONNX_MODEL_PATH = "smart-turn-multimodal.onnx"
VIDEO_FRAMES = 32
VIDEO_SIZE = 112
VIDEO_MEAN = np.array([0.43216, 0.394666, 0.37645]).reshape(3, 1, 1)
VIDEO_STD = np.array([0.22803, 0.22145, 0.216989]).reshape(3, 1, 1)


def build_session(onnx_path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)


feature_extractor = WhisperFeatureExtractor(chunk_length=8)
session = build_session(ONNX_MODEL_PATH)


def process_video_frames(video_path):
    """
    Decodes, Resizes, and Normalizes video frames.
    Returns: (1, 3, 32, 112, 112) numpy array
    """
    # 1. Handle Missing Video -> Zero Tensor
    if not video_path or not os.path.exists(video_path):
        return np.zeros(
            (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
        )

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]

        # 2. Decode & Resize
        frames = []
        for frame in container.decode(stream):
            # Resize directly on PIL image
            img = frame.to_image().resize((VIDEO_SIZE, VIDEO_SIZE))
            frames.append(img)

        # 3. Temporal Slicing (Last 32 frames)
        if len(frames) > VIDEO_FRAMES:
            frames = frames[-VIDEO_FRAMES:]
        elif len(frames) < 8:
            # Too short to be useful, return zeros
            return np.zeros(
                (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )

        # 4. Convert to Numpy & Stack
        # Shape: (T, H, W, C)
        arr = np.stack([np.array(f) for f in frames])
        arr = arr.astype(np.float32) / 255.0

        # 5. Permute to (C, T, H, W)
        arr = np.transpose(arr, (3, 0, 1, 2))

        # 6. Normalize
        arr = (arr - VIDEO_MEAN[:, None, None]) / VIDEO_STD[:, None, None]

        # 7. Padding (if between 8 and 32 frames)
        if arr.shape[1] < VIDEO_FRAMES:
            pad_amt = VIDEO_FRAMES - arr.shape[1]
            pad_width = ((0, 0), (0, pad_amt), (0, 0), (0, 0))
            arr = np.pad(arr, pad_width, mode="constant")

        # 8. Add Batch Dimension -> (1, 3, 32, 112, 112)
        return np.expand_dims(arr, axis=0).astype(np.float32)

    except Exception as e:
        print(f"Video processing error: {e}")
        return np.zeros(
            (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
        )


def predict_endpoint(audio_array, video_path=None):
    """
    Multimodal prediction.
    """
    # --- AUDIO PROCESSING ---
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # (1, 80, 800)

    # --- VIDEO PROCESSING ---
    pixel_values = process_video_frames(video_path)  # (1, 3, 32, 112, 112)

    # --- INFERENCE ---
    outputs = session.run(
        None, {"input_features": input_features, "pixel_values": pixel_values}
    )

    probability = outputs[0][0].item()
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }


# Example usage
if __name__ == "__main__":
    # Dummy inputs
    dummy_audio = np.random.randn(16000).astype(np.float32)
    dummy_video_path = "video/test_clip_face_cropped.mp4"

    result = predict_endpoint(dummy_audio, video_path=dummy_video_path)

    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    if result["probability"] > 0.5:
        print("✅ Turn Complete")
    else:
        print("❌ Still Speaking")
