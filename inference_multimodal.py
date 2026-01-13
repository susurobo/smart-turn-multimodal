import numpy as np
import onnxruntime as ort
import av
import os
from collections import deque
from transformers import WhisperFeatureExtractor

from audio_utils import truncate_audio_to_last_n_seconds

# --- CONFIG ---
ONNX_MODEL_PATH = "smart-turn-multimodal.onnx"

# Audio constants
SAMPLING_RATE = 16000
AUDIO_SECONDS = 8

# Video constants
VIDEO_FRAMES = 32
VIDEO_SIZE = 112
VIDEO_MEAN = np.array([0.43216, 0.394666, 0.37645])
VIDEO_STD = np.array([0.22803, 0.22145, 0.216989])


def build_session(onnx_path):
    """Build ONNX inference session with optimized settings."""
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so)


# Lazy-loaded globals
_feature_extractor = None
_session = None
_current_model_path = None


def get_feature_extractor():
    """Get or create the feature extractor (lazy initialization)."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = WhisperFeatureExtractor(chunk_length=AUDIO_SECONDS)
    return _feature_extractor


def get_session(model_path=None):
    """Get or create the ONNX session (lazy initialization)."""
    global _session, _current_model_path
    if model_path is None:
        model_path = ONNX_MODEL_PATH
    if _session is None or _current_model_path != model_path:
        _session = build_session(model_path)
        _current_model_path = model_path
    return _session


def process_video_frames(video_path):
    """
    Decodes, resizes, and normalizes video frames.

    Args:
        video_path: Path to video file, or None for audio-only inference

    Returns:
        numpy array of shape (1, 3, 32, 112, 112)
    """
    # Handle missing video -> zero tensor
    if not video_path or not os.path.exists(video_path):
        return np.zeros(
            (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
        )

    try:
        # Use context manager to ensure memory is freed immediately
        with av.open(video_path) as container:
            # Use deque to efficiently keep only the last VIDEO_FRAMES frames
            frame_buffer = deque(maxlen=VIDEO_FRAMES)

            # Stream, resize, and store
            for frame in container.decode(video=0):
                # Resize to 112x112 immediately (memory saving)
                img = frame.to_image().resize((VIDEO_SIZE, VIDEO_SIZE))
                img_np = np.array(img, dtype=np.float32) / 255.0
                frame_buffer.append(img_np)

            frames = list(frame_buffer)

        # Check if we have enough frames
        if not frames:
            return np.zeros(
                (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )

        # Pad with last frame if fewer than VIDEO_FRAMES
        while len(frames) < VIDEO_FRAMES:
            frames.append(frames[-1])

        # Stack frames: (32, 112, 112, 3)
        video = np.stack(frames)

        # Permute to (3, 32, 112, 112)
        video = video.transpose(3, 0, 1, 2)

        # Normalize with correct broadcasting shape
        mean = VIDEO_MEAN.reshape(3, 1, 1, 1).astype(np.float32)
        std = VIDEO_STD.reshape(3, 1, 1, 1).astype(np.float32)
        video = (video - mean) / std

        # Add batch dimension -> (1, 3, 32, 112, 112)
        return np.expand_dims(video, axis=0).astype(np.float32)

    except Exception as e:
        print(f"Video processing error: {e}")
        return np.zeros(
            (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
        )


def predict_endpoint(audio_array, video_path=None, model_path=None):
    """
    Predict whether a turn is complete using multimodal (audio + video) input.

    Args:
        audio_array: Numpy array containing audio samples at 16kHz
        video_path: Optional path to video file. If None, performs audio-only inference.
        model_path: Optional path to ONNX model. If None, uses default model.

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """
    # Get lazy-loaded components
    feature_extractor = get_feature_extractor()
    session = get_session(model_path)

    # --- AUDIO PROCESSING ---
    # Truncate to last 8 seconds (keeping the end) or pad to 8 seconds
    audio_array = truncate_audio_to_last_n_seconds(
        audio_array, n_seconds=AUDIO_SECONDS
    )

    # Process audio using Whisper's feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=AUDIO_SECONDS * SAMPLING_RATE,
        truncation=True,
        do_normalize=True,
    )

    # Extract features and ensure correct shape for ONNX
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # (1, 80, 800)

    # --- VIDEO PROCESSING ---
    pixel_values = process_video_frames(video_path)  # (1, 3, 32, 112, 112)

    # --- INFERENCE ---
    outputs = session.run(
        None, {"input_features": input_features, "pixel_values": pixel_values}
    )

    # Extract probability (ONNX model returns sigmoid probabilities)
    probability = outputs[0][0].item()

    # Make prediction (1 for Complete, 0 for Incomplete)
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }


def predict_endpoint_with_frames(
    audio_array, pixel_values=None, model_path=None
):
    """
    Predict whether a turn is complete using multimodal (audio + video) input.
    This version accepts pre-processed pixel_values directly (no video file).

    Args:
        audio_array: Numpy array containing audio samples at 16kHz
        pixel_values: Pre-processed video tensor (1, 3, 32, 112, 112) or None
        model_path: Optional path to ONNX model. If None, uses default model.

    Returns:
        Dictionary containing prediction results:
        - prediction: 1 for complete, 0 for incomplete
        - probability: Probability of completion (sigmoid output)
    """
    # Get lazy-loaded components
    feature_extractor = get_feature_extractor()
    session = get_session(model_path)

    # --- AUDIO PROCESSING ---
    audio_array = truncate_audio_to_last_n_seconds(
        audio_array, n_seconds=AUDIO_SECONDS
    )

    inputs = feature_extractor(
        audio_array,
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=AUDIO_SECONDS * SAMPLING_RATE,
        truncation=True,
        do_normalize=True,
    )

    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)  # (1, 80, 800)

    # --- VIDEO PROCESSING ---
    if pixel_values is None:
        # No video provided - use zero tensor
        pixel_values = np.zeros(
            (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
        )

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


def load_audio(audio_path):
    """Load audio file and return numpy array at 16kHz."""
    import librosa

    audio_array, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    return audio_array.astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multimodal turn-taking prediction (audio + optional video)"
    )
    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        help="Path to audio file (WAV, 16kHz recommended). If not provided, uses random dummy audio.",
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default=None,
        help="Optional path to video file (MP4). If not provided, performs audio-only inference.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=ONNX_MODEL_PATH,
        help=f"Path to ONNX model file (default: {ONNX_MODEL_PATH})",
    )

    args = parser.parse_args()

    # Show model being used
    print(f"Model: {args.model}")

    # Load audio
    if args.audio:
        print(f"Loading audio: {args.audio}")
        audio_array = load_audio(args.audio)
        print(f"  Duration: {len(audio_array) / SAMPLING_RATE:.2f}s")
    else:
        print("No audio file provided, using random dummy audio (1 second)")
        audio_array = np.random.randn(SAMPLING_RATE).astype(np.float32)

    # Video info
    if args.video:
        print(f"Loading video: {args.video}")
    else:
        print("No video provided, performing audio-only inference")

    # Run prediction
    print("\nRunning inference...")
    result = predict_endpoint(
        audio_array, video_path=args.video, model_path=args.model
    )

    # Display results
    print(f"\n{'=' * 40}")
    print(
        f"Prediction:  {result['prediction']} ({'Complete' if result['prediction'] == 1 else 'Incomplete'})"
    )
    print(f"Probability: {result['probability']:.4f}")
    print(f"{'=' * 40}")

    if result["probability"] > 0.5:
        print("✅ Turn Complete - User has finished speaking")
    else:
        print("❌ Still Speaking - User may continue")
