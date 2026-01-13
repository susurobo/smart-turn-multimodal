"""
Real-time Multimodal Turn-Taking Prediction Server

Uses WebSocket for streaming audio, Silero VAD for speech detection,
and multimodal inference (audio + video) for turn prediction.
"""

import os
import time
import math
import urllib.request
from collections import deque
from threading import Lock

import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image

from inference_multimodal import (
    predict_endpoint_with_frames as predict_multimodal_frames,
    SAMPLING_RATE,
    VIDEO_FRAMES,
    VIDEO_SIZE,
    VIDEO_MEAN,
    VIDEO_STD,
)
from inference import predict_endpoint as predict_audio_only

# Face detection for cropping (matching training data)
try:
    import torch
    from facenet_pytorch import MTCNN

    FACE_DETECTOR = MTCNN(
        keep_all=False, select_largest=True, device=torch.device("cpu")
    )
    print("[FACE] MTCNN face detector loaded")
except ImportError:
    FACE_DETECTOR = None
    print("[FACE] facenet_pytorch not installed - face detection disabled")
    print("[FACE] Install with: pip install facenet-pytorch")


FACE_MARGIN = (
    20  # Smaller margin = tighter face crop (face fills more of frame)
)


def detect_and_crop_face(frame: np.ndarray, margin: int = FACE_MARGIN) -> tuple:
    """
    Detect face in frame and return crop coordinates.

    Args:
        frame: RGB numpy array (H, W, 3)
        margin: Pixels of margin around detected face

    Returns:
        (x1, y1, x2, y2) crop box, or None if no face detected
    """
    if FACE_DETECTOR is None:
        return None

    h, w = frame.shape[:2]

    # MTCNN expects RGB PIL Image or numpy array
    boxes, _ = FACE_DETECTOR.detect(frame)

    if boxes is None or len(boxes) == 0:
        return None

    # Get largest face box
    box = boxes[0]
    x1, y1, x2, y2 = map(int, box)

    # Add margin
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    # Make it square (centered)
    crop_w = x2 - x1
    crop_h = y2 - y1
    side = max(crop_w, crop_h)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    final_x1 = max(0, center_x - side // 2)
    final_y1 = max(0, center_y - side // 2)
    final_x2 = min(w, final_x1 + side)
    final_y2 = min(h, final_y1 + side)

    return (final_x1, final_y1, final_x2, final_y2)


def crop_and_resize_frames(frames: list, target_size: int = VIDEO_SIZE) -> list:
    """
    Detect face in first frame, then crop all frames to that region.

    Args:
        frames: List of RGB numpy arrays (H, W, 3)
        target_size: Output size (default 112 for model)

    Returns:
        List of cropped and resized frames
    """
    if not frames:
        return frames

    # Detect face in middle frame (more likely to have stable face position)
    mid_idx = len(frames) // 2
    face_box = detect_and_crop_face(frames[mid_idx])

    if face_box is None:
        # Try first frame as fallback
        face_box = detect_and_crop_face(frames[0])

    if face_box is None:
        print("[FACE] No face detected - using center crop")
        # Fallback: center crop to square
        h, w = frames[0].shape[:2]
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        face_box = (x1, y1, x1 + side, y1 + side)
    else:
        x1, y1, x2, y2 = face_box
        print(
            f"[FACE] Detected face: ({x1},{y1}) to ({x2},{y2}) = {x2 - x1}x{y2 - y1}px"
        )

    x1, y1, x2, y2 = face_box

    # Crop and resize all frames
    cropped_frames = []
    for frame in frames:
        # Crop
        cropped = frame[y1:y2, x1:x2]

        # Resize to target size using PIL (matches training)
        img = Image.fromarray(cropped)
        img = img.resize((target_size, target_size), Image.BILINEAR)
        cropped_frames.append(np.array(img))

    return cropped_frames


def frames_to_tensor(frames: list, flip_horizontal: bool = True) -> np.ndarray:
    """
    Convert list of RGB frames to model tensor format.

    Args:
        frames: List of numpy arrays (H, W, 3) uint8
        flip_horizontal: If True, flip frames horizontally (webcam selfie -> normal view)

    Returns:
        numpy array (1, 3, 32, 112, 112) float32 normalized
    """
    # Ensure we have exactly VIDEO_FRAMES frames
    if len(frames) < VIDEO_FRAMES:
        # Pad with last frame
        while len(frames) < VIDEO_FRAMES:
            frames.append(
                frames[-1]
                if frames
                else np.zeros((VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.uint8)
            )
    elif len(frames) > VIDEO_FRAMES:
        # Take last VIDEO_FRAMES
        frames = frames[-VIDEO_FRAMES:]

    # Flip horizontally if needed (webcam selfie view -> normal view)
    if flip_horizontal:
        frames = [np.fliplr(f) for f in frames]

    # Stack frames: (32, 112, 112, 3)
    video = np.stack(frames).astype(np.float32) / 255.0

    # Permute to (3, 32, 112, 112)
    video = video.transpose(3, 0, 1, 2)

    # Normalize
    mean = VIDEO_MEAN.reshape(3, 1, 1, 1).astype(np.float32)
    std = VIDEO_STD.reshape(3, 1, 1, 1).astype(np.float32)
    video = (video - mean) / std

    # Add batch dimension -> (1, 3, 32, 112, 112)
    return np.expand_dims(video, axis=0).astype(np.float32)


# --- Configuration ---
CHUNK_SAMPLES = 512  # Silero VAD expects 512 samples at 16 kHz
CHUNK_MS = (CHUNK_SAMPLES / SAMPLING_RATE) * 1000

VAD_THRESHOLD = 0.5
PRE_SPEECH_MS = 200
STOP_MS = 1000
MAX_DURATION_SECONDS = 8

# Silero VAD model
SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
SILERO_VAD_PATH = "silero_vad.onnx"

# Multimodal model
MULTIMODAL_MODEL_PATH = "model_fixed.onnx"

# Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "smart-turn-multimodal-demo"
# Use eventlet for proper WebSocket support, fall back to threading
try:
    import eventlet

    eventlet.monkey_patch()
    async_mode = "eventlet"
except ImportError:
    async_mode = "threading"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode=async_mode,
    max_http_buffer_size=50 * 1024 * 1024,  # 50MB for video blobs
    ping_timeout=60,
    ping_interval=25,
)

# Per-client state
client_states = {}
client_lock = Lock()


class SileroVAD:
    """Minimal Silero VAD ONNX wrapper for 16 kHz, mono, chunk=512."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.context_size = 64
        self._state = None
        self._context = None
        self._last_reset_time = time.time()
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def reset(self):
        self._init_states()
        self._last_reset_time = time.time()

    def prob(self, chunk_f32: np.ndarray) -> float:
        """Compute speech probability for chunk of 512 samples."""
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK_SAMPLES:
            raise ValueError(
                f"Expected {CHUNK_SAMPLES} samples, got {x.shape[1]}"
            )
        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64),
        }
        out, self._state = self.session.run(None, ort_inputs)
        self._context = x[:, -self.context_size :]

        return float(out[0][0])


def ensure_vad_model():
    """Download Silero VAD model if not present."""
    if not os.path.exists(SILERO_VAD_PATH):
        print("Downloading Silero VAD model...")
        urllib.request.urlretrieve(SILERO_VAD_URL, SILERO_VAD_PATH)
        print("Downloaded.")
    return SILERO_VAD_PATH


class ClientState:
    """Per-client state for audio buffering and VAD."""

    def __init__(self, sid: str):
        self.sid = sid
        self.vad = SileroVAD(ensure_vad_model())

        # Chunk counts
        self.pre_chunks = math.ceil(PRE_SPEECH_MS / CHUNK_MS)
        self.stop_chunks = math.ceil(STOP_MS / CHUNK_MS)
        self.max_chunks = math.ceil(
            MAX_DURATION_SECONDS / (CHUNK_SAMPLES / SAMPLING_RATE)
        )

        # Buffers
        self.pre_buffer = deque(maxlen=self.pre_chunks)
        self.segment = []
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger = 0

        # Audio accumulator for partial chunks
        self.audio_accumulator = np.array([], dtype=np.float32)

    def reset_segment(self):
        """Reset segment state after processing."""
        self.segment.clear()
        self.speech_active = False
        self.trailing_silence = 0
        self.since_trigger = 0
        self.pre_buffer.clear()
        self.vad.reset()


# Global VAD model (ensure downloaded on startup)
ensure_vad_model()


@app.route("/")
def index():
    return render_template("multimodal.html")


@socketio.on("connect")
def handle_connect():
    sid = request.sid
    with client_lock:
        client_states[sid] = ClientState(sid)
    print(f"Client connected: {sid}")
    emit("status", {"message": "Connected. Waiting for audio..."})


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    with client_lock:
        if sid in client_states:
            del client_states[sid]
    print(f"Client disconnected: {sid}")


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """
    Receive audio chunk from client.
    Data is base64-encoded PCM float32 samples at 16kHz.
    """
    import base64

    sid = request.sid
    with client_lock:
        state = client_states.get(sid)
    if state is None:
        return

    # Decode base64 to float32 array
    try:
        audio_b64 = data.get("audio_b64") if isinstance(data, dict) else data
        audio_bytes = base64.b64decode(audio_b64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception as e:
        print(f"Error decoding audio: {e}")
        return

    # Accumulate audio
    state.audio_accumulator = np.concatenate(
        [state.audio_accumulator, audio_data]
    )

    # Process complete chunks
    while len(state.audio_accumulator) >= CHUNK_SAMPLES:
        chunk = state.audio_accumulator[:CHUNK_SAMPLES]
        state.audio_accumulator = state.audio_accumulator[CHUNK_SAMPLES:]

        # Run VAD
        is_speech = state.vad.prob(chunk) > VAD_THRESHOLD

        if not state.speech_active:
            # Pre-speech buffering
            state.pre_buffer.append(chunk)
            if is_speech:
                # Speech started
                state.segment = list(state.pre_buffer)
                state.segment.append(chunk)
                state.speech_active = True
                state.trailing_silence = 0
                state.since_trigger = 1
                emit("vad_status", {"speaking": True})
        else:
            # In active speech segment
            state.segment.append(chunk)
            state.since_trigger += 1

            if is_speech:
                state.trailing_silence = 0
            else:
                state.trailing_silence += 1

            # Check end conditions
            if (
                state.trailing_silence >= state.stop_chunks
                or state.since_trigger >= state.max_chunks
            ):
                # Speech ended - request video from client
                audio_array = np.concatenate(state.segment, dtype=np.float32)
                duration_sec = len(audio_array) / SAMPLING_RATE

                emit("vad_status", {"speaking": False})
                emit(
                    "request_video",
                    {
                        "duration_sec": duration_sec,
                        "audio_samples": len(audio_array),
                    },
                )

                # Store audio for when video arrives
                state.pending_audio = audio_array
                state.reset_segment()


@socketio.on("video_response")
def handle_video_response(data):
    """
    Receive raw video frames from client after speech ended.
    Frames are pre-processed RGB arrays at 112x112.
    Run multimodal inference and return result.
    """
    import base64

    sid = request.sid
    with client_lock:
        state = client_states.get(sid)
    if state is None:
        return

    audio_array = getattr(state, "pending_audio", None)
    if audio_array is None:
        emit("error", {"message": "No pending audio for inference"})
        return

    duration_sec = len(audio_array) / SAMPLING_RATE
    emit("status", {"message": f"Processing {duration_sec:.1f}s segment..."})

    # Process raw frames from client
    pixel_values = None
    frames_b64 = data.get("frames")

    if frames_b64 and len(frames_b64) > 0:
        try:
            width = data.get("width", 480)
            height = data.get("height", 480)

            # Decode frames from base64
            frames = []
            for b64 in frames_b64:
                rgb_bytes = base64.b64decode(b64)
                rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(
                    height, width, 3
                )
                frames.append(rgb)

            print(f"[VIDEO] Received {len(frames)} frames at {width}x{height}")

            # Flip horizontally first (webcam selfie -> normal view)
            frames = [np.fliplr(f) for f in frames]
            print(
                "[VIDEO] Applied horizontal flip (webcam selfie -> normal view)"
            )

            # Face detection and cropping (matching training data)
            frames = crop_and_resize_frames(frames, target_size=VIDEO_SIZE)

            # Debug: Check frame stats after cropping
            first_frame = frames[0]
            last_frame = frames[-1]
            print(
                f"[VIDEO] After crop - First frame: min={first_frame.min()}, max={first_frame.max()}, mean={first_frame.mean():.1f}"
            )

            # Save debug frame to inspect visually
            try:
                debug_img = Image.fromarray(last_frame, mode="RGB")
                debug_img.save("/tmp/debug_frame.png")
                print(
                    f"[VIDEO] Saved debug frame ({last_frame.shape}) to /tmp/debug_frame.png"
                )
            except Exception as e:
                print(f"[VIDEO] Could not save debug frame: {e}")

            # Convert frames to model format (C, T, H, W)
            # flip already applied above, so pass False here
            pixel_values = frames_to_tensor(frames, flip_horizontal=False)
            print(
                f"[VIDEO] Tensor shape: {pixel_values.shape}, dtype: {pixel_values.dtype}"
            )
            print(
                f"[VIDEO] Tensor stats - min: {pixel_values.min():.3f}, max: {pixel_values.max():.3f}, mean: {pixel_values.mean():.3f}"
            )
            # Per-channel stats (should be ~0 mean after normalization if data is correct)
            for c, name in enumerate(["R", "G", "B"]):
                ch = pixel_values[0, c]
                print(
                    f"[VIDEO] Channel {name}: mean={ch.mean():.3f}, std={ch.std():.3f}"
                )

        except Exception as e:
            print(f"[VIDEO] Error processing frames: {e}")
            import traceback

            traceback.print_exc()
            pixel_values = None
    else:
        print(
            f"[VIDEO] No frames received! frames_b64={frames_b64 is not None}, len={len(frames_b64) if frames_b64 else 0}"
        )

    # Run both inferences
    try:
        # Audio-only inference
        print("[INFERENCE] Running audio-only inference...")
        t0_audio = time.perf_counter()
        result_audio = predict_audio_only(audio_array)
        audio_time_ms = (time.perf_counter() - t0_audio) * 1000
        print(
            f"[INFERENCE] Audio-only: {result_audio['probability']:.3f} ({audio_time_ms:.0f}ms)"
        )

        # Multimodal inference with raw frames
        has_video = pixel_values is not None
        n_frames = pixel_values.shape[2] if has_video else 0
        print(
            f"[INFERENCE] Running multimodal inference (video={'YES' if has_video else 'NO'}, frames={n_frames})..."
        )

        # Debug: Show frame variation to check temporal dynamics
        if has_video:
            frame_means = [
                pixel_values[0, :, t, :, :].mean()
                for t in range(min(5, n_frames))
            ]
            # Check frame-to-frame difference (should show motion)
            frame_diffs = []
            for t in range(1, min(5, n_frames)):
                diff = np.abs(
                    pixel_values[0, :, t] - pixel_values[0, :, t - 1]
                ).mean()
                frame_diffs.append(diff)
            print(
                f"[INFERENCE] First 5 frame means: {[f'{m:.3f}' for m in frame_means]}"
            )
            print(
                f"[INFERENCE] Frame-to-frame diffs: {[f'{d:.4f}' for d in frame_diffs]}"
            )

        t0_mm = time.perf_counter()
        result_mm = predict_multimodal_frames(
            audio_array,
            pixel_values=pixel_values,
            model_path=MULTIMODAL_MODEL_PATH,
        )
        mm_time_ms = (time.perf_counter() - t0_mm) * 1000
        print(
            f"[INFERENCE] Multimodal: {result_mm['probability']:.3f} ({mm_time_ms:.0f}ms)"
        )

        # Log comparison
        diff = result_mm["probability"] - result_audio["probability"]
        print(f"[INFERENCE] Difference (MM - Audio): {diff:+.3f}")

        emit(
            "prediction",
            {
                # Audio-only results
                "audio_prediction": result_audio["prediction"],
                "audio_probability": result_audio["probability"],
                "audio_time_ms": audio_time_ms,
                # Multimodal results
                "mm_prediction": result_mm["prediction"],
                "mm_probability": result_mm["probability"],
                "mm_time_ms": mm_time_ms,
                # Metadata
                "duration_sec": duration_sec,
                # Debug info
                "debug_has_video": has_video,
                "debug_frame_count": len(frames_b64) if frames_b64 else 0,
            },
        )
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback

        traceback.print_exc()
        emit("error", {"message": f"Inference error: {str(e)}"})
    finally:
        state.pending_audio = None

    emit("status", {"message": "Listening..."})


if __name__ == "__main__":
    print("=" * 60)
    print("Multimodal Turn-Taking Demo")
    print("=" * 60)
    print(f"VAD Model: {SILERO_VAD_PATH}")
    print(f"Multimodal Model: {MULTIMODAL_MODEL_PATH}")
    print(f"VAD Threshold: {VAD_THRESHOLD}")
    print(f"Stop after silence: {STOP_MS}ms")
    print(f"Max duration: {MAX_DURATION_SECONDS}s")
    print("=" * 60)
    print("Starting server at http://localhost:5001")
    print("=" * 60)

    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
