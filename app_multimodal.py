"""
Real-time Multimodal Turn-Taking Prediction Server

Uses WebSocket for streaming audio, Silero VAD for speech detection,
and multimodal inference (audio + video) for turn prediction.
"""

import os
import time
import math
import tempfile
import urllib.request
from collections import deque
from threading import Lock

import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

from inference_multimodal import (
    predict_endpoint as predict_multimodal,
    SAMPLING_RATE,
)
from inference import predict_endpoint as predict_audio_only

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
    Receive video blob from client after speech ended.
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

    video_b64 = data.get("video_b64")
    duration_sec = len(audio_array) / SAMPLING_RATE

    emit("status", {"message": f"Processing {duration_sec:.1f}s segment..."})

    # Save video to temp file and convert to MP4 for better compatibility
    video_path = None
    webm_path = None
    if video_b64:
        try:
            import subprocess

            video_bytes = base64.b64decode(video_b64)

            # Save as webm first
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(video_bytes)
                webm_path = f.name

            # Convert to MP4 using ffmpeg for better PyAV compatibility
            mp4_path = webm_path.replace(".webm", ".mp4")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    webm_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "28",
                    "-an",  # No audio needed
                    mp4_path,
                ],
                capture_output=True,
                timeout=10,
            )

            if result.returncode == 0 and os.path.exists(mp4_path):
                video_path = mp4_path
                print(
                    f"Converted video: {len(video_bytes)} bytes -> {video_path}"
                )
            else:
                print(
                    f"FFmpeg conversion failed: {result.stderr.decode()[:200]}"
                )
                video_path = None

        except FileNotFoundError:
            print("FFmpeg not found - video processing disabled")
            video_path = None
        except Exception as e:
            print(f"Error processing video: {e}")
            video_path = None

    # Run both inferences
    try:
        # Audio-only inference
        t0_audio = time.perf_counter()
        result_audio = predict_audio_only(audio_array)
        audio_time_ms = (time.perf_counter() - t0_audio) * 1000

        # Multimodal inference
        t0_mm = time.perf_counter()
        result_mm = predict_multimodal(
            audio_array, video_path=video_path, model_path=MULTIMODAL_MODEL_PATH
        )
        mm_time_ms = (time.perf_counter() - t0_mm) * 1000

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
            },
        )
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback

        traceback.print_exc()
        emit("error", {"message": f"Inference error: {str(e)}"})
    finally:
        # Cleanup temp video files
        if webm_path and os.path.exists(webm_path):
            os.unlink(webm_path)
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)

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
