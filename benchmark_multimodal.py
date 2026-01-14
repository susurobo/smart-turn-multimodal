import os
import time
import subprocess
import librosa
import pandas as pd
import matplotlib
from datetime import datetime

# Force headless backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from collections import deque

import modal
import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor

from datasets import load_dataset, load_from_disk

# --- CONSTANTS ---
SAMPLING_RATE = 16000
N_MELS = 80
N_FRAMES = 800
AUDIO_SECONDS = 8
FEATURE_SHAPE = (1, N_MELS, N_FRAMES)
# Video Constants
VIDEO_FRAMES = 32
VIDEO_SIZE = 112
VIDEO_MEAN_BASE = np.array([0.43216, 0.394666, 0.37645])
VIDEO_STD_BASE = np.array([0.22803, 0.22145, 0.216989])
# Performance test defaults
PERF_RUNS_DEFAULT = 100
PERF_WARMUP_DEFAULT = 10

app = modal.App("endpointing-benchmark-mm")
volume = modal.Volume.from_name("endpointing", create_if_missing=False)

# --- IMAGE DEFINITION ---
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "ffmpeg", "libsndfile1", "cmake", "wget")
    .pip_install(
        "numpy<2.0",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.36.2",
        "datasets==2.21.0",
        "scikit-learn",
        "onnx",
        "wandb",
        "accelerate",
        "evaluate",
        "jiwer",
        "av",
        "librosa",
        "soundfile",
        "pandas",
        "tqdm",
        "matplotlib",
        "pillow",
    )
    .run_commands(
        "pip install onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
    )
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/lib/x86_64-linux-gnu"
        }
    )
    .add_local_file(
        "train_multimodal.py", remote_path="/root/train_multimodal.py"
    )
    .add_local_python_source("logger", "audio_utils", "train")
)


def log_progress(message: str, level: str = "INFO"):
    """Log progress with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] [{level}] {message}")


def get_gpu_model_name() -> str:
    """Get the GPU model name using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0]
            if " " in gpu_name:
                return gpu_name.split()[-1]
            return gpu_name
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass
    return "GPU"


def _latency_stats(times: List[float]) -> Dict[str, float]:
    """Compute latency statistics from a list of timing measurements."""
    p50 = np.percentile(times, 50) * 1000
    p90 = np.percentile(times, 90) * 1000
    mean = np.mean(times) * 1000
    return {
        "latency_ms_p50": float(p50),
        "latency_ms_p90": float(p90),
        "latency_ms_mean": float(mean),
        "throughput_sps": float(1.0 / np.mean(times))
        if np.mean(times) > 0
        else 0,
    }


def build_session(onnx_path: str, providers: List[str]) -> ort.InferenceSession:
    """Build ONNX inference session with specified providers."""
    log_progress(f"Building ONNX session with providers: {providers}")
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(
        onnx_path, sess_options=sess_options, providers=providers
    )
    log_progress(
        f"Session created successfully. Active provider: {session.get_providers()[0]}"
    )
    return session


def run_perf_multimodal(
    session: ort.InferenceSession,
    runs: int = PERF_RUNS_DEFAULT,
    warmup: int = PERF_WARMUP_DEFAULT,
    include_video: bool = True,
) -> Dict[str, float]:
    """
    Run inference performance test with pre-computed zero features.

    Args:
        session: ONNX inference session
        runs: Number of timed runs
        warmup: Number of warmup runs
        include_video: If True, use video input; if False, use zero video tensor
    """
    provider_name = session.get_providers()[0]
    mode = "with video" if include_video else "audio-only"
    log_progress(
        f"Running inference performance test on {provider_name} ({mode}) "
        f"({warmup} warmup + {runs} runs)"
    )

    # Create zero inputs
    audio_inp = np.zeros(FEATURE_SHAPE, dtype=np.float32)
    video_inp = np.zeros(
        (1, 3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
    )

    feed = {
        "input_features": audio_inp,
        "pixel_values": video_inp,
    }

    # Warmup
    log_progress("  Warming up inference session...")
    for i in range(warmup):
        session.run(None, feed)
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

    # Timed runs
    log_progress("  Running timed inference...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append(time.perf_counter() - t0)

        if (i + 1) % max(1, runs // 10) == 0:
            current_mean = np.mean(times) * 1000
            log_progress(
                f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms"
            )

    stats = _latency_stats(times)
    log_progress(
        f"  Inference {provider_name} ({mode}) complete - "
        f"Mean: {stats['latency_ms_mean']:.2f}ms, "
        f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms"
    )
    return stats


def run_video_preprocessing_perf(
    video_path: Optional[str] = None,
    runs: int = PERF_RUNS_DEFAULT,
    warmup: int = PERF_WARMUP_DEFAULT,
) -> Dict[str, float]:
    """
    Run video preprocessing performance test.

    If video_path is None, creates a synthetic test by measuring
    the zero-tensor creation overhead.
    """
    log_progress(
        f"Running video preprocessing performance test ({warmup} warmup + {runs} runs)"
    )

    if video_path and os.path.exists(video_path):
        log_progress(f"  Using real video file: {video_path}")
        # Warmup
        for i in range(warmup):
            _ = process_video_numpy(video_path)
            if (i + 1) % max(1, warmup // 4) == 0:
                log_progress(f"    Warmup progress: {i + 1}/{warmup}")

        # Timed runs
        times = []
        for i in range(runs):
            t0 = time.perf_counter()
            _ = process_video_numpy(video_path)
            times.append(time.perf_counter() - t0)

            if (i + 1) % max(1, runs // 10) == 0:
                current_mean = np.mean(times) * 1000
                log_progress(
                    f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms"
                )
    else:
        log_progress("  No video file provided, measuring zero-tensor creation")
        # Warmup
        for _ in range(warmup):
            _ = np.zeros(
                (3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )

        # Timed runs
        times = []
        for i in range(runs):
            t0 = time.perf_counter()
            _ = np.zeros(
                (3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )
            times.append(time.perf_counter() - t0)

    stats = _latency_stats(times)
    log_progress(
        f"  Video preprocessing complete - "
        f"Mean: {stats['latency_ms_mean']:.2f}ms, "
        f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms"
    )
    return stats


def run_audio_fe_perf(
    feature_extractor,
    runs: int = PERF_RUNS_DEFAULT,
    warmup: int = PERF_WARMUP_DEFAULT,
) -> Dict[str, float]:
    """Run audio feature extraction performance test using zero audio."""
    log_progress(
        f"Running audio feature extraction performance test ({warmup} warmup + {runs} runs)"
    )

    zero_audio = np.zeros(SAMPLING_RATE * AUDIO_SECONDS, dtype=np.float32)

    def extract_features():
        inputs = feature_extractor(
            zero_audio,
            sampling_rate=SAMPLING_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        return inputs.input_features[0][:, :N_FRAMES]

    # Warmup
    log_progress("  Warming up feature extractor...")
    for i in range(warmup):
        _ = extract_features()
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

    # Timed runs
    log_progress("  Running timed feature extraction...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        _ = extract_features()
        times.append(time.perf_counter() - t0)

        if (i + 1) % max(1, runs // 10) == 0:
            current_mean = np.mean(times) * 1000
            log_progress(
                f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms"
            )

    stats = _latency_stats(times)
    log_progress(
        f"  Audio feature extraction complete - "
        f"Mean: {stats['latency_ms_mean']:.2f}ms, "
        f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms"
    )
    return stats


def run_e2e_perf_multimodal(
    session: ort.InferenceSession,
    feature_extractor,
    video_path: Optional[str] = None,
    runs: int = PERF_RUNS_DEFAULT,
    warmup: int = PERF_WARMUP_DEFAULT,
) -> Dict[str, float]:
    """
    Run end-to-end performance test: audio feature extraction + video preprocessing + inference.
    """
    provider_name = session.get_providers()[0]
    has_video = video_path and os.path.exists(str(video_path))
    mode = "with video" if has_video else "audio-only"

    log_progress(
        f"Running end-to-end performance test on {provider_name} ({mode}) "
        f"({warmup} warmup + {runs} runs)"
    )

    zero_audio = np.zeros(SAMPLING_RATE * AUDIO_SECONDS, dtype=np.float32)

    def run_full_pipeline():
        # Audio feature extraction
        inputs = feature_extractor(
            zero_audio,
            sampling_rate=SAMPLING_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        audio_feats = inputs.input_features[0][:, :N_FRAMES]
        if audio_feats.shape[-1] < N_FRAMES:
            pad = np.zeros(
                (N_MELS, N_FRAMES - audio_feats.shape[-1]), dtype=np.float32
            )
            audio_feats = np.concatenate([audio_feats, pad], axis=1)
        audio_feats = audio_feats[np.newaxis, ...]  # Add batch dim

        # Video preprocessing
        if has_video:
            video_feats = process_video_numpy(video_path)
        else:
            video_feats = np.zeros(
                (3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )
        video_feats = video_feats[np.newaxis, ...]  # Add batch dim

        # Inference
        feed = {
            "input_features": audio_feats,
            "pixel_values": video_feats,
        }
        return session.run(None, feed)

    # Warmup
    log_progress("  Warming up end-to-end pipeline...")
    for i in range(warmup):
        _ = run_full_pipeline()
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

    # Timed runs
    log_progress("  Running timed end-to-end inference...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        _ = run_full_pipeline()
        times.append(time.perf_counter() - t0)

        if (i + 1) % max(1, runs // 10) == 0:
            current_mean = np.mean(times) * 1000
            log_progress(
                f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms"
            )

    stats = _latency_stats(times)
    log_progress(
        f"  End-to-end {provider_name} ({mode}) complete - "
        f"Mean: {stats['latency_ms_mean']:.2f}ms, "
        f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms"
    )
    return stats


def generate_output_path(
    onnx_path: str, run_description: str, extension: str = "md"
) -> str:
    """Generate dynamic output path based on ONNX path and run description."""
    model_name = None
    if onnx_path.startswith("/data/output/"):
        path_parts = onnx_path.split("/")
        if len(path_parts) >= 4:
            model_name = path_parts[3]

    if model_name is None:
        onnx_filename = os.path.basename(onnx_path)
        model_name = os.path.splitext(onnx_filename)[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/data/benchmark/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    return f"{output_dir}/{run_description}_{timestamp}.{extension}"


# --- VIDEO PREPROCESSING (STABLE) ---
def process_video_numpy(video_path: str) -> np.ndarray:
    import av

    if not video_path or not os.path.exists(video_path):
        return np.zeros(
            (3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
        )

    try:
        # Use context manager to ensure memory is freed immediately
        with av.open(video_path) as container:
            frame_buffer = deque(maxlen=VIDEO_FRAMES)

            # Stream, resize, and store
            for frame in container.decode(video=0):
                # Squash to 112x112 immediately (Memory saving)
                img = frame.to_image().resize((VIDEO_SIZE, VIDEO_SIZE))
                img_np = np.array(img, dtype=np.float32) / 255.0
                frame_buffer.append(img_np)

            frames = list(frame_buffer)

        if not frames:
            return np.zeros(
                (3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )

        while len(frames) < VIDEO_FRAMES:
            frames.append(frames[-1])

        video = np.stack(frames)  # (32, 112, 112, 3)
        video = video.transpose(3, 0, 1, 2)  # (3, 32, 112, 112)

        # Normalize
        mean = VIDEO_MEAN_BASE.reshape(3, 1, 1, 1).astype(np.float32)
        std = VIDEO_STD_BASE.reshape(3, 1, 1, 1).astype(np.float32)
        video = (video - mean) / std

        return video.astype(np.float32)

    except Exception:
        return np.zeros(
            (3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
        )


class MultimodalBenchmarkDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor, audio_only: bool = False):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.audio_only = audio_only  # Force zero video tensors for all samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # --- METADATA COLLECTION ---
        video_path = item.get("video_path", None)
        audio_path = item.get("audio_path", None) or item.get("audio", None)
        if isinstance(audio_path, dict):
            audio_path = audio_path.get("path", None)

        # Audio
        audio_array = None
        audio_duration_sec = 0.0
        audio_load_error = None

        if (
            "audio" in item
            and isinstance(item["audio"], dict)
            and "array" in item["audio"]
        ):
            audio_array = item["audio"]["array"]
            audio_duration_sec = len(audio_array) / SAMPLING_RATE
        elif "audio" in item and isinstance(item["audio"], str):
            audio_file_path = item["audio"]
            if not os.path.exists(audio_file_path):
                audio_load_error = f"File not found: {audio_file_path}"
            else:
                try:
                    audio_array, _ = librosa.load(
                        audio_file_path, sr=SAMPLING_RATE, mono=True
                    )
                    audio_duration_sec = len(audio_array) / SAMPLING_RATE
                except Exception as e:
                    audio_load_error = f"librosa error: {e}"
        else:
            audio_load_error = f"No audio field or wrong type: audio={'audio' in item}, type={type(item.get('audio', None))}"

        if audio_array is None:
            # Log first few failures for debugging
            if idx < 5:
                print(
                    f"[DEBUG] Sample {idx}: Audio load failed - {audio_load_error}"
                )
                print(f"[DEBUG]   Available keys: {list(item.keys())}")
                if "audio" in item:
                    print(
                        f"[DEBUG]   item['audio'] type: {type(item['audio'])}, value: {str(item['audio'])[:100]}"
                    )
            audio_array = np.zeros(16000 * 8, dtype=np.float32)

        # --- AUDIO TAIL ALIGNMENT ---
        max_samples = 16000 * 8
        if len(audio_array) > max_samples:
            audio_array = audio_array[-max_samples:]

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=480000,
            truncation=True,
        )
        feats = inputs.input_features[0][:, :N_FRAMES]
        if feats.shape[-1] < N_FRAMES:
            pad = np.zeros((80, N_FRAMES - feats.shape[-1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=1)

        # Force zero video tensors if audio_only mode
        if self.audio_only:
            pixel_values = np.zeros(
                (3, VIDEO_FRAMES, VIDEO_SIZE, VIDEO_SIZE), dtype=np.float32
            )
        else:
            pixel_values = process_video_numpy(video_path)
        has_video = video_path is not None and os.path.exists(str(video_path))

        label = item.get("endpoint_bool")
        if label is None:
            label = item.get("label", 0)

        file_id = item.get("file_name") or item.get("id") or str(idx)

        return {
            "input_features": feats,
            "pixel_values": pixel_values,
            "labels": label,
            "id": file_id,
            # Detailed metadata for per-file output
            "video_path": video_path or "",
            "audio_path": audio_path or "",
            "has_video": has_video,
            "audio_duration_sec": audio_duration_sec,
        }


@dataclass
class MultimodalCollator:
    def __call__(self, features):
        input_features = np.stack([f["input_features"] for f in features])
        pixel_values = np.stack([f["pixel_values"] for f in features])
        labels = np.array(
            [1 if f["labels"] else 0 for f in features], dtype=np.int32
        )
        ids = [f["id"] for f in features]

        # Pass through metadata
        video_paths = [f["video_path"] for f in features]
        audio_paths = [f["audio_path"] for f in features]
        has_video = [f["has_video"] for f in features]
        audio_durations = [f["audio_duration_sec"] for f in features]

        return {
            "input_features": input_features,
            "pixel_values": pixel_values,
            "labels": labels,
            "ids": ids,
            "video_paths": video_paths,
            "audio_paths": audio_paths,
            "has_video": has_video,
            "audio_durations": audio_durations,
        }


# --- MARKDOWN TABLE HELPER (from benchmark.py) ---
class MarkdownTable:
    """Render a padded/aligned Markdown table (monospace-friendly)."""

    def __init__(self, headers: List[str], align: Optional[List[str]] = None):
        self.headers = [str(h) for h in headers]
        self.rows: List[List[str]] = []
        self.align = align or ["left"] * len(self.headers)

    def add_row(self, row: List[Any]) -> None:
        self.rows.append(["" if v is None else str(v) for v in row])

    @staticmethod
    def _separator_cell(width: int, align: str) -> str:
        width = max(3, width)
        if align == "left":
            return ":" + ("-" * (width - 1))
        if align == "right":
            return ("-" * (width - 1)) + ":"
        if align == "center":
            return ":" + ("-" * (width - 2)) + ":"
        return "-" * width

    def render(self) -> str:
        widths = [max(3, len(h)) for h in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def _format_row(cells: List[str]) -> str:
            padded: List[str] = []
            for i, cell in enumerate(cells):
                if self.align[i] == "right":
                    padded.append(cell.rjust(widths[i]))
                elif self.align[i] == "center":
                    padded.append(cell.center(widths[i]))
                else:
                    padded.append(cell.ljust(widths[i]))
            return "| " + " | ".join(padded) + " |"

        header_line = _format_row(self.headers)
        sep_line = (
            "| "
            + " | ".join(
                self._separator_cell(widths[i], self.align[i])
                for i in range(len(widths))
            )
            + " |"
        )
        body_lines = [_format_row(r) for r in self.rows]
        return "\n".join([header_line, sep_line] + body_lines)


def compute_metrics_with_confusion(
    probs: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    """Compute metrics including false positive and false negative rates."""
    preds = (probs > 0.5).astype(int)

    false_positives = np.sum((preds == 1) & (labels == 0))
    false_negatives = np.sum((preds == 0) & (labels == 1))
    total = len(labels)

    return {
        "sample_count": int(total),
        "accuracy": float(accuracy_score(labels, preds)) * 100,
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "false_positive_rate": float(false_positives / total) * 100
        if total > 0
        else 0.0,
        "false_negative_rate": float(false_negatives / total) * 100
        if total > 0
        else 0.0,
    }


def format_markdown_report(results: Dict, gpu_model_name: str = "GPU") -> str:
    """Format results into a comprehensive Markdown report."""
    md_lines = []

    # Header
    audio_only = results.get("audio_only", False)
    mode_str = "Audio-Only" if audio_only else "Video+Audio"
    md_lines.append("# Multimodal Endpointing Benchmark Report")
    md_lines.append(f"\n**Model:** `{results['onnx_path']}`")
    md_lines.append(
        f"\n**Run Description:** {results.get('run_description', 'N/A')}"
    )
    md_lines.append(f"\n**Mode:** {mode_str}")
    md_lines.append(
        f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    )

    # Dataset Info
    md_lines.append("\n## Dataset Summary")
    md_lines.append(f"\n- **Total Samples:** {results['total_samples']:,}")
    md_lines.append(
        f"- **Samples with Video:** {results['samples_with_video']:,} ({results['video_percentage']:.1f}%)"
    )
    md_lines.append(
        f"- **Inference Time:** {results['inference_time_sec']:.2f}s"
    )
    md_lines.append(
        f"- **Throughput:** {results['throughput_sps']:.1f} samples/sec"
    )

    # Overall Metrics
    md_lines.append("\n## Overall Performance")
    overall = results["metrics"]
    overall_tbl = MarkdownTable(
        headers=["Metric", "Value"],
        align=["left", "right"],
    )
    overall_tbl.add_row(["Accuracy", f"{overall['accuracy']:.2f}%"])
    overall_tbl.add_row(["Precision", f"{overall['precision']:.3f}"])
    overall_tbl.add_row(["Recall", f"{overall['recall']:.3f}"])
    overall_tbl.add_row(["F1 Score", f"{overall['f1']:.3f}"])
    overall_tbl.add_row(
        ["False Positive Rate", f"{overall['false_positive_rate']:.2f}%"]
    )
    overall_tbl.add_row(
        ["False Negative Rate", f"{overall['false_negative_rate']:.2f}%"]
    )
    overall_tbl.add_row(["ROC AUC", f"{results.get('roc_auc', 0):.4f}"])
    overall_tbl.add_row(["PR AUC", f"{results.get('pr_auc', 0):.4f}"])
    md_lines.append("\n" + overall_tbl.render())

    # Video vs No-Video Breakdown
    if "metrics_with_video" in results and "metrics_no_video" in results:
        md_lines.append("\n## Performance by Modality")
        mod_tbl = MarkdownTable(
            headers=[
                "Modality",
                "Samples",
                "Accuracy (%)",
                "Precision",
                "Recall",
                "F1",
                "FPR (%)",
                "FNR (%)",
            ],
            align=[
                "left",
                "right",
                "right",
                "right",
                "right",
                "right",
                "right",
                "right",
            ],
        )

        if results["metrics_with_video"]["sample_count"] > 0:
            m = results["metrics_with_video"]
            mod_tbl.add_row(
                [
                    "With Video",
                    f"{m['sample_count']:,}",
                    f"{m['accuracy']:.2f}",
                    f"{m['precision']:.3f}",
                    f"{m['recall']:.3f}",
                    f"{m['f1']:.3f}",
                    f"{m['false_positive_rate']:.2f}",
                    f"{m['false_negative_rate']:.2f}",
                ]
            )

        if results["metrics_no_video"]["sample_count"] > 0:
            m = results["metrics_no_video"]
            mod_tbl.add_row(
                [
                    "Audio Only",
                    f"{m['sample_count']:,}",
                    f"{m['accuracy']:.2f}",
                    f"{m['precision']:.3f}",
                    f"{m['recall']:.3f}",
                    f"{m['f1']:.3f}",
                    f"{m['false_positive_rate']:.2f}",
                    f"{m['false_negative_rate']:.2f}",
                ]
            )

        md_lines.append("\n" + mod_tbl.render())

    # Optimal Threshold
    md_lines.append("\n## Threshold Analysis")
    md_lines.append(
        f"\n**Best Threshold:** {results['best_threshold']:.2f} (F1: {results['best_f1']:.4f})"
    )

    # Probability Distribution
    md_lines.append("\n## Probability Distribution")
    md_lines.append(f"\n- **Min:** {results['prob_min']:.4f}")
    md_lines.append(f"- **Max:** {results['prob_max']:.4f}")
    md_lines.append(f"- **Mean:** {results['prob_mean']:.4f}")
    md_lines.append(f"- **Std:** {results['prob_std']:.4f}")

    # Performance Results
    if any(k.startswith("perf_") for k in results.keys()):
        md_lines.append("\n## Inference Performance")

        # Direct Inference Performance
        if "perf_cpu" in results or "perf_gpu" in results:
            md_lines.append("\n### Direct Inference Performance")
            md_lines.append(
                "*Using pre-computed zero features (inference only)*"
            )
            direct_tbl = MarkdownTable(
                headers=[
                    "Provider",
                    "P50 Latency (ms)",
                    "P90 Latency (ms)",
                    "Mean Latency (ms)",
                    "Throughput (samples/sec)",
                ],
                align=["left", "right", "right", "right", "right"],
            )

            if "perf_cpu" in results and "note" not in results["perf_cpu"]:
                cpu_perf = results["perf_cpu"]
                direct_tbl.add_row(
                    [
                        "CPU",
                        f"{cpu_perf['latency_ms_p50']:.2f}",
                        f"{cpu_perf['latency_ms_p90']:.2f}",
                        f"{cpu_perf['latency_ms_mean']:.2f}",
                        f"{cpu_perf['throughput_sps']:.1f}",
                    ]
                )

            if "perf_gpu" in results and "note" not in results["perf_gpu"]:
                gpu_perf = results["perf_gpu"]
                direct_tbl.add_row(
                    [
                        gpu_model_name,
                        f"{gpu_perf['latency_ms_p50']:.2f}",
                        f"{gpu_perf['latency_ms_p90']:.2f}",
                        f"{gpu_perf['latency_ms_mean']:.2f}",
                        f"{gpu_perf['throughput_sps']:.1f}",
                    ]
                )
            md_lines.append("\n" + direct_tbl.render())

        # Feature Extraction Performance
        if "perf_audio_fe" in results:
            md_lines.append("\n### Audio Feature Extraction Performance")
            md_lines.append("*Whisper feature extraction from 8-second audio*")
            fe_perf = results["perf_audio_fe"]
            fe_tbl = MarkdownTable(
                headers=[
                    "Component",
                    "P50 Latency (ms)",
                    "P90 Latency (ms)",
                    "Mean Latency (ms)",
                    "Throughput (samples/sec)",
                ],
                align=["left", "right", "right", "right", "right"],
            )
            fe_tbl.add_row(
                [
                    "Audio Feature Extractor",
                    f"{fe_perf['latency_ms_p50']:.2f}",
                    f"{fe_perf['latency_ms_p90']:.2f}",
                    f"{fe_perf['latency_ms_mean']:.2f}",
                    f"{fe_perf['throughput_sps']:.1f}",
                ]
            )
            md_lines.append("\n" + fe_tbl.render())

        # Video Preprocessing Performance
        if "perf_video_preprocess" in results:
            md_lines.append("\n### Video Preprocessing Performance")
            md_lines.append("*Video decoding, resizing, and normalization*")
            vp_perf = results["perf_video_preprocess"]
            vp_tbl = MarkdownTable(
                headers=[
                    "Component",
                    "P50 Latency (ms)",
                    "P90 Latency (ms)",
                    "Mean Latency (ms)",
                    "Throughput (samples/sec)",
                ],
                align=["left", "right", "right", "right", "right"],
            )
            vp_tbl.add_row(
                [
                    "Video Preprocessor",
                    f"{vp_perf['latency_ms_p50']:.2f}",
                    f"{vp_perf['latency_ms_p90']:.2f}",
                    f"{vp_perf['latency_ms_mean']:.2f}",
                    f"{vp_perf['throughput_sps']:.1f}",
                ]
            )
            md_lines.append("\n" + vp_tbl.render())

        # End-to-End Performance
        if "perf_e2e_cpu" in results or "perf_e2e_gpu" in results:
            md_lines.append("\n### End-to-End Performance")
            md_lines.append(
                "*Audio feature extraction + video preprocessing + inference*"
            )
            e2e_tbl = MarkdownTable(
                headers=[
                    "Provider",
                    "P50 Latency (ms)",
                    "P90 Latency (ms)",
                    "Mean Latency (ms)",
                    "Throughput (samples/sec)",
                ],
                align=["left", "right", "right", "right", "right"],
            )

            if (
                "perf_e2e_cpu" in results
                and "note" not in results["perf_e2e_cpu"]
            ):
                e2e_cpu = results["perf_e2e_cpu"]
                e2e_tbl.add_row(
                    [
                        "CPU",
                        f"{e2e_cpu['latency_ms_p50']:.2f}",
                        f"{e2e_cpu['latency_ms_p90']:.2f}",
                        f"{e2e_cpu['latency_ms_mean']:.2f}",
                        f"{e2e_cpu['throughput_sps']:.1f}",
                    ]
                )

            if (
                "perf_e2e_gpu" in results
                and "note" not in results["perf_e2e_gpu"]
            ):
                e2e_gpu = results["perf_e2e_gpu"]
                e2e_tbl.add_row(
                    [
                        gpu_model_name,
                        f"{e2e_gpu['latency_ms_p50']:.2f}",
                        f"{e2e_gpu['latency_ms_p90']:.2f}",
                        f"{e2e_gpu['latency_ms_mean']:.2f}",
                        f"{e2e_gpu['throughput_sps']:.1f}",
                    ]
                )
            md_lines.append("\n" + e2e_tbl.render())

        # Notes about skipped tests
        notes = []
        if "perf_gpu" in results and "note" in results["perf_gpu"]:
            notes.append("- GPU inference: " + results["perf_gpu"]["note"])
        if "perf_e2e_gpu" in results and "note" in results["perf_e2e_gpu"]:
            notes.append("- GPU end-to-end: " + results["perf_e2e_gpu"]["note"])
        if notes:
            md_lines.append("\n### Notes")
            md_lines.extend(notes)

    # Output Files
    md_lines.append("\n## Output Files")
    md_lines.append(f"\n- **Detailed CSV:** `{results.get('csv_path', 'N/A')}`")
    md_lines.append(f"- **ROC/PR Curves:** `{results.get('plot_path', 'N/A')}`")

    return "\n".join(md_lines)


def generate_comparison_report(
    results_video: Dict, results_audio: Dict, run_description: str
) -> str:
    """Generate a comparison markdown report between video+audio and audio-only modes."""
    md_lines = []

    md_lines.append("# Multimodal vs Audio-Only Comparison Report")
    md_lines.append(f"\n**Model:** `{results_video['onnx_path']}`")
    md_lines.append(f"\n**Run Description:** {run_description}")
    md_lines.append(
        f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    )

    # Dataset info
    md_lines.append("\n## Dataset Summary")
    md_lines.append(
        f"\n- **Total Samples:** {results_video['total_samples']:,}"
    )
    md_lines.append(
        f"- **Samples with Video Available:** {results_video['samples_with_video']:,} ({results_video['video_percentage']:.1f}%)"
    )

    # Performance Comparison Table
    md_lines.append("\n## Performance Comparison")

    metrics_v = results_video["metrics"]
    metrics_a = results_audio["metrics"]

    # Calculate deltas
    def delta_str(v_val, a_val, higher_is_better=True, is_percent=False):
        diff = v_val - a_val
        if abs(diff) < 0.001:
            return "—"
        sign = "+" if diff > 0 else ""
        color = ""
        if higher_is_better:
            color = "🟢" if diff > 0 else "🔴"
        else:
            color = "🔴" if diff > 0 else "🟢"
        if is_percent:
            return f"{color} {sign}{diff:.2f}%"
        return f"{color} {sign}{diff:.3f}"

    comp_tbl = MarkdownTable(
        headers=["Metric", "Video+Audio", "Audio-Only", "Δ (Video Impact)"],
        align=["left", "right", "right", "right"],
    )

    comp_tbl.add_row(
        [
            "Accuracy",
            f"{metrics_v['accuracy']:.2f}%",
            f"{metrics_a['accuracy']:.2f}%",
            delta_str(metrics_v["accuracy"], metrics_a["accuracy"], True, True),
        ]
    )
    comp_tbl.add_row(
        [
            "Precision",
            f"{metrics_v['precision']:.3f}",
            f"{metrics_a['precision']:.3f}",
            delta_str(metrics_v["precision"], metrics_a["precision"], True),
        ]
    )
    comp_tbl.add_row(
        [
            "Recall",
            f"{metrics_v['recall']:.3f}",
            f"{metrics_a['recall']:.3f}",
            delta_str(metrics_v["recall"], metrics_a["recall"], True),
        ]
    )
    comp_tbl.add_row(
        [
            "F1 Score",
            f"{metrics_v['f1']:.3f}",
            f"{metrics_a['f1']:.3f}",
            delta_str(metrics_v["f1"], metrics_a["f1"], True),
        ]
    )
    comp_tbl.add_row(
        [
            "False Positive Rate",
            f"{metrics_v['false_positive_rate']:.2f}%",
            f"{metrics_a['false_positive_rate']:.2f}%",
            delta_str(
                metrics_v["false_positive_rate"],
                metrics_a["false_positive_rate"],
                False,
                True,
            ),
        ]
    )
    comp_tbl.add_row(
        [
            "False Negative Rate",
            f"{metrics_v['false_negative_rate']:.2f}%",
            f"{metrics_a['false_negative_rate']:.2f}%",
            delta_str(
                metrics_v["false_negative_rate"],
                metrics_a["false_negative_rate"],
                False,
                True,
            ),
        ]
    )
    comp_tbl.add_row(
        [
            "ROC AUC",
            f"{results_video.get('roc_auc', 0):.4f}",
            f"{results_audio.get('roc_auc', 0):.4f}",
            delta_str(
                results_video.get("roc_auc", 0),
                results_audio.get("roc_auc", 0),
                True,
            ),
        ]
    )
    comp_tbl.add_row(
        [
            "PR AUC",
            f"{results_video.get('pr_auc', 0):.4f}",
            f"{results_audio.get('pr_auc', 0):.4f}",
            delta_str(
                results_video.get("pr_auc", 0),
                results_audio.get("pr_auc", 0),
                True,
            ),
        ]
    )

    md_lines.append("\n" + comp_tbl.render())

    # Threshold Analysis
    md_lines.append("\n## Threshold Analysis")
    thresh_tbl = MarkdownTable(
        headers=["Mode", "Best Threshold", "Best F1"],
        align=["left", "right", "right"],
    )
    thresh_tbl.add_row(
        [
            "Video+Audio",
            f"{results_video['best_threshold']:.2f}",
            f"{results_video['best_f1']:.4f}",
        ]
    )
    thresh_tbl.add_row(
        [
            "Audio-Only",
            f"{results_audio['best_threshold']:.2f}",
            f"{results_audio['best_f1']:.4f}",
        ]
    )
    md_lines.append("\n" + thresh_tbl.render())

    # Probability Distribution Comparison
    md_lines.append("\n## Probability Distribution")
    prob_tbl = MarkdownTable(
        headers=["Statistic", "Video+Audio", "Audio-Only"],
        align=["left", "right", "right"],
    )
    prob_tbl.add_row(
        [
            "Min",
            f"{results_video['prob_min']:.4f}",
            f"{results_audio['prob_min']:.4f}",
        ]
    )
    prob_tbl.add_row(
        [
            "Max",
            f"{results_video['prob_max']:.4f}",
            f"{results_audio['prob_max']:.4f}",
        ]
    )
    prob_tbl.add_row(
        [
            "Mean",
            f"{results_video['prob_mean']:.4f}",
            f"{results_audio['prob_mean']:.4f}",
        ]
    )
    prob_tbl.add_row(
        [
            "Std",
            f"{results_video['prob_std']:.4f}",
            f"{results_audio['prob_std']:.4f}",
        ]
    )
    md_lines.append("\n" + prob_tbl.render())

    # Summary
    md_lines.append("\n## Summary")
    acc_diff = metrics_v["accuracy"] - metrics_a["accuracy"]
    f1_diff = metrics_v["f1"] - metrics_a["f1"]

    if acc_diff > 0.5:
        md_lines.append(
            f"\n✅ **Video contributes positively:** +{acc_diff:.2f}% accuracy, +{f1_diff:.3f} F1"
        )
    elif acc_diff < -0.5:
        md_lines.append(
            f"\n⚠️ **Video may be hurting performance:** {acc_diff:.2f}% accuracy, {f1_diff:.3f} F1"
        )
    else:
        md_lines.append(
            f"\n➖ **Video has minimal impact:** {acc_diff:+.2f}% accuracy, {f1_diff:+.3f} F1"
        )

    # Output files
    md_lines.append("\n## Output Files")
    md_lines.append("\n### Video+Audio")
    md_lines.append(f"- CSV: `{results_video.get('csv_path', 'N/A')}`")
    md_lines.append(f"- Plot: `{results_video.get('plot_path', 'N/A')}`")
    md_lines.append("\n### Audio-Only")
    md_lines.append(f"- CSV: `{results_audio.get('csv_path', 'N/A')}`")
    md_lines.append(f"- Plot: `{results_audio.get('plot_path', 'N/A')}`")

    report = "\n".join(md_lines)

    # Save the comparison report
    model_name = None
    onnx_path = results_video["onnx_path"]
    if onnx_path.startswith("/data/output/"):
        path_parts = onnx_path.split("/")
        if len(path_parts) >= 4:
            model_name = path_parts[3]
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/data/benchmark/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = (
        f"{output_dir}/{run_description}_comparison_{timestamp}.md"
    )

    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(report)
    log_progress(f"📊 Comparison report saved to: {comparison_path}")

    return report


def load_robust_dataset(dataset_path: str):
    metadata_path = os.path.join(dataset_path, "metadata.jsonl")
    if os.path.exists(dataset_path) and os.path.exists(metadata_path):
        log_progress(
            "Detected RAW local dataset. Loading from metadata.jsonl..."
        )
        ds = load_dataset("json", data_files=metadata_path, split="train")

        def fix_paths(example):
            # --- AUDIO PATH ---
            # Try audio_path first, then fall back to id/file_name (which often contains audio path)
            audio_rel_path = None
            if "audio_path" in example and example["audio_path"]:
                audio_rel_path = example["audio_path"]
            elif (
                "id" in example
                and example["id"]
                and str(example["id"]).endswith(".wav")
            ):
                audio_rel_path = example["id"]
            elif (
                "file_name" in example
                and example["file_name"]
                and str(example["file_name"]).endswith(".wav")
            ):
                audio_rel_path = example["file_name"]

            if audio_rel_path:
                if not audio_rel_path.startswith("/"):
                    example["audio"] = os.path.join(
                        dataset_path, audio_rel_path
                    )
                else:
                    example["audio"] = audio_rel_path
                example["audio_path"] = audio_rel_path  # Store for logging

            # --- VIDEO PATH ---
            if "video_path" in example and example["video_path"]:
                if not example["video_path"].startswith("/"):
                    example["video_path"] = os.path.join(
                        dataset_path, example["video_path"]
                    )
            return example

        ds = ds.map(fix_paths)
        # Filter out entries where endpoint_bool is null
        ds = ds.filter(lambda example: example.get("endpoint_bool") is not None)
        log_progress(
            f"Filtered dataset to {len(ds)} samples with valid endpoint_bool"
        )
        return ds

    if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
        log_progress("Detected PRE-PROCESSED local dataset (Arrow).")
        return load_from_disk(dataset_path)

    log_progress("Attempting to load from Hugging Face Hub...")
    ds = load_dataset(dataset_path)
    if "test" in ds:
        return ds["test"]
    if "validation" in ds:
        return ds["validation"]
    if "train" in ds:
        return ds["train"]
    return ds


def run_benchmark_remote(
    onnx_path: str,
    dataset_path: str,
    run_description: str,
    limit: Optional[int],
    batch_size: int,
    perf_runs: int = PERF_RUNS_DEFAULT,
    skip_perf: bool = False,
    audio_only: bool = False,
):
    mode_str = "AUDIO-ONLY" if audio_only else "VIDEO+AUDIO"
    log_progress("=" * 80)
    log_progress(f"MULTIMODAL BENCHMARK ({mode_str})")
    log_progress("=" * 80)
    log_progress(f"Model: {onnx_path}")
    log_progress(f"Dataset: {dataset_path}")
    log_progress(f"Run Description: {run_description}")
    log_progress(f"Mode: {mode_str}")
    log_progress(f"Limit: {limit if limit else 'None'}")
    log_progress(f"Performance Runs: {perf_runs}")
    log_progress(f"Skip Performance Tests: {skip_perf}")

    # Generate output paths
    csv_path = generate_output_path(onnx_path, run_description, "csv")
    md_path = generate_output_path(onnx_path, run_description, "md")
    plot_path = generate_output_path(onnx_path, run_description, "png")

    log_progress(f"Loading dataset: {dataset_path}")
    try:
        raw_dataset = load_robust_dataset(dataset_path)
        if limit:
            raw_dataset = raw_dataset.select(
                range(min(limit, len(raw_dataset)))
            )
    except Exception as e:
        log_progress(f"Error loading dataset: {e}", "ERROR")
        return {}

    log_progress(f"Dataset loaded: {len(raw_dataset)} samples")
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    test_ds = MultimodalBenchmarkDataset(raw_dataset, fe, audio_only=audio_only)
    if audio_only:
        log_progress("⚠️ Audio-only mode: All video inputs will be zero tensors")

    # --- STABILITY FIX: num_workers=0 to prevent OOM kills ---
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=MultimodalCollator(),
        num_workers=0,
    )

    log_progress(f"Loading ONNX: {onnx_path}")
    gpu_model_name = get_gpu_model_name()

    # Check available providers
    available_providers = ort.get_available_providers()
    log_progress(f"Available providers: {available_providers}")

    cpu_providers = ["CPUExecutionProvider"]
    gpu_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers
        else None
    )

    # Build sessions
    cpu_session = build_session(onnx_path, cpu_providers)
    gpu_session = (
        build_session(onnx_path, gpu_providers) if gpu_providers else None
    )

    # Use GPU session for accuracy if available, else CPU
    session = gpu_session if gpu_session else cpu_session
    if gpu_session:
        log_progress(f"✅ CUDA Provider Active ({gpu_model_name})")
    else:
        log_progress("⚠️ CUDA not available. Using CPU for inference.")

    output_name = session.get_outputs()[0].name

    # --- COLLECT ALL DATA ---
    all_probs = []
    all_labels = []
    all_ids = []
    all_video_paths = []
    all_audio_paths = []
    all_has_video = []
    all_audio_durations = []

    log_progress("Starting Inference...")
    total_batches = len(loader)
    t0 = time.time()

    for i, batch in enumerate(loader):
        inputs = {
            "input_features": batch["input_features"],
            "pixel_values": batch["pixel_values"],
        }
        out = session.run([output_name], inputs)[0]
        probs = out.squeeze()
        if probs.ndim == 0:
            probs = np.array([probs])

        all_probs.extend(probs.tolist())
        all_labels.extend(batch["labels"].tolist())
        all_ids.extend(batch["ids"])
        all_video_paths.extend(batch["video_paths"])
        all_audio_paths.extend(batch["audio_paths"])
        all_has_video.extend(batch["has_video"])
        all_audio_durations.extend(batch["audio_durations"])

        if (i + 1) % max(1, total_batches // 20) == 0 or i + 1 == total_batches:
            elapsed = time.time() - t0
            samples_done = (i + 1) * batch_size
            rate = samples_done / elapsed if elapsed > 0 else 0
            log_progress(
                f"  Batch {i + 1}/{total_batches} | {samples_done:,} samples | {rate:.1f} samples/sec"
            )

    inference_time = time.time() - t0
    log_progress(f"Inference complete in {inference_time:.2f}s")

    # --- CONVERT TO NUMPY ---
    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    has_video_np = np.array(all_has_video)

    # --- COMPUTE METRICS ---
    log_progress("Computing metrics...")

    # Overall metrics
    overall_metrics = compute_metrics_with_confusion(probs_np, labels_np)

    # Metrics by modality
    video_mask = has_video_np
    metrics_with_video = (
        compute_metrics_with_confusion(
            probs_np[video_mask], labels_np[video_mask]
        )
        if video_mask.sum() > 0
        else {"sample_count": 0}
    )
    metrics_no_video = (
        compute_metrics_with_confusion(
            probs_np[~video_mask], labels_np[~video_mask]
        )
        if (~video_mask).sum() > 0
        else {"sample_count": 0}
    )

    # ROC/PR AUC
    roc_auc = 0.0
    pr_auc = 0.0
    try:
        fpr, tpr, _ = roc_curve(labels_np, probs_np)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(
            labels_np, probs_np
        )
        pr_auc = average_precision_score(labels_np, probs_np)

        # Generate plots
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC (AUC = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(
            recall_curve,
            precision_curve,
            color="blue",
            lw=2,
            label=f"PR (AUC = {pr_auc:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        log_progress(f"📈 Curves saved to: {plot_path}")
        plt.close()
    except Exception as e:
        log_progress(f"Error generating plots: {e}", "WARN")

    # Best threshold search
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.1, 0.9, 17):
        p_t = (probs_np > thresh).astype(int)
        f1_t = f1_score(labels_np, p_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh

    # --- SAVE DETAILED CSV ---
    log_progress("Saving detailed per-file results...")
    preds_05 = (probs_np > 0.5).astype(int)
    preds_best = (probs_np > best_thresh).astype(int)
    correct_05 = (preds_05 == labels_np).astype(int)
    correct_best = (preds_best == labels_np).astype(int)

    df = pd.DataFrame(
        {
            "id": all_ids,
            "true_label": all_labels,
            "prob": all_probs,
            "pred_0.5": preds_05,
            "correct_0.5": correct_05,
            f"pred_{best_thresh:.2f}": preds_best,
            f"correct_{best_thresh:.2f}": correct_best,
            "has_video": all_has_video,
            "video_path": all_video_paths,
            "audio_path": all_audio_paths,
            "audio_duration_sec": all_audio_durations,
        }
    )
    df.to_csv(csv_path, index=False)
    log_progress(f"💾 Detailed results saved to: {csv_path}")

    # --- PERFORMANCE BENCHMARKS ---
    perf_results = {}

    if not skip_perf:
        log_progress("=" * 50)
        log_progress("PERFORMANCE BENCHMARKS")
        log_progress("=" * 50)

        # 1. Direct inference performance (CPU)
        log_progress("Testing direct inference performance (CPU)...")
        perf_results["perf_cpu"] = run_perf_multimodal(
            cpu_session, runs=perf_runs, warmup=PERF_WARMUP_DEFAULT
        )

        # 2. Direct inference performance (GPU)
        if gpu_session:
            log_progress("Testing direct inference performance (GPU)...")
            perf_results["perf_gpu"] = run_perf_multimodal(
                gpu_session, runs=perf_runs, warmup=PERF_WARMUP_DEFAULT
            )
        else:
            perf_results["perf_gpu"] = {"note": "CUDA not available; skipped."}

        # 3. Audio feature extraction performance
        log_progress("Testing audio feature extraction performance...")
        perf_results["perf_audio_fe"] = run_audio_fe_perf(
            fe, runs=perf_runs, warmup=PERF_WARMUP_DEFAULT
        )

        # 4. Video preprocessing performance (use first video if available)
        sample_video_path = None
        for vp in all_video_paths:
            if vp and os.path.exists(str(vp)):
                sample_video_path = vp
                break

        if sample_video_path:
            log_progress(
                f"Testing video preprocessing performance (using {os.path.basename(sample_video_path)})..."
            )
            perf_results["perf_video_preprocess"] = (
                run_video_preprocessing_perf(
                    video_path=sample_video_path,
                    runs=min(perf_runs, 50),
                    warmup=5,
                )
            )
        else:
            log_progress("No video files found for preprocessing benchmark")

        # 5. End-to-end performance (CPU)
        log_progress("Testing end-to-end performance (CPU, audio-only)...")
        perf_results["perf_e2e_cpu"] = run_e2e_perf_multimodal(
            cpu_session,
            fe,
            video_path=None,
            runs=perf_runs,
            warmup=PERF_WARMUP_DEFAULT,
        )

        # 6. End-to-end performance (GPU)
        if gpu_session:
            log_progress("Testing end-to-end performance (GPU, audio-only)...")
            perf_results["perf_e2e_gpu"] = run_e2e_perf_multimodal(
                gpu_session,
                fe,
                video_path=None,
                runs=perf_runs,
                warmup=PERF_WARMUP_DEFAULT,
            )
        else:
            perf_results["perf_e2e_gpu"] = {
                "note": "CUDA not available; skipped."
            }

        log_progress("Performance benchmarks complete!")
    else:
        log_progress("Skipping performance benchmarks (skip_perf=True)")

    # --- BUILD RESULTS DICT ---
    samples_with_video = int(has_video_np.sum())
    results = {
        "onnx_path": onnx_path,
        "dataset_path": dataset_path,
        "run_description": run_description,
        "audio_only": audio_only,
        "total_samples": len(labels_np),
        "samples_with_video": samples_with_video,
        "video_percentage": (samples_with_video / len(labels_np) * 100)
        if len(labels_np) > 0
        else 0,
        "inference_time_sec": inference_time,
        "throughput_sps": len(labels_np) / inference_time
        if inference_time > 0
        else 0,
        "metrics": overall_metrics,
        "metrics_with_video": metrics_with_video,
        "metrics_no_video": metrics_no_video,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "prob_min": float(probs_np.min()),
        "prob_max": float(probs_np.max()),
        "prob_mean": float(probs_np.mean()),
        "prob_std": float(probs_np.std()),
        "csv_path": csv_path,
        "plot_path": plot_path,
        **perf_results,  # Include all performance results
    }

    # --- GENERATE MARKDOWN REPORT ---
    log_progress("Generating markdown report...")
    md_report = format_markdown_report(results, gpu_model_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    log_progress(f"📄 Markdown report saved to: {md_path}")

    # --- PRINT SUMMARY ---
    print("\n" + "=" * 60)
    print("MULTIMODAL BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Samples:      {results['total_samples']:,}")
    print(
        f"Samples with Video: {results['samples_with_video']:,} ({results['video_percentage']:.1f}%)"
    )
    print(f"Inference Time:     {results['inference_time_sec']:.2f}s")
    print(f"Throughput:         {results['throughput_sps']:.1f} samples/sec")
    print(f"\nOverall Accuracy:   {overall_metrics['accuracy']:.2f}%")
    print(f"Overall F1:         {overall_metrics['f1']:.3f}")
    print(f"ROC AUC:            {roc_auc:.4f}")
    print(f"PR AUC:             {pr_auc:.4f}")
    print(f"\nBest Threshold:     {best_thresh:.2f} (F1: {best_f1:.4f})")
    print(
        f"\n[Probs] Min: {probs_np.min():.4f} | Max: {probs_np.max():.4f} | Mean: {probs_np.mean():.4f}"
    )

    # Print performance summary if available
    if perf_results:
        print("\n--- PERFORMANCE ---")
        if (
            "perf_cpu" in perf_results
            and "note" not in perf_results["perf_cpu"]
        ):
            print(
                f"CPU Inference:      {perf_results['perf_cpu']['latency_ms_mean']:.2f}ms mean"
            )
        if (
            "perf_gpu" in perf_results
            and "note" not in perf_results["perf_gpu"]
        ):
            print(
                f"GPU Inference:      {perf_results['perf_gpu']['latency_ms_mean']:.2f}ms mean"
            )
        if "perf_audio_fe" in perf_results:
            print(
                f"Audio FE:           {perf_results['perf_audio_fe']['latency_ms_mean']:.2f}ms mean"
            )
        if "perf_video_preprocess" in perf_results:
            print(
                f"Video Preprocess:   {perf_results['perf_video_preprocess']['latency_ms_mean']:.2f}ms mean"
            )
        if (
            "perf_e2e_cpu" in perf_results
            and "note" not in perf_results["perf_e2e_cpu"]
        ):
            print(
                f"E2E CPU:            {perf_results['perf_e2e_cpu']['latency_ms_mean']:.2f}ms mean"
            )
        if (
            "perf_e2e_gpu" in perf_results
            and "note" not in perf_results["perf_e2e_gpu"]
        ):
            print(
                f"E2E GPU:            {perf_results['perf_e2e_gpu']['latency_ms_mean']:.2f}ms mean"
            )

    print("=" * 60 + "\n")

    return results


# 24GB Memory + T4 GPU
@app.function(
    image=image, gpu="T4", volumes={"/data": volume}, timeout=3600, memory=24576
)
def benchmark_entrypoint(
    onnx_path: str,
    dataset_path: str,
    run_description: str = "benchmark",
    limit: Optional[int] = None,
    batch_size: int = 8,
    perf_runs: int = PERF_RUNS_DEFAULT,
    skip_perf: bool = False,
    audio_only: bool = False,
):
    """Remote benchmark entrypoint running on Modal GPU."""
    return run_benchmark_remote(
        onnx_path=onnx_path,
        dataset_path=dataset_path,
        run_description=run_description,
        limit=limit,
        batch_size=batch_size,
        perf_runs=perf_runs,
        skip_perf=skip_perf,
        audio_only=audio_only,
    )


@app.local_entrypoint()
def main(
    onnx_path: str,
    dataset_path: str,
    run_description: str = "benchmark",
    limit: Optional[int] = None,
    batch_size: int = 8,
    perf_runs: int = PERF_RUNS_DEFAULT,
    skip_perf: bool = False,
    audio_only: bool = False,
    compare_modalities: bool = False,
):
    """
    Run multimodal endpointing benchmark.

    Args:
        onnx_path: Path to ONNX model (e.g., /data/output/mm_run_20260111_1904/model_fixed.onnx)
        dataset_path: Path to dataset (e.g., /data/datasets/casual_conversations)
        run_description: Description for this benchmark run (used in output filenames)
        limit: Optional limit on number of samples to evaluate
        batch_size: Batch size for inference (default 8, safer for video decoding)
        perf_runs: Number of runs for performance benchmarks (default 100)
        skip_perf: Skip performance benchmarks entirely (default False)
        audio_only: Run benchmark with zero video tensors (audio-only mode)
        compare_modalities: Run both video+audio and audio-only, then generate comparison report
    """
    log_progress(f"Starting multimodal benchmark: {run_description}")
    log_progress(f"Model: {onnx_path}")
    log_progress(f"Dataset: {dataset_path}")
    log_progress(f"Performance runs: {perf_runs}, Skip perf: {skip_perf}")

    if compare_modalities:
        log_progress("=" * 60)
        log_progress("COMPARISON MODE: Running both Video+Audio and Audio-Only")
        log_progress("=" * 60)

        # Run with video+audio
        log_progress("\n[1/2] Running Video+Audio benchmark...")
        results_video = benchmark_entrypoint.remote(
            onnx_path=onnx_path,
            dataset_path=dataset_path,
            run_description=f"{run_description}_video_audio",
            limit=limit,
            batch_size=batch_size,
            perf_runs=perf_runs,
            skip_perf=skip_perf,
            audio_only=False,
        )

        # Run audio-only
        log_progress("\n[2/2] Running Audio-Only benchmark...")
        results_audio = benchmark_entrypoint.remote(
            onnx_path=onnx_path,
            dataset_path=dataset_path,
            run_description=f"{run_description}_audio_only",
            limit=limit,
            batch_size=batch_size,
            perf_runs=perf_runs,
            skip_perf=skip_perf,
            audio_only=True,
        )

        # Generate comparison report
        if results_video and results_audio:
            comparison_report = generate_comparison_report(
                results_video, results_audio, run_description
            )
            print(comparison_report)

        print("\n" + "=" * 60)
        print("COMPARISON BENCHMARK COMPLETE")
        print("=" * 60)
        if results_video:
            print(
                f"Video+Audio results: {results_video.get('csv_path', 'N/A')}"
            )
        if results_audio:
            print(f"Audio-Only results: {results_audio.get('csv_path', 'N/A')}")
        print("=" * 60)
    else:
        mode_str = "audio-only" if audio_only else "video+audio"
        log_progress(f"Mode: {mode_str}")

        results = benchmark_entrypoint.remote(
            onnx_path=onnx_path,
            dataset_path=dataset_path,
            run_description=run_description,
            limit=limit,
            batch_size=batch_size,
            perf_runs=perf_runs,
            skip_perf=skip_perf,
            audio_only=audio_only,
        )

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        if results:
            print(f"Results saved to: {results.get('csv_path', 'N/A')}")
        print("=" * 60)
