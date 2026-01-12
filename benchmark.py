import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Union, Optional

import matplotlib

# Force headless backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import modal
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
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
from huggingface_hub import hf_hub_download
import librosa

app = modal.App("endpointing-benchmark")

# Default HuggingFace audio-only model for comparison
HF_AUDIO_ONLY_MODEL = "onnx-community/smart-turn-v3-ONNX"
HF_AUDIO_ONLY_FILE = "onnx/model.onnx"
volume = modal.Volume.from_name("endpointing", create_if_missing=False)


def resolve_onnx_path(onnx_path: str) -> str:
    """
    Resolve ONNX model path - supports local paths, 'hf:' prefix for HuggingFace,
    or 'hf-audio-only' shortcut for the default audio-only model.
    """
    if onnx_path == "hf-audio-only":
        # Shortcut for the default audio-only smart-turn model
        log_progress(
            f"Downloading audio-only model from HuggingFace: {HF_AUDIO_ONLY_MODEL}"
        )
        local_path = hf_hub_download(
            repo_id=HF_AUDIO_ONLY_MODEL, filename=HF_AUDIO_ONLY_FILE
        )
        log_progress(f"Model downloaded to: {local_path}")
        return local_path

    if onnx_path.startswith("hf:"):
        # Format: hf:repo_id/filename or hf:repo_id (uses default onnx/model.onnx)
        hf_ref = onnx_path[3:]  # Remove "hf:" prefix
        if "/" in hf_ref and hf_ref.count("/") >= 2:
            # Has filename: hf:org/repo/path/to/model.onnx
            parts = hf_ref.split("/")
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
        else:
            # Just repo: hf:org/repo
            repo_id = hf_ref
            filename = "onnx/model.onnx"

        log_progress(
            f"Downloading model from HuggingFace: {repo_id}/{filename}"
        )
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        log_progress(f"Model downloaded to: {local_path}")
        return local_path

    # Regular local/volume path
    return onnx_path


image = modal.Image.debian_slim().pip_install(
    "numpy==2.3.4",
    "torch==2.9.0",
    "datasets==4.4.1",
    "transformers[torch]==4.48.2",
    "scikit-learn==1.6.1",
    "onnxruntime-gpu==1.23.0",
    "librosa",
    "soundfile",
    "huggingface_hub",
    "matplotlib",
    "pandas",
)

SAMPLING_RATE = 16000
N_MELS = 80
N_FRAMES = 800  # 8s → 800 frames
FEATURE_SHAPE = (1, N_MELS, N_FRAMES)
AUDIO_SECONDS = 8

# Language code to full name mapping with flag emojis
LANGUAGE_MAPPING = {
    "eng": "🇬🇧 🇺🇸 English",
    "rus": "🇷🇺 Russian",
    "por": "🇵🇹 Portuguese",
    "nld": "🇳🇱 Dutch",
    "deu": "🇩🇪 German",
    "hin": "🇮🇳 Hindi",
    "spa": "🇪🇸 Spanish",
    "fra": "🇫🇷 French",
    "vie": "🇻🇳 Vietnamese",
    "ind": "🇮🇩 Indonesian",
    "nor": "🇳🇴 Norwegian",
    "fin": "🇫🇮 Finnish",
    "ben": "🇧🇩 Bengali",
    "pol": "🇵🇱 Polish",
    "ara": "🇸🇦 Arabic",
    "tur": "🇹🇷 Turkish",
    "zho": "🇨🇳 Chinese",
    "ukr": "🇺🇦 Ukrainian",
    "kor": "🇰🇷 Korean",
    "jpn": "🇯🇵 Japanese",
    "dan": "🇩🇰 Danish",
    "ita": "🇮🇹 Italian",
    "mar": "🇮🇳 Marathi",
}


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
            gpu_name = result.stdout.strip().split("\n")[0]  # Get first GPU
            # Extract just the model name (e.g., "NVIDIA L4" -> "L4")
            if " " in gpu_name:
                model_name = gpu_name.split()[-1]  # Get last part
                return model_name
            return gpu_name
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    # Fallback: try to get from CUDA device properties
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if " " in device_name:
                model_name = device_name.split()[-1]
                return model_name
            return device_name
    except Exception:
        pass

    # Final fallback
    return "GPU"


def generate_output_path(
    onnx_path: str, run_description: Optional[str], extension: str = "md"
) -> str:
    """Generate dynamic output path based on ONNX path and current datetime."""
    # Try to extract model name from /data/output/{model_name}/... format
    model_name = None
    if onnx_path.startswith("/data/output/"):
        # Split the path and get the part after /data/output/
        path_parts = onnx_path.split("/")
        if len(path_parts) >= 4:  # /data/output/{model_name}/...
            model_name = path_parts[3]

    # Fallback to filename if path doesn't match expected format
    if model_name is None:
        onnx_filename = os.path.basename(onnx_path)
        model_name = os.path.splitext(onnx_filename)[0]

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_description is None:
        run_description = "report"

    # Create output path
    output_path = f"/data/benchmark/{model_name}/{run_description}_{timestamp}.{extension}"

    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    return output_path


def load_dataset_at(path: str):
    log_progress(f"Loading dataset from: {path}")
    if path.startswith("/"):
        dataset = load_from_disk(path)["train"]
    else:
        dataset = load_dataset(path)["train"]
    log_progress(f"Dataset loaded successfully. Size: {len(dataset):,} samples")
    return dataset


def load_robust_dataset(dataset_path: str):
    """Load dataset from various formats including casual_conversations with metadata.jsonl."""
    metadata_path = os.path.join(dataset_path, "metadata.jsonl")
    if os.path.exists(dataset_path) and os.path.exists(metadata_path):
        log_progress(
            "Detected RAW local dataset. Loading from metadata.jsonl..."
        )
        ds = load_dataset("json", data_files=metadata_path, split="train")

        def fix_paths(example):
            # --- AUDIO PATH ---
            audio_rel_path = None
            if "audio_path" in example and example["audio_path"]:
                audio_rel_path = example["audio_path"]
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
                example["audio_path"] = audio_rel_path
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


class AudioOnlyBenchmarkDataset(Dataset):
    """Dataset for benchmarking audio-only models on casual_conversations format."""

    def __init__(self, hf_dataset, feature_extractor):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load audio from various sources
        audio_array = None
        if (
            "audio" in item
            and isinstance(item["audio"], dict)
            and "array" in item["audio"]
        ):
            # HuggingFace datasets format
            audio_array = item["audio"]["array"]
        elif "audio" in item and isinstance(item["audio"], str):
            # File path format (casual_conversations)
            try:
                audio_array, _ = librosa.load(
                    item["audio"], sr=SAMPLING_RATE, mono=True
                )
            except Exception as e:
                if idx < 5:
                    log_progress(
                        f"[WARN] Failed to load audio for sample {idx}: {e}"
                    )

        if audio_array is None:
            audio_array = np.zeros(
                SAMPLING_RATE * AUDIO_SECONDS, dtype=np.float32
            )

        # Truncate to last 8 seconds
        max_samples = SAMPLING_RATE * AUDIO_SECONDS
        if len(audio_array) > max_samples:
            audio_array = audio_array[-max_samples:]

        # Get label
        label = item.get("endpoint_bool")
        if label is None:
            label = item.get("label", 0)
        label = 1 if label else 0

        # Extract features
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=SAMPLING_RATE * AUDIO_SECONDS,
            truncation=True,
            do_normalize=True,
        )

        # Get sample ID
        file_id = item.get("file_name") or item.get("id") or str(idx)
        audio_path = item.get("audio_path") or item.get("audio", "")
        if isinstance(audio_path, dict):
            audio_path = audio_path.get("path", "")

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "language": item.get("language", "eng"),
            "dataset": item.get("dataset", "unknown"),
            "id": file_id,
            "audio_path": str(audio_path),
        }


@dataclass
class WhisperDataCollator:
    def __call__(
        self, features: List[Dict[str, Union[torch.Tensor, int, str]]]
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        input_features = torch.stack([f["input_features"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        languages = [f["language"] for f in features]
        datasets = [f["dataset"] for f in features]
        ids = [f["id"] for f in features]
        audio_paths = [f["audio_path"] for f in features]
        return {
            "input_features": input_features,
            "labels": labels,
            "languages": languages,
            "datasets": datasets,
            "ids": ids,
            "audio_paths": audio_paths,
        }


def process_predictions(logits: np.ndarray):
    probs = logits.squeeze()
    # Ensure probs is always at least 1D to avoid concatenation issues
    # squeeze() can make (1,) -> () (0D), which breaks concatenation
    if probs.ndim == 0:
        probs = probs.reshape(1)
    preds = (probs > 0.5).astype(int)
    return probs, preds


def compute_metrics_with_confusion(probs: np.ndarray, labels: np.ndarray):
    """Compute metrics including false positive and false negative rates."""
    preds = (probs > 0.5).astype(int)

    # Calculate confusion matrix components
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


def compute_per_category_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    categories: List[str],
    category_name: str,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each category (language or dataset) separately."""
    log_progress(f"Computing per-{category_name} metrics...")
    category_metrics = {}

    # Group by category
    cat_data = defaultdict(lambda: {"probs": [], "labels": []})
    for prob, label, cat in zip(probs, labels, categories):
        cat_data[cat]["probs"].append(prob)
        cat_data[cat]["labels"].append(label)

    # Compute metrics for each category
    unique_categories = list(cat_data.keys())
    log_progress(
        f"Found {len(unique_categories)} unique {category_name}s: {sorted(unique_categories)}"
    )

    for i, (cat, data) in enumerate(cat_data.items()):
        cat_probs = np.array(data["probs"])
        cat_labels = np.array(data["labels"])

        if len(cat_labels) > 0:  # Only compute if we have samples
            category_metrics[cat] = compute_metrics_with_confusion(
                cat_probs, cat_labels
            )
            log_progress(
                f"  [{i + 1}/{len(unique_categories)}] {cat}: {len(cat_labels)} samples, "
                f"accuracy: {category_metrics[cat]['accuracy']:.2f}%"
            )
        else:
            category_metrics[cat] = {
                "sample_count": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
            }
            log_progress(
                f"  [{i + 1}/{len(unique_categories)}] {cat}: 0 samples"
            )

    return category_metrics


def format_language_name(lang_code: str) -> str:
    """Convert language code to full name with flag emoji."""
    return LANGUAGE_MAPPING.get(lang_code, lang_code)


class MarkdownTable:
    """Render a padded/aligned Markdown table (monospace-friendly)."""

    def __init__(self, headers: List[str], align: Optional[List[str]] = None):
        self.headers = [str(h) for h in headers]
        self.rows: List[List[str]] = []
        self.align = align or ["left"] * len(self.headers)
        if len(self.align) != len(self.headers):
            raise ValueError("align must match headers length")

    def add_row(self, row: List[Any]) -> None:
        if len(row) != len(self.headers):
            raise ValueError("row must match headers length")
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


def format_markdown_report(results: Dict, gpu_model_name: str = "GPU") -> str:
    """Format the results into a comprehensive Markdown report."""
    md_lines = []

    # Header
    md_lines.append("# Endpointing Model Benchmark Report")
    md_lines.append(f"\n**Model:** `{results['onnx_path']}`")
    md_lines.append(
        f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    )

    # Accuracy Results
    if "accuracy" in results and "note" not in results["accuracy"]:
        acc_data = results["accuracy"]

        md_lines.append("\n## Accuracy Results")
        md_lines.append(f"\n**Total Samples:** {acc_data['total_samples']:,}")
        formatted_languages = [
            format_language_name(lang) for lang in acc_data["unique_languages"]
        ]
        md_lines.append(
            f"\n**Unique Languages:** {', '.join(formatted_languages)}"
        )
        if "unique_datasets" in acc_data:
            md_lines.append(
                f"\n**Unique Datasets:** {', '.join(acc_data['unique_datasets'])}"
            )

        # Overall Accuracy Table
        md_lines.append("\n### Overall Performance")
        overall = acc_data["overall"]
        overall_tbl = MarkdownTable(
            headers=[
                "Metric",
                "Sample Count",
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
        overall_tbl.add_row(
            [
                "Overall",
                f"{overall['sample_count']:,}",
                f"{overall['accuracy']:.2f}",
                f"{overall['precision']:.3f}",
                f"{overall['recall']:.3f}",
                f"{overall['f1']:.3f}",
                f"{overall['false_positive_rate']:.2f}",
                f"{overall['false_negative_rate']:.2f}",
            ]
        )
        md_lines.append("\n" + overall_tbl.render())

        # ROC/PR AUC if available
        if "roc_auc" in acc_data:
            md_lines.append(f"\n**ROC AUC:** {acc_data['roc_auc']:.4f}")
            md_lines.append(f"\n**PR AUC:** {acc_data['pr_auc']:.4f}")

        # Best threshold if available
        if "best_threshold" in acc_data:
            md_lines.append(
                f"\n**Best Threshold:** {acc_data['best_threshold']:.2f} "
                f"(F1: {acc_data['best_f1']:.4f})"
            )

        # Probability distribution if available
        if "prob_min" in acc_data:
            md_lines.append("\n### Probability Distribution")
            md_lines.append(f"\n- **Min:** {acc_data['prob_min']:.4f}")
            md_lines.append(f"- **Max:** {acc_data['prob_max']:.4f}")
            md_lines.append(f"- **Mean:** {acc_data['prob_mean']:.4f}")
            md_lines.append(f"- **Std:** {acc_data['prob_std']:.4f}")

        # Per-Language Accuracy Table
        if "per_language" in acc_data and acc_data["per_language"]:
            md_lines.append("\n### Performance by Language")
            lang_tbl = MarkdownTable(
                headers=[
                    "Language",
                    "Sample Count",
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

            # Sort languages by accuracy in descending order
            sorted_languages = sorted(
                acc_data["per_language"].keys(),
                key=lambda lang: acc_data["per_language"][lang]["accuracy"],
                reverse=True,
            )

            for lang in sorted_languages:
                metrics = acc_data["per_language"][lang]
                formatted_lang = format_language_name(lang)
                lang_tbl.add_row(
                    [
                        formatted_lang,
                        f"{metrics['sample_count']:,}",
                        f"{metrics['accuracy']:.2f}",
                        f"{metrics['precision']:.3f}",
                        f"{metrics['recall']:.3f}",
                        f"{metrics['f1']:.3f}",
                        f"{metrics['false_positive_rate']:.2f}",
                        f"{metrics['false_negative_rate']:.2f}",
                    ]
                )
            md_lines.append("\n" + lang_tbl.render())

        # Per-Dataset Accuracy Table
        if "per_dataset" in acc_data and acc_data["per_dataset"]:
            md_lines.append("\n### Performance by Dataset")
            ds_tbl = MarkdownTable(
                headers=[
                    "Dataset",
                    "Sample Count",
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

            # Sort datasets by accuracy in descending order
            sorted_datasets = sorted(
                acc_data["per_dataset"].keys(),
                key=lambda dataset: acc_data["per_dataset"][dataset][
                    "accuracy"
                ],
                reverse=True,
            )

            for dataset in sorted_datasets:
                metrics = acc_data["per_dataset"][dataset]
                ds_tbl.add_row(
                    [
                        dataset,
                        f"{metrics['sample_count']:,}",
                        f"{metrics['accuracy']:.2f}",
                        f"{metrics['precision']:.3f}",
                        f"{metrics['recall']:.3f}",
                        f"{metrics['f1']:.3f}",
                        f"{metrics['false_positive_rate']:.2f}",
                        f"{metrics['false_negative_rate']:.2f}",
                    ]
                )
            md_lines.append("\n" + ds_tbl.render())

    # Performance Results
    md_lines.append("\n## Inference Performance")

    # Direct Inference (pre-computed features)
    md_lines.append("\n### Direct Inference Performance")
    md_lines.append("*Using pre-computed zero features (inference only)*")
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

    if "perf_cpu" in results:
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
    if "perf_feature_extractor" in results:
        md_lines.append("\n### Feature Extraction Performance")
        md_lines.append("*Whisper feature extraction from 8-second audio*")
        fe_perf = results["perf_feature_extractor"]
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
                "Feature Extractor",
                f"{fe_perf['latency_ms_p50']:.2f}",
                f"{fe_perf['latency_ms_p90']:.2f}",
                f"{fe_perf['latency_ms_mean']:.2f}",
                f"{fe_perf['throughput_sps']:.1f}",
            ]
        )
        md_lines.append("\n" + fe_tbl.render())

    # End-to-End Performance
    md_lines.append("\n### End-to-End Performance")
    md_lines.append("*Feature extraction + inference from raw audio*")
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

    if "perf_e2e_cpu" in results:
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

    if "perf_e2e_gpu" in results and "note" not in results["perf_e2e_gpu"]:
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

    # Add notes about any skipped measurements
    notes = []
    if "perf_gpu" in results and "note" in results["perf_gpu"]:
        notes.append("- GPU inference: " + results["perf_gpu"]["note"])
    if "perf_e2e_gpu" in results and "note" in results["perf_e2e_gpu"]:
        notes.append("- GPU end-to-end: " + results["perf_e2e_gpu"]["note"])
    if "accuracy" in results and "note" in results["accuracy"]:
        notes.append("- Accuracy evaluation: " + results["accuracy"]["note"])

    if notes:
        md_lines.append("\n## Notes")
        md_lines.extend(notes)

    # Output files section
    if "accuracy" in results and "note" not in results["accuracy"]:
        acc_data = results["accuracy"]
        if acc_data.get("csv_path") or acc_data.get("plot_path"):
            md_lines.append("\n## Output Files")
            if acc_data.get("csv_path"):
                md_lines.append(
                    f"\n- **Detailed CSV:** `{acc_data['csv_path']}`"
                )
            if acc_data.get("plot_path"):
                md_lines.append(
                    f"\n- **ROC/PR Curves:** `{acc_data['plot_path']}`"
                )

    return "\n".join(md_lines)


def _zero_audio(
    n_seconds: int = AUDIO_SECONDS, sample_rate: int = SAMPLING_RATE
) -> np.ndarray:
    return np.zeros(n_seconds * sample_rate, dtype=np.float32)


def _extract_features_np(
    fe: WhisperFeatureExtractor,
    audio: np.ndarray,
) -> np.ndarray:
    """Return (1, 80, 800) np.float32 features from 8s audio."""
    out = fe(
        audio,
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",  # returns numpy arrays
        padding="max_length",
        max_length=AUDIO_SECONDS * SAMPLING_RATE,
        truncation=True,
        do_normalize=True,
        device="cuda",
    )["input_features"].astype(np.float32)
    # Ensure (1,80,800)
    if out.shape != FEATURE_SHAPE:
        out = out.reshape(FEATURE_SHAPE)
    return out


def _latency_stats(times: List[float]) -> Dict[str, float]:
    p50 = np.percentile(times, 50) * 1000
    p90 = np.percentile(times, 90) * 1000
    mean = np.mean(times) * 1000
    return {
        "latency_ms_p50": float(p50),
        "latency_ms_p90": float(p90),
        "latency_ms_mean": float(mean),
        "throughput_sps": float(1.0 / np.mean(times)),
    }


def run_fe_perf(
    fe: WhisperFeatureExtractor,
    audio: np.ndarray,
    runs: int = 1000,
    warmup: int = 100,
) -> Dict[str, float]:
    log_progress(
        f"Running feature extraction performance test ({warmup} warmup + {runs} runs)"
    )

    # warmup
    log_progress("  Warming up feature extractor...")
    for i in range(warmup):
        _ = _extract_features_np(fe, audio)
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

    log_progress("  Running timed feature extraction...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        _ = _extract_features_np(fe, audio)
        times.append(time.perf_counter() - t0)

        if (i + 1) % max(1, runs // 10) == 0:
            current_mean = np.mean(times) * 1000
            log_progress(
                f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms"
            )

    stats = _latency_stats(times)
    log_progress(
        f"  Feature extraction complete - Mean: {stats['latency_ms_mean']:.2f}ms, "
        f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms"
    )
    return stats


def run_e2e_perf(
    session: ort.InferenceSession,
    fe: WhisperFeatureExtractor,
    audio: np.ndarray,
    runs: int = 1000,
    warmup: int = 100,
) -> Dict[str, float]:
    provider_name = session.get_providers()[0]
    log_progress(
        f"Running end-to-end performance test on {provider_name} ({warmup} warmup + {runs} runs)"
    )

    inp_name = session.get_inputs()[0].name

    # warmup
    log_progress("  Warming up end-to-end pipeline...")
    for i in range(warmup):
        feats = _extract_features_np(fe, audio)  # (1,80,800)
        _ = session.run(None, {inp_name: feats})
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

    log_progress("  Running timed end-to-end inference...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        feats = _extract_features_np(fe, audio)
        _ = session.run(None, {inp_name: feats})
        times.append(time.perf_counter() - t0)

        if (i + 1) % max(1, runs // 10) == 0:
            current_mean = np.mean(times) * 1000
            log_progress(
                f"    Progress: {i + 1}/{runs} | Current mean latency: {current_mean:.2f}ms"
            )

    stats = _latency_stats(times)
    log_progress(
        f"  End-to-end {provider_name} complete - Mean: {stats['latency_ms_mean']:.2f}ms, "
        f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms"
    )
    return stats


def run_perf(session: ort.InferenceSession, runs: int = 100, warmup: int = 10):
    provider_name = session.get_providers()[0]
    log_progress(
        f"Running inference performance test on {provider_name} ({warmup} warmup + {runs} runs)"
    )

    inp = np.zeros(FEATURE_SHAPE, dtype=np.float32)
    feed = {session.get_inputs()[0].name: inp}

    # warmup
    log_progress("  Warming up inference session...")
    for i in range(warmup):
        session.run(None, feed)
        if (i + 1) % max(1, warmup // 4) == 0:
            log_progress(f"    Warmup progress: {i + 1}/{warmup}")

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
        f"  Inference {provider_name} complete - Mean: {stats['latency_ms_mean']:.2f}ms, "
        f"P50: {stats['latency_ms_p50']:.2f}ms, P90: {stats['latency_ms_p90']:.2f}ms"
    )
    return stats


def build_session(onnx_path: str, providers: List[str]):
    log_progress(f"Building ONNX session with providers: {providers}")
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_path, sess_options=so, providers=providers
    )
    log_progress(
        f"Session created successfully. Active provider: {session.get_providers()[0]}"
    )
    return session


def run_accuracy(
    onnx_path: str,
    dataset,
    limit: Optional[int],
    batch_size: int = 2,
    csv_path: Optional[str] = None,
    plot_path: Optional[str] = None,
):
    log_progress("=" * 50)
    log_progress("ACCURACY EVALUATION")

    if limit is not None:
        n = min(limit, len(dataset))
        indices = list(range(n))
        dataset = torch.utils.data.Subset(dataset, indices)
        log_progress(f"Limited dataset to {n:,} samples (limit: {limit:,})")
    else:
        log_progress(f"Using full dataset: {len(dataset):,} samples")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=WhisperDataCollator(),
    )
    total_batches = len(loader)
    log_progress(
        f"Created data loader: {total_batches} batches of size {batch_size}"
    )

    log_progress("Building inference session...")
    # Prefer CUDA if available, fallback to CPU
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        session_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        log_progress("Using CUDA provider for accuracy evaluation")
    else:
        session_providers = ["CPUExecutionProvider"]
        log_progress(
            "CUDA not available, using CPU provider for accuracy evaluation"
        )

    sess = build_session(onnx_path, providers=session_providers)
    inp_name = sess.get_inputs()[0].name

    log_progress("Starting inference on dataset...")
    probs_all = []
    labels_all = []
    languages_all = []
    datasets_all = []
    ids_all = []
    audio_paths_all = []

    samples_processed = 0
    batch_start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        batch_inference_start = time.time()

        x = batch["input_features"].numpy()  # (B,80,800)
        y = batch["labels"].numpy()
        langs = batch["languages"]
        dsets = batch["datasets"]
        ids = batch["ids"]
        audio_paths = batch["audio_paths"]

        # Run inference
        out = sess.run(None, {inp_name: x})
        probs, _ = process_predictions(out[0])

        probs_all.append(probs)
        labels_all.append(y)
        languages_all.extend(langs)
        datasets_all.extend(dsets)
        ids_all.extend(ids)
        audio_paths_all.extend(audio_paths)

        samples_processed += len(y)
        batch_inference_time = time.time() - batch_inference_start

        # Progress logging
        if (batch_idx + 1) % max(
            1, total_batches // 20
        ) == 0 or batch_idx + 1 == total_batches:
            elapsed = time.time() - batch_start_time
            samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
            eta_seconds = (
                (len(dataset) - samples_processed) / samples_per_sec
                if samples_per_sec > 0
                else 0
            )
            eta_str = (
                f"{int(eta_seconds // 60)}:{int(eta_seconds % 60):02d}"
                if eta_seconds < float("inf")
                else "N/A"
            )

            log_progress(
                f"  Batch {batch_idx + 1:,}/{total_batches:,} | "
                f"Samples: {samples_processed:,}/{len(dataset):,} | "
                f"Rate: {samples_per_sec:.1f} samples/sec | "
                f"Batch time: {batch_inference_time:.3f}s | "
                f"ETA: {eta_str}"
            )

    log_progress("Concatenating results...")
    # process_predictions now ensures all outputs are at least 1D, so we can concatenate directly
    probs_all = np.concatenate(probs_all, axis=0).astype(np.float32)
    labels_all = np.concatenate(labels_all, axis=0).astype(np.int32)

    log_progress(f"Computing metrics for {len(labels_all):,} samples...")

    # Compute overall metrics
    log_progress("Computing overall metrics...")
    overall_metrics = compute_metrics_with_confusion(probs_all, labels_all)
    log_progress(f"  Overall accuracy: {overall_metrics['accuracy']:.2f}%")

    # Compute per-language metrics
    per_language_metrics = compute_per_category_metrics(
        probs_all, labels_all, languages_all, "language"
    )

    # Compute per-dataset metrics
    per_dataset_metrics = compute_per_category_metrics(
        probs_all, labels_all, datasets_all, "dataset"
    )

    # --- ROC/PR AUC and Curves ---
    roc_auc_val = 0.0
    pr_auc_val = 0.0
    try:
        fpr, tpr, _ = roc_curve(labels_all, probs_all)
        roc_auc_val = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(
            labels_all, probs_all
        )
        pr_auc_val = average_precision_score(labels_all, probs_all)

        # Generate plots if path provided
        if plot_path:
            log_progress("Generating ROC/PR curves...")
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC (AUC = {roc_auc_val:.2f})",
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
                label=f"PR (AUC = {pr_auc_val:.2f})",
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
        log_progress(f"Error computing ROC/PR curves: {e}", "WARN")

    # --- Best threshold search ---
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.1, 0.9, 17):
        p_t = (probs_all > thresh).astype(int)
        f1_t = f1_score(labels_all, p_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
    log_progress(f"Best threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")

    # --- Save detailed CSV ---
    if csv_path:
        log_progress("Saving detailed per-sample results...")
        preds_05 = (probs_all > 0.5).astype(int)
        preds_best = (probs_all > best_thresh).astype(int)
        correct_05 = (preds_05 == labels_all).astype(int)
        correct_best = (preds_best == labels_all).astype(int)

        df = pd.DataFrame(
            {
                "id": ids_all,
                "true_label": labels_all.tolist(),
                "prob": probs_all.tolist(),
                "pred_0.5": preds_05.tolist(),
                "correct_0.5": correct_05.tolist(),
                f"pred_{best_thresh:.2f}": preds_best.tolist(),
                f"correct_{best_thresh:.2f}": correct_best.tolist(),
                "language": languages_all,
                "dataset": datasets_all,
                "audio_path": audio_paths_all,
            }
        )
        df.to_csv(csv_path, index=False)
        log_progress(f"💾 Detailed results saved to: {csv_path}")

    log_progress("Accuracy evaluation complete!")
    return {
        "overall": overall_metrics,
        "per_language": per_language_metrics,
        "per_dataset": per_dataset_metrics,
        "total_samples": len(labels_all),
        "unique_languages": sorted(list(set(languages_all))),
        "unique_datasets": sorted(list(set(datasets_all))),
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "prob_min": float(probs_all.min()),
        "prob_max": float(probs_all.max()),
        "prob_mean": float(probs_all.mean()),
        "prob_std": float(probs_all.std()),
        "csv_path": csv_path,
        "plot_path": plot_path,
    }


@app.function(
    image=image,
    gpu="T4",
    memory=8192,
    cpu=6.0,
    volumes={"/data": volume},
    timeout=60 * 60,
)
def benchmark_modal(
    onnx_path: str,
    run_description: Optional[str] = None,
    dataset_path: Optional[str] = None,
    limit: Optional[int] = None,
    perf_runs: int = 100,
    markdown_output: Optional[str] = None,
):
    # Resolve ONNX path (supports HuggingFace downloads)
    resolved_onnx_path = resolve_onnx_path(onnx_path)

    fe = WhisperFeatureExtractor(chunk_length=AUDIO_SECONDS)  # 8 seconds

    # Check if this is a casual_conversations style dataset (has metadata.jsonl)
    metadata_path = (
        os.path.join(dataset_path, "metadata.jsonl") if dataset_path else None
    )

    if metadata_path and os.path.exists(metadata_path):
        # Use robust loader for casual_conversations format
        log_progress("Detected casual_conversations format dataset")
        raw_dataset = load_robust_dataset(dataset_path)
        dataset = AudioOnlyBenchmarkDataset(raw_dataset, fe)
    else:
        # Use original HuggingFace format
        import train

        raw_dataset = load_dataset_at(dataset_path)
        dataset = train.OnDemandSmartTurnDataset(raw_dataset, fe)

    return benchmark(
        onnx_path=resolved_onnx_path,
        run_description=run_description,
        dataset=dataset,
        limit=limit,
        perf_runs=perf_runs,
        markdown_output=markdown_output,
    )


def benchmark(
    onnx_path: str,
    run_description: Optional[str] = None,
    dataset: Optional[Dataset] = None,
    limit: Optional[int] = None,
    perf_runs: int = 100,
    markdown_output: Optional[str] = None,
    batch_size: int = 32,
):
    # Generate output paths if not provided
    if markdown_output is None:
        markdown_output = generate_output_path(
            onnx_path=onnx_path, run_description=run_description, extension="md"
        )
    # Generate CSV and plot paths based on markdown path
    base_path = markdown_output.rsplit(".", 1)[0]
    csv_path = f"{base_path}.csv"
    plot_path = f"{base_path}.png"

    log_progress("=" * 80)
    log_progress("Starting benchmark")
    log_progress("=" * 80)
    log_progress(f"Model: {onnx_path}")
    log_progress(f"Sample limit: {limit if limit else 'None'}")
    log_progress(f"Performance runs: {perf_runs}")
    log_progress(f"Output file: {markdown_output}")
    log_progress("")

    results = {"onnx_path": onnx_path}

    # Detect GPU model name
    gpu_model_name = get_gpu_model_name()
    log_progress(f"Detected GPU model: {gpu_model_name}")

    # Providers
    log_progress("Checking available ONNX providers...")
    providers = ort.get_available_providers()
    log_progress(f"Available providers: {providers}")

    cpu_prov = ["CPUExecutionProvider"]
    gpu_prov = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in providers
        else None
    )

    if gpu_prov:
        log_progress("GPU (CUDA) provider available")
    else:
        log_progress(
            "GPU (CUDA) provider not available - will skip GPU benchmarks"
        )

    # Sessions
    log_progress("Building inference sessions...")
    cpu_sess = build_session(onnx_path, providers=cpu_prov)
    gpu_sess = (
        build_session(onnx_path, providers=gpu_prov) if gpu_prov else None
    )

    # ---------- Performance (zeros → direct) ----------
    log_progress("=" * 50)
    log_progress("Direct inference performance")

    results["perf_cpu"] = run_perf(cpu_sess, runs=perf_runs)
    if gpu_sess:
        results["perf_gpu"] = run_perf(gpu_sess, runs=perf_runs)
    else:
        results["perf_gpu"] = {
            "note": "CUDAExecutionProvider not available; skipped."
        }
        log_progress(
            "Skipping GPU inference performance test (CUDA not available)"
        )

    # ---------- Feature extraction on 8s zero audio ----------
    log_progress("=" * 50)
    log_progress("Feature extraction performance")

    fe = WhisperFeatureExtractor(chunk_length=AUDIO_SECONDS)
    zero_audio = _zero_audio(AUDIO_SECONDS, SAMPLING_RATE)
    results["perf_feature_extractor"] = run_fe_perf(
        fe, zero_audio, runs=perf_runs
    )

    # ---------- End-to-end (feature extraction + inference) ----------
    log_progress("=" * 50)
    log_progress("End-to-end performance")

    results["perf_e2e_cpu"] = run_e2e_perf(
        cpu_sess, fe, zero_audio, runs=perf_runs
    )
    if gpu_sess:
        results["perf_e2e_gpu"] = run_e2e_perf(
            gpu_sess, fe, zero_audio, runs=perf_runs
        )
    else:
        results["perf_e2e_gpu"] = {
            "note": "CUDAExecutionProvider not available; skipped."
        }

    # ---------- Accuracy (dataset) ----------
    if dataset:
        results["accuracy"] = run_accuracy(
            onnx_path=onnx_path,
            dataset=dataset,
            limit=limit,
            batch_size=batch_size,
            csv_path=csv_path,
            plot_path=plot_path,
        )
    else:
        results["accuracy"] = {"note": "No dataset_path provided; skipped."}

    # Generate markdown report
    markdown_report = format_markdown_report(results, gpu_model_name)

    # Write markdown report to file
    with open(markdown_output, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    # Print both the raw results (for debugging) and confirmation of file write
    print("=" * 80)
    print("Raw results:")
    print(results)
    print("\n" + "=" * 80)
    print(f"Markdown report written to: {markdown_output}")
    if dataset:
        print(f"Detailed CSV written to: {csv_path}")
        print(f"ROC/PR curves written to: {plot_path}")

    return results


@app.local_entrypoint()
def main(
    onnx_path: str,
    run_description: Optional[str] = None,
    dataset_path: str = "",
    limit: Optional[int] = None,
    perf_runs: int = 100,
    markdown_output: Optional[str] = None,
):
    res = benchmark_modal.remote(
        onnx_path,
        run_description,
        dataset_path if dataset_path else None,
        limit,
        perf_runs,
        markdown_output,
    )
    print(res)
