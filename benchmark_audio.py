import os
import time
import json
import librosa
import pandas as pd
import matplotlib

# Force headless backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import gc

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
from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk

# --- CONSTANTS ---
SAMPLING_RATE = 16000
N_MELS = 80
N_FRAMES = 800  # 8s
FEATURE_SHAPE = (1, N_MELS, N_FRAMES)
AUDIO_SECONDS = 8

app = modal.App("endpointing-benchmark-audio")
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
        "huggingface_hub",
        "matplotlib",
    )
    .run_commands(
        "pip install onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
    )
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/lib/x86_64-linux-gnu"
        }
    )
)


def log_progress(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


class AudioBenchmarkDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        audio_array = None

        # Robust Load
        if (
            "audio" in item
            and isinstance(item["audio"], dict)
            and "array" in item["audio"]
        ):
            audio_array = item["audio"]["array"]
        elif "audio" in item and isinstance(item["audio"], str):
            try:
                audio_array, _ = librosa.load(
                    item["audio"], sr=SAMPLING_RATE, mono=True
                )
            except Exception:
                pass

        if audio_array is None:
            audio_array = np.zeros(16000 * 8, dtype=np.float32)

        # --- FIX: TAIL TRUNCATION ---
        max_samples = 16000 * 8
        if len(audio_array) > max_samples:
            audio_array = audio_array[-max_samples:]  # Take the END

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=max_samples,
            truncation=True,
        )
        feats = inputs.input_features[0][:, :N_FRAMES]
        if feats.shape[-1] < N_FRAMES:
            pad = np.zeros((80, N_FRAMES - feats.shape[-1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=1)

        label = item.get("endpoint_bool")
        if label is None:
            label = item.get("label", 0)

        file_id = item.get("file_name") or item.get("id") or str(idx)
        lang = item.get("language", "unknown")
        ds_name = item.get("dataset", "unknown")

        return {
            "input_features": feats,
            "labels": label,
            "id": file_id,
            "language": lang,
            "dataset": ds_name,
        }


@dataclass
class AudioCollator:
    def __call__(self, features):
        input_features = np.stack([f["input_features"] for f in features])
        labels = np.array(
            [1 if f["labels"] else 0 for f in features], dtype=np.int32
        )
        ids = [f["id"] for f in features]
        languages = [f["language"] for f in features]
        datasets = [f["dataset"] for f in features]

        return {
            "input_features": input_features,
            "labels": labels,
            "ids": ids,
            "languages": languages,
            "datasets": datasets,
        }


def load_robust_dataset(dataset_path: str):
    metadata_path = os.path.join(dataset_path, "metadata.jsonl")
    if os.path.exists(dataset_path) and os.path.exists(metadata_path):
        log_progress(
            "Detected RAW local dataset. Loading from metadata.jsonl..."
        )
        ds = load_dataset("json", data_files=metadata_path, split="train")

        def fix_paths(example):
            if "audio_path" in example and example["audio_path"]:
                if not example["audio_path"].startswith("/"):
                    example["audio"] = os.path.join(
                        dataset_path, example["audio_path"]
                    )
                else:
                    example["audio"] = example["audio_path"]
            return example

        ds = ds.map(fix_paths)
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


def run_benchmark_remote(repo_id, filename, dataset_path, limit, batch_size):
    if os.path.exists(repo_id):
        model_path = repo_id
    else:
        log_progress(f"Downloading model {filename} from {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    log_progress(f"Model path: {model_path}")
    log_progress(f"Loading dataset: {dataset_path}")
    try:
        raw_dataset = load_robust_dataset(dataset_path)
        if limit:
            raw_dataset = raw_dataset.select(range(limit))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

    log_progress(f"Dataset loaded: {len(raw_dataset)} samples")
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    test_ds = AudioBenchmarkDataset(raw_dataset, fe)

    # Num workers 0 for safety
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=AudioCollator(),
        num_workers=0,
    )

    sess_options = ort.SessionOptions()
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CUDAExecutionProvider"],
        )
        if "CUDAExecutionProvider" in session.get_providers():
            log_progress("✅ Using CUDA Execution Provider")
        else:
            log_progress("⚠️ CUDA Requested but not active. Using CPU.")
    except Exception as e:
        log_progress(f"⚠️ CUDA Init Failed: {e}. Using CPU.")
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name

    all_probs = []
    all_labels = []
    all_ids = []
    all_langs = []
    all_datasets = []

    log_progress("Starting Inference...")
    t0 = time.time()

    for i, batch in enumerate(loader):
        inputs = {input_name: batch["input_features"]}
        out = session.run([output_name], inputs)[0]
        probs = out.squeeze()
        if probs.ndim == 0:
            probs = np.array([probs])

        all_probs.extend(probs.tolist())
        all_labels.extend(batch["labels"].tolist())
        all_ids.extend(batch["ids"])
        all_langs.extend(batch["languages"])
        all_datasets.extend(batch["datasets"])

        if (i + 1) % 50 == 0:
            log_progress(f"Batch {i + 1}/{len(loader)}")
            gc.collect()  # Force cleanup

    dt = time.time() - t0
    log_progress(f"Inference done in {dt:.2f}s")

    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)

    # --- DIAGNOSTICS & PLOTS ---
    print("\n" + "=" * 60)
    print("AUDIO-ONLY DIAGNOSTIC REPORT")
    print("=" * 60)

    print(
        f"\n[Probs] Min: {probs_np.min():.4f} | Max: {probs_np.max():.4f} | Mean: {probs_np.mean():.4f}"
    )

    try:
        fpr, tpr, _ = roc_curve(labels_np, probs_np)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(labels_np, probs_np)
        pr_auc = average_precision_score(labels_np, probs_np)

        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"PR AUC Score:  {pr_auc:.4f}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = "/data/benchmark"
        os.makedirs(output_dir, exist_ok=True)

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
        plt.title("Audio-Only ROC Curve")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR (AUC = {pr_auc:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Audio-Only PR Curve")
        plt.legend(loc="lower left")

        plot_path = f"{output_dir}/audio_curves_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"\n📈 Curves saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Error plotting curves: {e}")

    df = pd.DataFrame(
        {
            "id": all_ids,
            "language": all_langs,
            "dataset": all_datasets,
            "true_label": all_labels,
            "prob": all_probs,
            "pred_0.5": (probs_np > 0.5).astype(int),
        }
    )
    csv_path = f"{output_dir}/audio_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Detailed results saved to: {csv_path}")
    print("=" * 60 + "\n")

    return {}


# 24GB Memory + T4 GPU
@app.function(
    image=image, gpu="T4", volumes={"/data": volume}, timeout=3600, memory=24576
)
def benchmark_entrypoint(
    repo_id: str, filename: str, dataset_path: str, limit: int = None
):
    return run_benchmark_remote(
        repo_id, filename, dataset_path, limit, batch_size=8
    )  # Lower batch size


@app.local_entrypoint()
def main(
    repo_id: str = "onnx-community/smart-turn-v3-ONNX",
    filename: str = "onnx/model.onnx",
    dataset_path: str = "/data/datasets/casual_conversations",
    limit: int = None,
):
    print(f"Starting Audio-Only Benchmark on {dataset_path}")
    benchmark_entrypoint.remote(repo_id, filename, dataset_path, limit)
