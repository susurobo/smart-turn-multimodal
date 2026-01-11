import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import glob
from datetime import datetime
from typing import Optional, List

import modal

app = modal.App("endpointing-training")
volume = modal.Volume.from_name("endpointing", create_if_missing=False)

# In train_modal.py

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.9.0",
        "transformers[torch]==4.48.2",
        "datasets==4.4.1",
        "scikit-learn==1.6.1",
        "numpy==2.3.4",
        "librosa",
        "soundfile",
        "wandb",
        "torchaudio==2.9.0",
        "torchcodec==0.8.1",
        "onnx==1.19.1",
        "onnxruntime-gpu==1.23.2",
        "onnxscript==0.5.6",
        # --- NEW DEPENDENCIES ---
        "torchvision",
        "av",
        # ------------------------
    )
    .add_local_python_source("logger")
    .add_local_python_source("train")
    .add_local_python_source("benchmark")
    .add_local_python_source("audio_utils")
    # Add your new script here once created
    .add_local_python_source("train_multimodal")
)

# In train_modal.py


@app.function(
    image=image,
    gpu="A10G",
    memory=32768,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
    env={
        "HF_HOME": "/data/cache/huggingface",  # Caches the Audio Dataset
        "TORCH_HOME": "/data/cache/torch",  # Caches the R3D-18 Video Model
    },
)
def multimodal_run(dataset_folder_name: str, run_name: str, base_model: str):
    import train_multimodal

    train_multimodal.run_multimodal_training(
        dataset_path=f"/data/datasets/{dataset_folder_name}",
        run_name=run_name,
        base_model_path=base_model,  # Pass it through
    )


@app.function(
    image=image,
    gpu="L4",
    memory=32768,
    cpu=8.0,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def training_run(run_name: str):
    import train

    return train.do_training_run(run_name=run_name)


@app.function(
    image=image,
    memory=131072,
    cpu=16.0,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def quantization_run(fp32_model_path: str):
    import train

    return train.do_quantization_run(
        fp32_model_path=fp32_model_path,
    )


@app.function(
    image=image,
    gpu="T4",
    memory=32768,
    cpu=8.0,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def benchmark_run(model_root: List[str]):
    import train

    model_paths = glob.glob(f"{model_root}/*.onnx")
    return train.do_benchmark_run(model_paths=model_paths)


@app.local_entrypoint()
def main(
    training_run_name: Optional[str] = None,
    quantize: Optional[str] = None,
    benchmark: Optional[str] = None,
    multimodal: Optional[str] = None,
    audio_checkpoint: str = "openai/whisper-tiny",
):
    if training_run_name is not None:
        training_run.remote(run_name=training_run_name)

    if quantize is not None:
        quantization_run.remote(fp32_model_path=quantize)

    if benchmark is not None:
        benchmark_run.remote(model_root=benchmark)

    if multimodal is not None:
        multimodal_run.remote(
            dataset_folder_name=multimodal,
            run_name=f"mm_run_{datetime.now().strftime('%Y%m%d_%H%M')}",
            base_model=audio_checkpoint,
        )
