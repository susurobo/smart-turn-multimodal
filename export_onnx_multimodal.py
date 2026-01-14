import modal
import torch
import torch.nn as nn
import os
import numpy as np
from safetensors.torch import load_file
from transformers import WhisperModel, WhisperConfig
from torchvision.models.video import r3d_18
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    quant_pre_process,
    QuantFormat,
    CalibrationMethod,
)

# --- CONFIGURATION ---
DEFAULT_RUN_DIR = "/data/output/mm_run_20260111_1904"
CALIBRATION_SAMPLES = 256  # Number of samples for INT8 calibration

app = modal.App("export-mm-onnx")
volume = modal.Volume.from_name("endpointing", create_if_missing=False)

# Minimal Image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.36.2",
        "safetensors",
        "numpy<2.0",
        "av",
        "onnx",
        "onnxruntime",
    )
)


# --- 1. DEFINE ARCHITECTURE (MATCHING CHECKPOINT) ---
class SmartTurnV3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = WhisperModel(config).encoder
        self.d_model = config.d_model

        # --- FIX 1: Pool Attention Hidden Size = 256 ---
        # (Default was 128, but your checkpoint has 256)
        self.pool_attention = nn.Sequential(
            nn.Linear(self.d_model, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # --- FIX 2: Classifier matches train.py architecture ---
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_features):
        outputs = self.encoder(input_features)
        hidden_states = outputs.last_hidden_state
        attn_weights = torch.softmax(self.pool_attention(hidden_states), dim=1)
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1)
        return self.classifier(pooled_output)


class SmartTurnMultimodal(SmartTurnV3Model):
    def __init__(self, config):
        super().__init__(config)

        # Video Encoder
        self.video_backbone = r3d_18(weights=None)
        self.video_backbone.fc = nn.Linear(
            self.video_backbone.fc.in_features, 256
        )

        # Fusion Layer
        audio_dim = config.d_model
        video_dim = 256

        self.fusion_layer = nn.Sequential(
            nn.Linear(audio_dim + video_dim, audio_dim),
            nn.LayerNorm(audio_dim),
            nn.GELU(),
        )

    def forward(self, input_features, pixel_values):
        encoder_outputs = self.encoder(input_features=input_features)
        hidden_states = encoder_outputs.last_hidden_state

        attn_weights = self.pool_attention(hidden_states)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        audio_emb = torch.sum(hidden_states * attn_weights, dim=1)

        video_emb = self.video_backbone(pixel_values)

        combined = torch.cat((audio_emb, video_emb), dim=1)
        fused_emb = self.fusion_layer(combined)

        logits = self.classifier(fused_emb)
        probs = torch.sigmoid(logits)

        return probs


# --- CALIBRATION CLASSES FOR INT8 QUANTIZATION ---
class MultimodalCalibrationDataset:
    """Generates synthetic calibration data for multimodal quantization."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        print(f"📊 Calibration dataset: {num_samples} synthetic samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random but realistic-range inputs
        # Audio: mel spectrogram values typically range [-1, 1] after normalization
        audio = np.random.randn(80, 800).astype(np.float32) * 0.5
        # Video: normalized pixel values (after ImageNet normalization)
        video = np.random.randn(3, 32, 112, 112).astype(np.float32) * 0.25
        return audio, video


class MultimodalCalibrationDataReader(CalibrationDataReader):
    """ONNX Runtime calibration data reader for multimodal model."""

    def __init__(self, calibration_dataset):
        self.calibration_dataset = calibration_dataset
        self.iterator = iter(range(len(calibration_dataset)))

    def get_next(self):
        try:
            idx = next(self.iterator)
            audio, video = self.calibration_dataset[idx]
            # Add batch dimension
            audio = np.expand_dims(audio, axis=0)
            video = np.expand_dims(video, axis=0)
            return {
                "input_features": audio,
                "pixel_values": video,
            }
        except StopIteration:
            return None


def quantize_multimodal_onnx(
    fp32_path: str, output_path: str, num_samples: int
):
    """Quantize multimodal ONNX model to INT8."""
    print(f"🔧 Starting quantization of {fp32_path}...")

    # 1. Pre-process the model
    pre_path = fp32_path.replace(".onnx", "_pre.onnx")
    print("📦 Running quant_pre_process...")
    quant_pre_process(
        fp32_path,
        pre_path,
        skip_optimization=False,
        skip_symbolic_shape=True,
        verbose=0,
    )

    # 2. Create calibration dataset
    calibration_dataset = MultimodalCalibrationDataset(num_samples)
    calibration_reader = MultimodalCalibrationDataReader(calibration_dataset)

    # 3. Run static quantization
    print(
        f"⚙️ Running quantize_static with {num_samples} calibration samples..."
    )
    quantize_static(
        model_input=pre_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Entropy,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )

    # 4. Clean up temp file
    if os.path.exists(pre_path):
        os.remove(pre_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ Quantized model saved: {output_path} ({size_mb:.2f} MB)")

    return output_path


@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=1800)
def run_export(run_dir: str):
    checkpoint_path = f"{run_dir}/final_multimodal/model.safetensors"
    gpu_onnx_path = f"{run_dir}/smart-turn-multimodal-gpu.onnx"
    cpu_onnx_path = f"{run_dir}/smart-turn-multimodal-cpu.onnx"

    print(f"🚀 Starting export from: {checkpoint_path}")
    print(f"   GPU model -> {gpu_onnx_path}")
    print(f"   CPU model -> {cpu_onnx_path}")

    # 1. Configure
    config = WhisperConfig.from_pretrained("openai/whisper-tiny")

    # --- FIX 3: Positional Embeddings = 400 ---
    config.max_source_positions = 400
    print(f"🔧 Configured max_source_positions: {config.max_source_positions}")

    print("🏗️ Initializing model architecture (Width: 256)...")
    model = SmartTurnMultimodal(config)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ Cannot find checkpoint at {checkpoint_path}"
        )

    print("💾 Loading safetensors...")
    state_dict = load_file(checkpoint_path)

    # Load weights (Should be strict match now)
    model.load_state_dict(state_dict, strict=True)
    print("✅ Weights loaded perfectly (Strict Match)!")

    model.eval()
    model.cpu()

    # 4. Create Dummy Inputs
    print("🎨 Creating dummy inputs...")
    # 800 frames / 2 stride = 400 positions
    audio_dummy = torch.randn(1, 80, 800)
    video_dummy = torch.randn(1, 3, 32, 112, 112)

    # 5. Export FP32 GPU Model
    print("\n" + "=" * 50)
    print("📦 Step 1: Exporting FP32 GPU model...")
    print("=" * 50)
    torch.onnx.export(
        model,
        (audio_dummy, video_dummy),
        gpu_onnx_path,
        input_names=["input_features", "pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    # Verify GPU model
    onnx_model = onnx.load(gpu_onnx_path)
    onnx.checker.check_model(onnx_model)

    gpu_size_mb = os.path.getsize(gpu_onnx_path) / 1024 / 1024
    print(f"✅ GPU Model Export Complete! Size: {gpu_size_mb:.2f} MB")

    # 6. Quantize to INT8 CPU Model
    print("\n" + "=" * 50)
    print("📦 Step 2: Quantizing to INT8 CPU model...")
    print("=" * 50)
    quantize_multimodal_onnx(
        fp32_path=gpu_onnx_path,
        output_path=cpu_onnx_path,
        num_samples=CALIBRATION_SAMPLES,
    )

    cpu_size_mb = os.path.getsize(cpu_onnx_path) / 1024 / 1024

    # 7. Summary
    print("\n" + "=" * 50)
    print("🎉 Export Complete!")
    print("=" * 50)
    print(f"   GPU Model (FP32): {gpu_onnx_path} ({gpu_size_mb:.2f} MB)")
    print(f"   CPU Model (INT8): {cpu_onnx_path} ({cpu_size_mb:.2f} MB)")
    print(f"   Size Reduction: {(1 - cpu_size_mb / gpu_size_mb) * 100:.1f}%")


@app.local_entrypoint()
def main(run_dir: str = DEFAULT_RUN_DIR):
    run_export.remote(run_dir)
