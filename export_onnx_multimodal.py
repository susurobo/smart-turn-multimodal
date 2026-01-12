import modal
import torch
import torch.nn as nn
import os
import sys
from safetensors.torch import load_file
from transformers import WhisperModel, WhisperConfig
from torchvision.models.video import r3d_18, R3D_18_Weights

# --- CONFIGURATION ---
RUN_DIR = "/data/output/mm_run_20260111_1904"
CHECKPOINT_PATH = f"{RUN_DIR}/final_multimodal/model.safetensors"
OUTPUT_ONNX_PATH = f"{RUN_DIR}/model_fixed.onnx"

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


@app.function(image=image, volumes={"/data": volume}, gpu="T4")
def run_export():
    print(f"🚀 Starting export from: {CHECKPOINT_PATH}")

    # 1. Configure
    config = WhisperConfig.from_pretrained("openai/whisper-tiny")

    # --- FIX 3: Positional Embeddings = 400 ---
    config.max_source_positions = 400
    print(f"🔧 Configured max_source_positions: {config.max_source_positions}")

    print("🏗️ Initializing model architecture (Width: 256)...")
    model = SmartTurnMultimodal(config)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"❌ Cannot find checkpoint at {CHECKPOINT_PATH}"
        )

    print(f"💾 Loading safetensors...")
    state_dict = load_file(CHECKPOINT_PATH)

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

    # 5. Export
    print(f"📦 Exporting to {OUTPUT_ONNX_PATH}...")
    torch.onnx.export(
        model,
        (audio_dummy, video_dummy),
        OUTPUT_ONNX_PATH,
        input_names=["input_features", "pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    size_mb = os.path.getsize(OUTPUT_ONNX_PATH) / 1024 / 1024
    print(f"✅ Export Complete! Size: {size_mb:.2f} MB")


@app.local_entrypoint()
def main():
    run_export.remote()
