import os
import torch
import torch.nn as nn
import av
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
from transformers import (
    WhisperFeatureExtractor,
    WhisperConfig,
    Trainer,
    TrainingArguments,
)

from torchvision.models.video import r3d_18, R3D_18_Weights
from datasets import load_dataset, concatenate_datasets

# Import your existing components
from train import (
    SmartTurnV3Model,
    CONFIG,
    compute_metrics,
    prepare_datasets_ondemand,
)
from audio_utils import truncate_audio_to_last_n_seconds
from logger import log


# --- 1. The Multimodal Model Architecture ---
class SmartTurnMultimodal(SmartTurnV3Model):
    def __init__(self, config, pretrained_audio_path=None):
        super().__init__(config)

        # 1. Video Encoder -> Projects to fixed 256 dim
        self.video_backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.video_backbone.fc = nn.Linear(
            self.video_backbone.fc.in_features, 256
        )

        # 2. Fusion Layer -> Dynamically sized
        # Audio Size comes from config (384 for Tiny, 512 for Base)
        audio_dim = config.d_model
        video_dim = 256

        # --- FIX: Output must match audio_dim (384) to satisfy the frozen classifier ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(
                audio_dim + video_dim, audio_dim
            ),  # Output matches config.d_model
            nn.LayerNorm(audio_dim),
            nn.GELU(),
        )

        if pretrained_audio_path:
            self.load_audio_weights(pretrained_audio_path)

    def load_audio_weights(self, path):
        log.info(f"Loading pretrained audio weights from: {path}")

        # 1. Load the state dictionary (works for both Local Paths and HF Hub)
        from transformers import WhisperModel

        try:
            # We use WhisperModel to load the weights safely regardless of shape
            base_audio = WhisperModel.from_pretrained(path)
            state_dict = base_audio.state_dict()
            del base_audio
        except Exception as e:
            log.error(f"Failed to load checkpoint from {path}. Error: {e}")
            raise e

        # 2. Inspect Positional Embeddings
        pos_key = "encoder.embed_positions.weight"

        if pos_key in state_dict:
            # The size our model expects (400) vs what is in the file (1500 OR 400)
            target_size = self.encoder.embed_positions.weight.shape[0]
            loaded_size = state_dict[pos_key].shape[0]

            log.info(
                f"Positional Embeddings - Model expects: {target_size}, Checkpoint has: {loaded_size}"
            )

            if loaded_size > target_size:
                log.info(
                    f"Action: SLICING weights from {loaded_size} to {target_size} (Generic Whisper Mode)"
                )
                state_dict[pos_key] = state_dict[pos_key][:target_size, :]
            elif loaded_size == target_size:
                log.info(f"Action: DIRECT LOAD (Smart Turn Audio-Only Mode)")
            else:
                # Should not happen, but good to catch
                raise ValueError(
                    f"Checkpoint is too small ({loaded_size}) for this model ({target_size})!"
                )

        # 3. Load cleanly
        # strict=False allows us to ignore 'decoder', 'video_backbone', etc.
        keys = self.load_state_dict(state_dict, strict=False)
        log.info(f"Weights loaded successfully.")

    def freeze_audio_branch(self):
        log.info("Freezing Audio Branch (Encoder + Classifier)...")
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.pool_attention.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        log.info("Unfreezing all parameters for joint fine-tuning...")
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_features, pixel_values=None, labels=None):
        # 1. Audio Forward
        encoder_outputs = self.encoder(input_features=input_features)
        hidden_states = encoder_outputs.last_hidden_state

        # Attention Pooling (Preserves d_model dimension)
        attn_weights = self.pool_attention(hidden_states)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        audio_emb = torch.sum(
            hidden_states * attn_weights, dim=1
        )  # [B, d_model]

        # 2. Video Forward
        if pixel_values is not None:
            video_emb = self.video_backbone(pixel_values)  # [B, 256]
        else:
            # Create Zeros of correct size (256)
            batch_size = audio_emb.shape[0]
            video_emb = torch.zeros(
                batch_size, 256, device=audio_emb.device, dtype=audio_emb.dtype
            )

        # 3. Late Fusion
        combined = torch.cat(
            (audio_emb, video_emb), dim=1
        )  # [B, d_model + 256]

        # This will now output [B, d_model] (384), matching the classifier
        fused_emb = self.fusion_layer(combined)

        # 4. Classification
        logits = self.classifier(fused_emb)
        probs = torch.sigmoid(logits)

        if labels is not None:
            pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(
                min=0.1, max=10.0
            )
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits.view(-1), labels.float().view(-1))
            return {"loss": loss, "logits": probs}

        return {"logits": probs}


# --- 2. Multimodal Dataset Loader ---


class OnDemandMultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, feature_extractor, dataset_root):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.dataset_root = dataset_root

        # Standard video transform stats (Kinetics-400)
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645])
        self.std = torch.tensor([0.22803, 0.22145, 0.216989])

    def __len__(self):
        return len(self.dataset)

    def _load_video_frames(self, rel_path):
        """Loads last 32 frames of video, resizes to 112x112"""
        # --- FIX: Handle missing video paths safely ---
        if rel_path is None:
            return None

        # Handle absolute paths vs relative paths
        if os.path.isabs(rel_path):
            full_path = rel_path
        else:
            full_path = os.path.join(self.dataset_root, rel_path)

        if not os.path.exists(full_path):
            log.warning(f"Video file not found: {full_path}")
            return None
        # ----------------------------------------------

        try:
            container = av.open(full_path)
            stream = container.streams.video[0]

            frames = []
            for frame in container.decode(stream):
                frames.append(frame)

            # Take last 32 frames (approx 1 sec at 30fps)
            clip = frames[-32:]

            if len(clip) < 8:  # Too short to be useful
                return None

            # Convert to Tensor [T, H, W, C] -> [C, T, H, W]
            tensors = []
            for frame in clip:
                img = frame.to_image().resize(
                    (112, 112)
                )  # Resize small for R3D-18
                t_img = torch.from_numpy(np.array(img)).float() / 255.0
                tensors.append(t_img)

            video = torch.stack(tensors)
            video = video.permute(3, 0, 1, 2)

            # Normalize
            video = (video - self.mean[:, None, None, None]) / self.std[
                :, None, None, None
            ]

            return video  # [3, 32, 112, 112]

        except Exception as e:
            log.warning(f"Failed to load video {rel_path}: {e}")
            return None

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # A. Process Audio (Standard)
        audio_array = sample["audio"]["array"]
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )

        # B. Process Video (New)
        pixel_values = None
        if sample.get("video_path"):
            pixel_values = self._load_video_frames(sample["video_path"])

        label = 1 if sample["endpoint_bool"] else 0

        return {
            "input_features": inputs.input_features.squeeze(0),
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


@dataclass
class MultimodalCollator:
    def __call__(self, features):
        batch = {}
        batch["input_features"] = torch.stack(
            [f["input_features"] for f in features]
        )
        batch["labels"] = torch.stack([f["labels"] for f in features])

        # Stack videos. If any in batch is None, we must handle padding or drop.
        # Simple approach: If video missing, supply a zero-tensor of correct shape.
        videos = []
        has_video = False
        for f in features:
            v = f.get("pixel_values")
            if v is not None:
                videos.append(v)
                has_video = True
            else:
                # Padding shape [3, 32, 112, 112]
                videos.append(torch.zeros(3, 32, 112, 112))

        if has_video:
            batch["pixel_values"] = torch.stack(videos)
        else:
            batch["pixel_values"] = None

        return batch


def run_multimodal_training(
    dataset_path: str,
    run_name: str,
    # Add a default, but allow overriding
    base_model_path: str = "openai/whisper-tiny",
):
    # 1. Setup Config & Feature Extractor
    # We start from the base Whisper model since v3 weights are ONNX-only
    config = WhisperConfig.from_pretrained("openai/whisper-tiny")
    feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    # 2. Load Datasets
    log.info("Loading datasets...")

    # A. Your Video Dataset (Uploaded to Modal Volume)
    # The 'data_dir' argument tells audiofolder where to look
    video_dataset = load_dataset(
        "audiofolder", data_dir=dataset_path, split="train"
    )

    # B. The Official Smart Turn Audio Dataset (From Hugging Face)
    # We use the official training data so the model retains its core accuracy
    audio_only_dataset = load_dataset(
        "pipecat-ai/smart-turn-data-v3.2-train", split="train"
    )

    # 3. Align Columns for Concatenation
    # The audio dataset doesn't have 'video_path', so we add it as None
    # This allows us to merge them seamlessly.
    def add_missing_columns(example):
        example["video_path"] = None
        example["visual_label"] = None
        return example

    audio_only_dataset = audio_only_dataset.map(add_missing_columns)

    # Ensure both have the same set of columns before merging
    # We select only the columns our loader cares about
    common_columns = ["audio", "endpoint_bool", "video_path"]

    video_dataset = video_dataset.select_columns(common_columns)
    audio_only_dataset = audio_only_dataset.select_columns(common_columns)

    # 4. Merge & Split
    # We use 100% of video data for training + a portion of audio data
    # (Or mix them fully. Here we concatenate everything).
    full_dataset = concatenate_datasets([video_dataset, audio_only_dataset])

    # 4. Merge & Split with OVERSAMPLING

    # Calculate how many times we need to repeat the video dataset
    # to make it roughly 10% of the total size.
    total_audio = len(audio_only_dataset)  # ~270,000
    total_video = len(video_dataset)  # ~22

    # Target: We want ~30,000 video samples (10% of 270k)
    # 22 * X = 30,000  =>  X = ~1300
    if total_video > 0:
        repeat_factor = int((total_audio * 0.1) / total_video)
        repeat_factor = max(1, repeat_factor)

        log.info(
            f"Oversampling video dataset {repeat_factor} times to balance training..."
        )

        # Create a list of the dataset repeated N times
        video_datasets_list = [video_dataset] * repeat_factor
        balanced_video_dataset = concatenate_datasets(video_datasets_list)
    else:
        balanced_video_dataset = video_dataset

    # Now concatenate the massive audio set with the boosted video set
    full_dataset = concatenate_datasets(
        [balanced_video_dataset, audio_only_dataset]
    )

    # Create Train/Test Split (90/10)
    # SHUFFLE IS CRITICAL here so the repeated videos are spread out
    # We shuffle to ensure batches contain a mix of Audio-Only and Audio-Video samples
    split = full_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)

    # 5. Wrap in Multimodal Loader
    # Note: dataset_root is only needed for the video_dataset part.
    # Our loader handles absolute paths or relative paths.
    # Since 'audio_only_dataset' entries have video_path=None, the loader will generate zero-tensors for them.
    train_ds = OnDemandMultimodalDataset(
        split["train"], feature_extractor, dataset_root=dataset_path
    )
    eval_ds = OnDemandMultimodalDataset(
        split["test"], feature_extractor, dataset_root=dataset_path
    )

    # 6. Initialize Model
    # We load the PRETRAINED WHISPER weights (not SmartTurn v3 ONNX)
    model = SmartTurnMultimodal(
        config,
        pretrained_audio_path=base_model_path,
    )

    # --- STAGE 1: FREEZE AUDIO ---
    # We still freeze audio initially to let the R3D-18 video encoder learn
    # to output embeddings compatible with Whisper's latent space.
    model.freeze_audio_branch()

    args = TrainingArguments(
        output_dir=f"/data/output/{run_name}",
        per_device_train_batch_size=32,
        learning_rate=5e-4,
        num_train_epochs=1,  # 1 Epoch is enough for alignment since we have mixed data
        save_steps=200,
        logging_steps=50,
        report_to=["wandb"],
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=MultimodalCollator(),
        compute_metrics=compute_metrics,
    )

    log.info("Starting Stage 1: Alignment (Audio Frozen)...")
    trainer.train()
    trainer.save_model(f"/data/output/{run_name}/stage1_aligned")

    # --- STAGE 2: JOINT FINETUNING ---
    model.unfreeze_all()

    args.learning_rate = 1e-5  # Very low LR to polish both modalities
    args.num_train_epochs = 3
    args.output_dir = f"/data/output/{run_name}_stage2"

    # Update trainer args (hacky but works)
    trainer.args = args

    log.info("Starting Stage 2: Joint Finetuning...")
    trainer.train()
    trainer.save_model(f"/data/output/{run_name}/final_multimodal")
