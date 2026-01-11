import csv
import json
import uuid
import shutil
import os
from pathlib import Path

# --- Configuration ---
INPUT_CSV = "segmented_output/manual_labels.csv"
OUTPUT_ROOT = "smart_turn_multimodal_dataset"
DATASET_NAME = "CasualConversations_Video"  # Tag for the 'dataset' column


def create_hf_fully_compatible_dataset():
    root = Path(OUTPUT_ROOT)
    audio_dir = root / "audio"
    video_dir = root / "video"

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    entries = []

    # Resolve paths relative to the CSV file's directory
    csv_path = Path(INPUT_CSV)
    csv_dir = csv_path.parent

    print(f"Processing {INPUT_CSV}...")

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        for row in reader:
            # --- File Handling ---
            # Paths in CSV are relative to the CSV file's directory
            relative_path = Path(row["filename"].strip())
            original_video_path = csv_dir / relative_path
            original_audio_path = original_video_path.with_suffix(".wav")

            # Create unique filename: "folder_filename"
            base_name = f"{relative_path.parent.name}_{relative_path.stem}"
            new_audio_name = f"{base_name}.wav"
            new_video_name = f"{base_name}.mp4"

            # Copy files (Skip if missing)
            try:
                shutil.copy2(original_audio_path, audio_dir / new_audio_name)
                shutil.copy2(original_video_path, video_dir / new_video_name)
            except FileNotFoundError:
                print(
                    f"Skipping missing file: {relative_path} (looked in {original_video_path})"
                )
                continue

            # --- Label Parsing ---
            is_complete = row["label"].strip().lower() == "complete"
            comment = row["comment"].strip() if row["comment"] else ""

            # --- Entry Construction ---
            entry = {
                # 1. CORE AUDIO FIELDS (Required by AudioFolder)
                "file_name": f"audio/{new_audio_name}",
                # 2. SMART TURN SCHEMA (Matches existing HF dataset)
                "id": str(uuid.uuid4()),
                "language": "eng",
                "endpoint_bool": is_complete,
                "midfiller": False,  # Default for now
                "endfiller": False,  # Default for now
                "synthetic": False,  # Real video data
                "spoken_text": None,  # Transcript not provided
                "dataset": DATASET_NAME,  # Helps track source in mixed training
                # 3. MULTIMODAL EXTENSION (New field, ignored by standard tools)
                "video_path": f"video/{new_video_name}",
                "visual_label": comment,  # Keep your rich labels for evaluation!
            }

            entries.append(entry)

    # Write Metadata
    with open(root / "metadata.jsonl", "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Done! Created fully compatible dataset at {root}")
    print(
        "Features: audio, id, language, endpoint_bool, midfiller, endfiller, "
        "synthetic, spoken_text, dataset, video_path, visual_label"
    )


if __name__ == "__main__":
    create_hf_fully_compatible_dataset()
