#!/usr/bin/env python3
"""
Unified Video Dataset Preparation Script

This script combines face cropping, VAD-based segmentation, and HF-compatible
metadata generation into a single pipeline.

Key features:
- Face detection on original video (once per video) for consistent cropping
- VAD-based segmentation using Silero VAD
- Outputs HF-compatible dataset with metadata.jsonl
- Supports processing multiple videos into the same dataset
- Generates placeholder labels (null) for manual annotation

Usage:
    # Process single video
    python prepare_video_dataset.py video.mp4 output_dir/

    # Process directory of videos
    python prepare_video_dataset.py videos_dir/ output_dir/

    # Append more videos to existing dataset
    python prepare_video_dataset.py another_video.mp4 output_dir/
"""

import os
import sys
import json
import uuid
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch
import torchaudio
from facenet_pytorch import MTCNN
from tqdm import tqdm


# --- Configuration Defaults ---
DEFAULT_SILENCE_MS = 500
DEFAULT_FACE_SIZE = 224
DEFAULT_MARGIN = 40
METADATA_FILE = "metadata.jsonl"


def check_ffmpeg():
    """Checks if FFmpeg is installed and accessible."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("❌ CRITICAL ERROR: FFmpeg is not installed or not in your PATH.")
        print("👉 On macOS, run: brew install ffmpeg")
        sys.exit(1)


def load_silero_vad():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks,
    ) = utils
    return model, get_speech_timestamps, read_audio


def extract_audio_from_video(video_path: str, output_audio_path: str) -> int:
    """
    Extract audio from video file and save as 16kHz mono WAV.
    Returns the sample rate (always 16000 for Silero VAD compatibility).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_audio_path,
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract audio from {video_path}")

    return 16000


def get_speech_segments(
    audio_path: str, model, get_speech_timestamps, read_audio
) -> list:
    """
    Get speech timestamps using Silero VAD.
    Returns list of dicts with 'start' and 'end' keys in seconds.
    """
    wav = read_audio(audio_path, sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    )

    sample_rate = 16000
    segments = []
    for ts in speech_timestamps:
        segments.append(
            {"start": ts["start"] / sample_rate, "end": ts["end"] / sample_rate}
        )

    return segments


def find_silence_cut_points(
    speech_segments: list, min_silence_ms: int, audio_duration: float
) -> list:
    """
    Find cut points where silence duration meets the minimum requirement.
    Returns list of cut points (end of silence periods) in seconds.
    """
    min_silence_sec = min_silence_ms / 1000.0
    cut_points = []

    for i in range(len(speech_segments) - 1):
        curr_end = speech_segments[i]["end"]
        next_start = speech_segments[i + 1]["start"]
        silence_duration = next_start - curr_end

        if silence_duration >= min_silence_sec:
            cut_points.append(next_start)

    # Always include the end of the audio as the final cut point
    if speech_segments:
        last_speech_end = speech_segments[-1]["end"]
        trailing_silence = audio_duration - last_speech_end
        if trailing_silence >= min_silence_sec or not cut_points:
            cut_points.append(audio_duration)
        elif cut_points[-1] != audio_duration:
            cut_points.append(audio_duration)
    else:
        cut_points.append(audio_duration)

    return cut_points


def detect_face_in_video(video_path: str, detector, margin: int) -> dict | None:
    """
    Detect face in the middle frame of the video.
    Returns face bounding box dict or None if no face detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    boxes, _ = detector.detect(frame_rgb)
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

    # Make it square
    crop_w = x2 - x1
    crop_h = y2 - y1
    side = max(crop_w, crop_h)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    final_x1 = max(0, center_x - side // 2)
    final_y1 = max(0, center_y - side // 2)
    final_x2 = min(w, final_x1 + side)
    final_y2 = min(h, final_y1 + side)

    return {
        "x1": final_x1,
        "y1": final_y1,
        "x2": final_x2,
        "y2": final_y2,
        "frame_w": w,
        "frame_h": h,
    }


def extract_segment_with_face_crop(
    video_path: str,
    start_time: float,
    end_time: float,
    face_box: dict,
    output_video_path: str,
    output_audio_path: str,
    target_size: int,
) -> bool:
    """
    Extract a segment from the video with face cropping applied.
    Uses ffmpeg for efficient extraction with crop filter.
    """
    x1, y1, x2, y2 = (
        face_box["x1"],
        face_box["y1"],
        face_box["x2"],
        face_box["y2"],
    )
    crop_w = x2 - x1
    crop_h = y2 - y1

    # ffmpeg crop filter: crop=w:h:x:y
    crop_filter = (
        f"crop={crop_w}:{crop_h}:{x1}:{y1},scale={target_size}:{target_size}"
    )

    # Extract video segment with crop
    video_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-t",
        str(end_time - start_time),
        "-vf",
        crop_filter,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-preset",
        "fast",
        output_video_path,
    ]

    # Extract audio segment
    audio_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-t",
        str(end_time - start_time),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_audio_path,
    ]

    try:
        subprocess.run(video_cmd, check=True)
        subprocess.run(audio_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ ffmpeg error: {e}")
        return False


def process_video(
    video_path: str,
    output_dir: str,
    input_root: str,
    vad_model,
    get_speech_timestamps,
    read_audio,
    face_detector,
    silence_ms: int,
    face_size: int,
    margin: int,
    dataset_name: str,
) -> list:
    """
    Process a single video file.
    Returns list of metadata entries for the segments.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    input_root = Path(input_root)

    # Build prefix from relative path (includes parent folders)
    # e.g., CasualConversationsA/1223/1223_09.mp4 -> CasualConversationsA_1223_1223_09
    try:
        relative_path = video_path.relative_to(input_root)
        # Join all parts: parent folders + stem
        parts = list(relative_path.parent.parts) + [video_path.stem]
        video_name = "_".join(parts) if parts else video_path.stem
    except ValueError:
        # video_path is not relative to input_root (single file case)
        video_name = video_path.stem

    print(f"\n📹 Processing: {video_path.name} -> {video_name}")

    # Create output directories
    video_out_dir = output_dir / "video"
    audio_out_dir = output_dir / "audio"
    video_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Detect face in original video
    print("  ⏳ Detecting face...")
    face_box = detect_face_in_video(str(video_path), face_detector, margin)
    if face_box is None:
        print("  ⚠️ No face detected, skipping video")
        return []
    print(
        f"  ✓ Face detected: {face_box['x2'] - face_box['x1']}x{face_box['y2'] - face_box['y1']}px"
    )

    # Step 2: Extract audio for VAD
    print("  ⏳ Running VAD...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        extract_audio_from_video(str(video_path), tmp_audio_path)

        # Get audio duration
        info = torchaudio.info(tmp_audio_path)
        audio_duration = info.num_frames / info.sample_rate

        # Run VAD
        speech_segments = get_speech_segments(
            tmp_audio_path, vad_model, get_speech_timestamps, read_audio
        )

        if not speech_segments:
            print("  ⚠️ No speech detected, skipping video")
            return []

        # Find cut points
        cut_points = find_silence_cut_points(
            speech_segments, silence_ms, audio_duration
        )
        print(f"  ✓ Found {len(cut_points)} segment(s)")

    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

    # Step 3: Extract segments with face cropping
    entries = []
    prev_cut = 0.0
    failed_count = 0

    segment_progress = tqdm(
        enumerate(cut_points),
        total=len(cut_points),
        desc="  Extracting",
        unit="seg",
        leave=False,
        dynamic_ncols=True,
    )

    for idx, cut_point in segment_progress:
        start_time = prev_cut
        end_time = cut_point

        if end_time <= start_time:
            prev_cut = cut_point
            continue

        segment_name = f"{video_name}_segment_{idx:04d}_{start_time:.1f}s_to_{end_time:.1f}s"
        out_video_path = video_out_dir / f"{segment_name}.mp4"
        out_audio_path = audio_out_dir / f"{segment_name}.wav"

        segment_progress.set_postfix(
            {"segment": idx, "duration": f"{end_time - start_time:.1f}s"}
        )

        success = extract_segment_with_face_crop(
            str(video_path),
            start_time,
            end_time,
            face_box,
            str(out_video_path),
            str(out_audio_path),
            face_size,
        )

        if success:
            entry = {
                "file_name": f"audio/{segment_name}.wav",
                "id": str(uuid.uuid4()),
                "language": "eng",
                "endpoint_bool": None,  # To be manually labeled
                "midfiller": False,
                "endfiller": False,
                "synthetic": False,
                "spoken_text": None,
                "dataset": dataset_name,
                "video_path": f"video/{segment_name}.mp4",
                "visual_label": None,  # To be manually labeled
            }
            entries.append(entry)
        else:
            failed_count += 1

        prev_cut = cut_point

    if failed_count > 0:
        print(f"  ⚠️ {failed_count} segment(s) failed")
    print(f"  ✓ Extracted {len(entries)} segment(s)")

    return entries


def find_video_files(input_path: str) -> list:
    """
    Find all video files in the input path.
    Recursively searches nested directories.
    """
    input_path = Path(input_path)
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    if input_path.is_file():
        if input_path.suffix.lower() in video_extensions:
            return [input_path]
        else:
            print(f"⚠️ Not a video file: {input_path}")
            return []
    elif input_path.is_dir():
        videos = []
        for ext in video_extensions:
            # Use rglob for recursive search through nested directories
            videos.extend(input_path.rglob(f"*{ext}"))
            videos.extend(input_path.rglob(f"*{ext.upper()}"))
        return sorted(videos)
    else:
        print(f"⚠️ Path not found: {input_path}")
        return []


def load_existing_metadata(output_dir: str) -> list:
    """Load existing metadata entries if metadata.jsonl exists."""
    meta_path = Path(output_dir) / METADATA_FILE
    if not meta_path.exists():
        return []

    entries = []
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def save_metadata(output_dir: str, entries: list):
    """Save metadata entries to metadata.jsonl."""
    meta_path = Path(output_dir) / METADATA_FILE
    with open(meta_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare video dataset with face cropping and VAD segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single video
    python prepare_video_dataset.py video.mp4 output_dir/

    # Process directory of videos
    python prepare_video_dataset.py videos_dir/ output_dir/

    # Add more videos to existing dataset
    python prepare_video_dataset.py another_video.mp4 output_dir/

Output:
    output_dir/
    ├── video/           # Face-cropped video segments
    ├── audio/           # Audio segments (16kHz mono WAV)
    └── metadata.jsonl   # HF-compatible metadata (edit to add labels)

Manual Labeling:
    After processing, edit metadata.jsonl to set:
    - "endpoint_bool": true/false (is the turn complete?)
    - "visual_label": "expressive", "breath-in", etc. (optional visual cues)
        """,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input video file or directory of videos",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_MS,
        help=f"Minimum silence duration (ms) to trigger segment cut (default: {DEFAULT_SILENCE_MS})",
    )
    parser.add_argument(
        "--face-size",
        type=int,
        default=DEFAULT_FACE_SIZE,
        help=f"Output face crop size in pixels (default: {DEFAULT_FACE_SIZE})",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=DEFAULT_MARGIN,
        help=f"Margin around detected face in pixels (default: {DEFAULT_MARGIN})",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="CasualConversations_Video",
        help="Dataset name tag for metadata (default: CasualConversations_Video)",
    )

    args = parser.parse_args()

    # Validate
    check_ffmpeg()

    # Find videos
    videos = find_video_files(args.input_path)
    if not videos:
        print("❌ No video files found")
        sys.exit(1)

    print(f"📂 Found {len(videos)} video(s) to process")

    # Initialize models
    print("⏳ Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    face_detector = MTCNN(keep_all=False, select_largest=True, device=device)
    vad_model, get_speech_timestamps, read_audio = load_silero_vad()

    # Load existing metadata (for appending)
    existing_entries = load_existing_metadata(args.output_dir)
    if existing_entries:
        print(
            f"📋 Found {len(existing_entries)} existing entries in metadata.jsonl"
        )

    # Determine input root (for building relative paths in segment names)
    input_path = Path(args.input_path)
    if input_path.is_dir():
        input_root = input_path
    else:
        input_root = input_path.parent

    # Process videos with progress bar
    all_entries = existing_entries.copy()
    total_segments = 0

    video_progress = tqdm(
        videos,
        desc="Processing videos",
        unit="video",
        dynamic_ncols=True,
    )

    for video_path in video_progress:
        video_progress.set_postfix(
            {"current": video_path.name, "segments": total_segments}
        )
        new_entries = process_video(
            str(video_path),
            args.output_dir,
            str(input_root),
            vad_model,
            get_speech_timestamps,
            read_audio,
            face_detector,
            args.silence_ms,
            args.face_size,
            args.margin,
            args.dataset_name,
        )
        all_entries.extend(new_entries)
        total_segments += len(new_entries)

    # Save metadata
    save_metadata(args.output_dir, all_entries)

    # Summary
    new_count = len(all_entries) - len(existing_entries)
    print("\n✅ Done!")
    print(f"   New segments: {new_count}")
    print(f"   Total segments: {len(all_entries)}")
    print(f"   Output: {args.output_dir}")
    print(
        f"\n📝 Next step: Edit {args.output_dir}/metadata.jsonl to add labels:"
    )
    print('   - Set "endpoint_bool" to true or false')
    print('   - Optionally set "visual_label" for visual cues')


if __name__ == "__main__":
    main()
