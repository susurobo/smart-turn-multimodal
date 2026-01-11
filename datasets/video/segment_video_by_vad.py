#!/usr/bin/env python3
"""
Silero VAD-based Video Segmenter

This script segments a video file based on speech activity detection.
It creates video segments that end with silence of at least the specified duration.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import torch
import torchaudio
from moviepy.video.io.VideoFileClip import VideoFileClip


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
    print("⏳ Extracting audio from video...")

    # Use ffmpeg for reliable audio extraction at 16kHz mono
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # PCM 16-bit
        "-ar",
        "16000",  # 16kHz sample rate (required by Silero VAD)
        "-ac",
        "1",  # Mono
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
    print("⏳ Running Silero VAD...")

    wav = read_audio(audio_path, sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        threshold=0.5,  # Speech probability threshold
        min_speech_duration_ms=250,  # Minimum speech duration
        min_silence_duration_ms=100,  # Minimum silence duration for splitting
        speech_pad_ms=30,  # Padding around speech
    )

    # Convert sample indices to seconds
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
    Each cut point marks where a segment should end.
    """
    min_silence_sec = min_silence_ms / 1000.0
    cut_points = []

    for i in range(len(speech_segments) - 1):
        curr_end = speech_segments[i]["end"]
        next_start = speech_segments[i + 1]["start"]
        silence_duration = next_start - curr_end

        if silence_duration >= min_silence_sec:
            # Cut at the end of silence (just before next speech starts)
            cut_points.append(next_start)

    # Always include the end of the audio as the final cut point
    if speech_segments:
        last_speech_end = speech_segments[-1]["end"]
        # Check if there's sufficient silence at the end
        trailing_silence = audio_duration - last_speech_end
        if trailing_silence >= min_silence_sec or not cut_points:
            cut_points.append(audio_duration)
        elif cut_points[-1] != audio_duration:
            # Include final segment even if trailing silence is short
            cut_points.append(audio_duration)
    else:
        # No speech detected, just return the full duration
        cut_points.append(audio_duration)

    return cut_points


def extract_video_segments(video_path: str, cut_points: list, output_dir: str):
    """
    Extract video segments based on cut points.

    Each segment runs from the previous cut point (or 0) to the current cut point.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"⏳ Extracting {len(cut_points)} segment(s)...")

    video = VideoFileClip(video_path)
    video_duration = video.duration

    prev_cut = 0.0
    for idx, cut_point in enumerate(cut_points):
        start_time = prev_cut
        end_time = min(cut_point, video_duration)

        if end_time <= start_time:
            continue

        segment_name = f"segment_{idx:04d}_{start_time:.1f}s_to_{end_time:.1f}s"
        out_video_path = os.path.join(output_dir, f"{segment_name}.mp4")
        out_audio_path = os.path.join(output_dir, f"{segment_name}.wav")

        try:
            clip = video.subclipped(start_time, end_time)

            # Write video
            clip.write_videofile(
                out_video_path, codec="libx264", audio_codec="aac", logger=None
            )

            # Write audio
            if clip.audio is not None:
                clip.audio.write_audiofile(out_audio_path, logger=None)

            print(
                f"  [{idx + 1}/{len(cut_points)}] Saved: {segment_name} ({end_time - start_time:.1f}s)"
            )

        except Exception as e:
            print(f"  ⚠️ Failed to process segment {idx}: {e}")

        prev_cut = cut_point

    video.close()


def segment_video_by_silence(
    video_path: str, output_dir: str, min_silence_ms: int = 500
):
    """
    Main function to segment a video file based on silence detection.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the output segments
        min_silence_ms: Minimum silence duration (in milliseconds) to trigger a cut
    """
    check_ffmpeg()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load Silero VAD model
    model, get_speech_timestamps, read_audio = load_silero_vad()

    # Extract audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        extract_audio_from_video(video_path, tmp_audio_path)

        # Get audio duration
        info = torchaudio.info(tmp_audio_path)
        audio_duration = info.num_frames / info.sample_rate
        print(f"📊 Audio duration: {audio_duration:.1f}s")

        # Run VAD
        speech_segments = get_speech_segments(
            tmp_audio_path, model, get_speech_timestamps, read_audio
        )
        print(f"📊 Found {len(speech_segments)} speech segments")

        if not speech_segments:
            print("⚠️ No speech detected in the video.")
            return

        # Find cut points
        cut_points = find_silence_cut_points(
            speech_segments, min_silence_ms, audio_duration
        )
        print(
            f"📊 Found {len(cut_points)} cut point(s) with silence >= {min_silence_ms}ms"
        )

        # Extract segments
        extract_video_segments(video_path, cut_points, output_dir)

        print("✅ Done!")

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


def main():
    parser = argparse.ArgumentParser(
        description="Segment a video file based on speech silence detection using Silero VAD."
    )
    parser.add_argument(
        "video_path", type=str, help="Path to the input video file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the output video segments",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=500,
        help="Minimum silence duration in milliseconds to trigger a segment cut (default: 500)",
    )

    args = parser.parse_args()

    segment_video_by_silence(
        video_path=args.video_path,
        output_dir=args.output_dir,
        min_silence_ms=args.silence_ms,
    )


if __name__ == "__main__":
    main()
