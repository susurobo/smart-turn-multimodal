#!/usr/bin/env python3
"""
CLI tool to compare two classifier results (e.g., multimodal vs audio-only).

Merges two CSV benchmark files into a single JSONL for comparison visualization.

Usage:
    python compare_classifiers.py \
        --multimodal path/to/multimodal_results.csv \
        --audio-only path/to/audio_only_results.csv \
        --output comparison.jsonl \
        --threshold 0.5

Example:
    python compare_classifiers.py \
        --multimodal benchmark/mm_run_20260113_2153/multimodal_perf_benchmark_*.csv \
        --audio-only benchmark/model/sm_audio_only_benchmark_*.csv \
        --output benchmark/comparison_mm_vs_audio.jsonl
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict


def load_csv(filepath: str) -> dict:
    """Load CSV and return dict keyed by 'id' field."""
    data = {}
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id", "")
            if row_id:
                data[row_id] = row
    return data


def parse_float(value: str, default: float = 0.0) -> float:
    """Safely parse a float value."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_int(value: str, default: int = 0) -> int:
    """Safely parse an int value."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_bool(value: str, default: bool = False) -> bool:
    """Safely parse a boolean value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return default


def merge_results(
    multimodal_csv: str,
    audio_only_csv: str,
    threshold: float = 0.5,
    video_base_path: str = None,
) -> list:
    """
    Merge multimodal and audio-only CSV results into comparison records.

    Args:
        multimodal_csv: Path to multimodal results CSV
        audio_only_csv: Path to audio-only results CSV
        threshold: Classification threshold (default 0.5)
        video_base_path: Optional base path for videos

    Returns:
        List of merged comparison records
    """
    print(f"Loading multimodal results from: {multimodal_csv}")
    mm_data = load_csv(multimodal_csv)
    print(f"  Loaded {len(mm_data)} records")

    print(f"Loading audio-only results from: {audio_only_csv}")
    ao_data = load_csv(audio_only_csv)
    print(f"  Loaded {len(ao_data)} records")

    # Find common IDs
    common_ids = set(mm_data.keys()) & set(ao_data.keys())
    mm_only = set(mm_data.keys()) - set(ao_data.keys())
    ao_only = set(ao_data.keys()) - set(mm_data.keys())

    print(f"\nMatching records: {len(common_ids)}")
    if mm_only:
        print(f"  Multimodal-only records (skipped): {len(mm_only)}")
    if ao_only:
        print(f"  Audio-only-only records (skipped): {len(ao_only)}")

    # Determine threshold column suffix
    thresh_key = f"pred_{threshold}"
    correct_key = f"correct_{threshold}"

    # Merge records
    results = []
    for record_id in sorted(common_ids):
        mm_row = mm_data[record_id]
        ao_row = ao_data[record_id]

        # Get ground truth (should be same in both)
        true_label = parse_int(
            mm_row.get("true_label", ao_row.get("true_label", 0))
        )

        # Extract multimodal predictions
        mm_prob = parse_float(mm_row.get("prob", 0))
        mm_pred = parse_int(mm_row.get(thresh_key, mm_row.get("pred_0.5", 0)))
        mm_correct = parse_int(
            mm_row.get(correct_key, mm_row.get("correct_0.5", 0))
        )

        # Extract audio-only predictions
        ao_prob = parse_float(ao_row.get("prob", 0))
        ao_pred = parse_int(ao_row.get(thresh_key, ao_row.get("pred_0.5", 0)))
        ao_correct = parse_int(
            ao_row.get(correct_key, ao_row.get("correct_0.5", 0))
        )

        # Get video/audio paths
        video_path = mm_row.get("video_path", "")
        audio_path = mm_row.get(
            "audio_path", ao_row.get("audio_path", record_id)
        )
        audio_duration = parse_float(mm_row.get("audio_duration_sec", 0))
        has_video = parse_bool(mm_row.get("has_video", True))

        # Optional metadata
        language = ao_row.get("language", "")
        dataset = ao_row.get("dataset", "")

        # Compute comparison metrics
        agrees = mm_pred == ao_pred
        both_correct = bool(mm_correct) and bool(ao_correct)
        both_wrong = not bool(mm_correct) and not bool(ao_correct)
        mm_better = bool(mm_correct) and not bool(ao_correct)
        ao_better = not bool(mm_correct) and bool(ao_correct)

        # Determine disagreement type
        if agrees:
            disagree_type = None
        elif mm_pred == 1 and ao_pred == 0:
            disagree_type = "mm_positive"  # MM says end-of-turn, AO says not
        else:
            disagree_type = "ao_positive"  # AO says end-of-turn, MM says not

        record = {
            "id": record_id,
            "true_label": true_label,
            "video_path": video_path,
            "audio_path": audio_path,
            "audio_duration_sec": audio_duration,
            "has_video": has_video,
            "language": language,
            "dataset": dataset,
            "multimodal": {
                "prob": round(mm_prob, 6),
                "pred": mm_pred,
                "correct": bool(mm_correct),
            },
            "audio_only": {
                "prob": round(ao_prob, 6),
                "pred": ao_pred,
                "correct": bool(ao_correct),
            },
            "threshold": threshold,
            "agrees": agrees,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "mm_better": mm_better,
            "ao_better": ao_better,
            "disagree_type": disagree_type,
        }

        results.append(record)

    return results


def compute_summary(results: list) -> dict:
    """Compute summary statistics from comparison results."""
    total = len(results)
    if total == 0:
        return {}

    agrees = sum(1 for r in results if r["agrees"])
    disagrees = total - agrees
    both_correct = sum(1 for r in results if r["both_correct"])
    both_wrong = sum(1 for r in results if r["both_wrong"])
    mm_better = sum(1 for r in results if r["mm_better"])
    ao_better = sum(1 for r in results if r["ao_better"])

    mm_correct_total = sum(1 for r in results if r["multimodal"]["correct"])
    ao_correct_total = sum(1 for r in results if r["audio_only"]["correct"])

    return {
        "total_records": total,
        "agrees": agrees,
        "agrees_pct": round(100 * agrees / total, 2),
        "disagrees": disagrees,
        "disagrees_pct": round(100 * disagrees / total, 2),
        "both_correct": both_correct,
        "both_correct_pct": round(100 * both_correct / total, 2),
        "both_wrong": both_wrong,
        "both_wrong_pct": round(100 * both_wrong / total, 2),
        "mm_better": mm_better,
        "mm_better_pct": round(100 * mm_better / total, 2),
        "ao_better": ao_better,
        "ao_better_pct": round(100 * ao_better / total, 2),
        "mm_accuracy": round(100 * mm_correct_total / total, 2),
        "ao_accuracy": round(100 * ao_correct_total / total, 2),
    }


def save_jsonl(results: list, output_path: str):
    """Save results to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two classifier results (multimodal vs audio-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_classifiers.py \\
      --multimodal results_multimodal.csv \\
      --audio-only results_audio.csv \\
      --output comparison.jsonl

  # With custom threshold
  python compare_classifiers.py \\
      --multimodal results_mm.csv \\
      --audio-only results_ao.csv \\
      --output comparison.jsonl \\
      --threshold 0.9
        """,
    )
    parser.add_argument(
        "--multimodal",
        "-m",
        type=str,
        required=True,
        help="Path to multimodal results CSV",
    )
    parser.add_argument(
        "--audio-only",
        "-a",
        type=str,
        required=True,
        help="Path to audio-only results CSV",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for comparison JSONL",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--video-base",
        type=str,
        default=None,
        help="Base path for video files (optional)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.multimodal).exists():
        print(f"❌ Multimodal CSV not found: {args.multimodal}")
        sys.exit(1)
    if not Path(args.audio_only).exists():
        print(f"❌ Audio-only CSV not found: {args.audio_only}")
        sys.exit(1)

    # Merge results
    results = merge_results(
        args.multimodal,
        args.audio_only,
        threshold=args.threshold,
        video_base_path=args.video_base,
    )

    if not results:
        print("❌ No matching records found between the two CSVs")
        sys.exit(1)

    # Compute and print summary
    summary = compute_summary(results)
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Total records:     {summary['total_records']}")
    print(f"Threshold:         {args.threshold}")
    print()
    print(f"Multimodal acc:    {summary['mm_accuracy']}%")
    print(f"Audio-only acc:    {summary['ao_accuracy']}%")
    print()
    print(f"Classifiers agree: {summary['agrees']} ({summary['agrees_pct']}%)")
    print(
        f"Classifiers differ:{summary['disagrees']} ({summary['disagrees_pct']}%)"
    )
    print()
    print(
        f"Both correct:      {summary['both_correct']} ({summary['both_correct_pct']}%)"
    )
    print(
        f"Both wrong:        {summary['both_wrong']} ({summary['both_wrong_pct']}%)"
    )
    print(
        f"MM better:         {summary['mm_better']} ({summary['mm_better_pct']}%)"
    )
    print(
        f"Audio-only better: {summary['ao_better']} ({summary['ao_better_pct']}%)"
    )
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, str(output_path))
    print(f"\n✅ Saved {len(results)} records to: {output_path}")

    # Also save summary as JSON
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary saved to: {summary_path}")

    print(f"\n🚀 Run visualizer with:")
    print(f"   python visualize_comparison.py {output_path}")


if __name__ == "__main__":
    main()
