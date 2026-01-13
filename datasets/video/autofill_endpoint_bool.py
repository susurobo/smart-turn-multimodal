#!/usr/bin/env python3
"""
Auto-fill endpoint_bool in metadata.jsonl based on segment position within each batch.

Rules (only applied when endpoint_bool is null):
- Indexes 0 and 1: Leave as null
- Indexes 2 to N-2 (inclusive): Set to false
- Index N-1 (last): Set to true
- Batches with fewer than 3 segments are ignored

Optional: Exclude specific videos based on transcription text patterns.
When --transcription-file is provided, videos containing the --exclude-text pattern
will have all their segments set to endpoint_bool=null (excluded from training).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_batch_key_and_segment_index(
    file_name: str,
) -> tuple[str, int] | None:
    """
    Extract the batch key (part before '_segment_') and segment index from file name.

    Example: 'audio/CasualConversationsA_1140_1140_00_segment_0000_0.0s_to_4.8s.wav'
    Returns: ('CasualConversationsA_1140_1140_00', 0)
    """
    # Remove the 'audio/' prefix if present
    base_name = file_name.replace("audio/", "").replace("video/", "")

    # Match pattern: {batch_key}_segment_{index}_{rest}
    match = re.match(r"^(.+?)_segment_(\d+)_.*", base_name)
    if match:
        batch_key = match.group(1)
        segment_index = int(match.group(2))
        return batch_key, segment_index
    return None


def video_path_to_core_key(video_path: str) -> str:
    """
    Extract the core identifying part from a video path (folder/video ID).

    Example: 'CasualConversationsA/1149/1149_08.MP4' -> '1149_1149_08'

    This ignores the prefix (CasualConversationsA, CasualConversationsH, etc.)
    to allow matching against metadata files that may have modified prefixes
    like 'CasualConversationsA 2'.
    """
    # Remove extension and get just the path parts
    path = video_path.replace(".MP4", "").replace(".mp4", "")
    parts = path.split("/")

    if len(parts) >= 3:
        # e.g., ['CasualConversationsA', '1149', '1149_08']
        # Return: '1149_1149_08'
        folder = parts[-2]
        video_id = parts[-1]
        return f"{folder}_{video_id}"
    elif len(parts) == 2:
        # e.g., ['1149', '1149_08']
        return f"{parts[0]}_{parts[1]}"
    else:
        return parts[-1]


def count_transcription_words(transcription: str) -> int:
    """
    Count actual spoken words in a transcription, ignoring annotation tags.

    Strips out:
    - Tags like <no-speech>, <spk_noise>, <colloquial>...</colloquial>, <noise>, etc.
    - Timing markers like [secondary_1.668_secondary], [/primary_75.138_primary/], [0.000], etc.
    - Special characters like %
    """
    text = transcription

    # Remove bracketed timing/speaker markers: [secondary_...], [/primary_.../], [0.000], etc.
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove XML-style tags: <no-speech>, <spk_noise>, </colloquial>, <overlap>, etc.
    text = re.sub(r"<[^>]*>", " ", text)

    # Remove special characters
    text = text.replace("%", " ").replace("=", " ")

    # Split and count non-empty words
    words = [w for w in text.split() if w.strip()]
    return len(words)


def load_excluded_video_keys(
    transcription_file: Path,
    exclude_texts: list[str] | None,
    min_word_count: int,
) -> tuple[set[str], set[str]]:
    """
    Load the transcription file and extract core keys for videos to exclude.

    Returns:
        - excluded_by_text: set of core keys for videos containing any of exclude_texts
        - excluded_by_word_count: set of core keys for videos with fewer than min_word_count words
    """
    excluded_by_text = set()
    excluded_by_word_count = set()

    with open(transcription_file, "r") as f:
        transcriptions = json.load(f)

    # Prepare lowercase patterns for matching
    exclude_patterns = (
        [t.lower() for t in exclude_texts] if exclude_texts else []
    )

    for entry in transcriptions:
        transcription = entry.get("transcription", "")
        video_path = entry.get("video_path", "")
        core_key = video_path_to_core_key(video_path)

        # Check for any exclude text pattern
        transcription_lower = transcription.lower()
        for pattern in exclude_patterns:
            if pattern in transcription_lower:
                excluded_by_text.add(core_key)
                break  # No need to check other patterns once matched

        # Check word count
        word_count = count_transcription_words(transcription)
        if word_count < min_word_count:
            excluded_by_word_count.add(core_key)

    return excluded_by_text, excluded_by_word_count


def batch_key_matches_excluded(batch_key: str, excluded_keys: set[str]) -> bool:
    """
    Check if a batch key matches any of the excluded core keys.

    batch_key: e.g., 'CasualConversationsA_1149_1149_08' or 'CasualConversationsA 2_1149_1149_08'
    excluded_keys: e.g., {'1149_1149_08', '1275_1275_07'}

    We check if the batch_key ends with any of the excluded core keys
    (after the prefix which may vary).
    """
    for core_key in excluded_keys:
        # The batch_key should end with the core_key pattern
        # e.g., 'CasualConversationsA_1149_1149_08' ends with '1149_1149_08'
        # or 'CasualConversationsA 2_1149_1149_08' ends with '1149_1149_08'
        if batch_key.endswith(core_key):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Auto-fill endpoint_bool in metadata.jsonl based on segment position within each batch."
    )
    parser.add_argument(
        "metadata_file",
        type=Path,
        help="Path to the metadata.jsonl file to modify",
    )
    parser.add_argument(
        "--transcription-file",
        type=Path,
        default=None,
        help="Path to transcription JSON file (e.g., CasualConversations_transcriptions.json). "
        "Used to exclude videos based on transcription text.",
    )
    parser.add_argument(
        "--exclude-text",
        type=str,
        nargs="*",
        default=["back to neutral"],
        help="Text pattern(s) to search for in transcriptions. Videos containing any of these texts "
        "will have all segments set to endpoint_bool=null (excluded from training). "
        "Default: 'back to neutral'. Pass no arguments to disable.",
    )
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=5,
        help="Minimum number of spoken words required in a video transcription. "
        "Videos with fewer words will have all segments set to endpoint_bool=null. "
        "Default: 5. Set to 0 to disable.",
    )
    args = parser.parse_args()

    metadata_path = args.metadata_file

    # Load excluded video keys if transcription file is provided
    excluded_keys: set[str] = set()
    if args.transcription_file:
        if not args.transcription_file.exists():
            print(
                f"Error: Transcription file not found: {args.transcription_file}"
            )
            return

        exclude_texts = args.exclude_text if args.exclude_text else None
        excluded_by_text, excluded_by_word_count = load_excluded_video_keys(
            args.transcription_file, exclude_texts, args.min_word_count
        )

        if exclude_texts:
            patterns_str = "', '".join(args.exclude_text)
            print(
                f"Found {len(excluded_by_text)} videos to exclude (containing any of: '{patterns_str}')"
            )
            if excluded_by_text:
                print(f"  Sample: {list(excluded_by_text)[:5]}")

        if args.min_word_count > 0:
            print(
                f"Found {len(excluded_by_word_count)} videos to exclude (fewer than {args.min_word_count} words)"
            )
            if excluded_by_word_count:
                print(f"  Sample: {list(excluded_by_word_count)[:5]}")

        # Combine both exclusion sets
        excluded_keys = excluded_by_text | excluded_by_word_count
        print(f"Total unique videos to exclude: {len(excluded_keys)}")

    # Read all entries
    entries = []
    with open(metadata_path, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    print(f"Loaded {len(entries)} entries")

    # Group entries by batch key
    batches = defaultdict(list)
    for idx, entry in enumerate(entries):
        result = extract_batch_key_and_segment_index(entry["file_name"])
        if result:
            batch_key, segment_index = result
            batches[batch_key].append((idx, segment_index, entry))

    print(f"Found {len(batches)} batches")

    # Process each batch
    modified_count = 0
    skipped_batches = 0

    for batch_key, batch_entries in batches.items():
        # Sort by segment index
        batch_entries.sort(key=lambda x: x[1])
        n = len(batch_entries)

        # Skip batches with fewer than 3 segments
        if n < 3:
            skipped_batches += 1
            continue

        for i, (entry_idx, segment_index, entry) in enumerate(batch_entries):
            # Only modify if endpoint_bool is null
            if entry["endpoint_bool"] is not None:
                continue

            # Apply rules based on position in batch
            if i == 0 or i == 1:
                # Indexes 0 and 1: leave as null
                pass
            elif i == n - 1:
                # Last index: set to true
                entries[entry_idx]["endpoint_bool"] = True
                modified_count += 1
            else:
                # Indexes 2 to N-2: set to false
                entries[entry_idx]["endpoint_bool"] = False
                modified_count += 1

    print(f"Skipped {skipped_batches} batches with fewer than 3 segments")
    print(f"Modified {modified_count} entries")

    # Final step: nullify endpoint_bool for excluded videos (based on transcription text)
    excluded_count = 0
    excluded_batches = 0
    if excluded_keys:
        print("\nApplying exclusions based on transcription text...")
        for batch_key, batch_entries in batches.items():
            if batch_key_matches_excluded(batch_key, excluded_keys):
                excluded_batches += 1
                for entry_idx, segment_index, entry in batch_entries:
                    # Set endpoint_bool to null for ALL segments to exclude from training
                    entries[entry_idx]["endpoint_bool"] = None
                    excluded_count += 1

        print(
            f"Excluded {excluded_batches} videos ({excluded_count} segments, all set to endpoint_bool=null)"
        )

    # Write back
    with open(metadata_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved to {metadata_path}")


if __name__ == "__main__":
    main()
