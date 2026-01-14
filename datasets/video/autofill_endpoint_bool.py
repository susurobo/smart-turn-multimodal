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
from datetime import datetime
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
    Convert a video path to a core key that preserves the prefix.

    Example: 'CasualConversationsA/1149/1149_08.MP4' -> 'CasualConversationsA_1149_1149_08'

    The prefix (CasualConversationsA, CasualConversationsH, etc.) is preserved
    to distinguish between different datasets with the same folder/video structure.
    """
    # Remove extension and get just the path parts
    path = video_path.replace(".MP4", "").replace(".mp4", "")
    parts = path.split("/")

    if len(parts) >= 3:
        # e.g., ['CasualConversationsA', '1149', '1149_08']
        # Return: 'CasualConversationsA_1149_1149_08'
        prefix = parts[0]
        folder = parts[-2]
        video_id = parts[-1]
        return f"{prefix}_{folder}_{video_id}"
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


def load_transcription_data(
    transcription_file: Path,
    exclude_texts: list[str] | None,
    min_word_count: int,
) -> tuple[set[str], set[str], dict[str, int], dict[str, str]]:
    """
    Load the transcription file and extract data for exclusion and logging.

    Returns:
        - excluded_by_text: set of core keys for videos containing any of exclude_texts
        - excluded_by_word_count: set of core keys for videos with fewer than min_word_count words
        - durations: dict mapping core_key -> duration_ms
        - exclusion_reasons: dict mapping core_key -> reason string
    """
    excluded_by_text = set()
    excluded_by_word_count = set()
    durations: dict[str, int] = {}
    exclusion_reasons: dict[str, str] = {}

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
        duration_ms = int(entry.get("duration_ms", 0))

        # Store duration
        durations[core_key] = duration_ms

        # Check for any exclude text pattern
        transcription_lower = transcription.lower()
        matched_pattern = None
        for pattern in exclude_patterns:
            if pattern in transcription_lower:
                excluded_by_text.add(core_key)
                matched_pattern = pattern
                break  # No need to check other patterns once matched

        # Check word count
        word_count = count_transcription_words(transcription)
        if word_count < min_word_count:
            excluded_by_word_count.add(core_key)

        # Build exclusion reason
        reasons = []
        if matched_pattern:
            reasons.append(f"contains '{matched_pattern}'")
        if word_count < min_word_count:
            reasons.append(f"word count {word_count} < {min_word_count}")
        if reasons:
            exclusion_reasons[core_key] = "; ".join(reasons)

    return (
        excluded_by_text,
        excluded_by_word_count,
        durations,
        exclusion_reasons,
    )


def batch_key_matches_excluded(batch_key: str, excluded_keys: set[str]) -> bool:
    """
    Check if a batch key matches any of the excluded core keys.

    batch_key: e.g., 'CasualConversationsA_1149_1149_08' or 'CasualConversationsA 2_1149_1149_08'
    excluded_keys: e.g., {'CasualConversationsA_1149_1149_08', 'CasualConversationsH_1275_1275_07'}

    Handles prefix variations where 'CasualConversationsA' in transcriptions
    may become 'CasualConversationsA 2', 'CasualConversationsA 3', etc. in metadata.
    """
    for core_key in excluded_keys:
        # Exact match
        if batch_key == core_key:
            return True

        # Check for suffix variations: 'CasualConversationsA_...' matches 'CasualConversationsA 2_...'
        # Find the first underscore in the core_key to split prefix from rest
        first_underscore = core_key.find("_")
        if first_underscore > 0:
            prefix = core_key[:first_underscore]  # e.g., 'CasualConversationsA'
            rest = core_key[first_underscore:]  # e.g., '_1149_1149_08'

            # Check if batch_key matches pattern: {prefix} {N}{rest}
            # e.g., 'CasualConversationsA 2_1149_1149_08' matches 'CasualConversationsA_1149_1149_08'
            if batch_key.startswith(prefix + " ") and batch_key.endswith(rest):
                # Verify the middle part is just a number
                middle = batch_key[len(prefix) + 1 : -len(rest)]
                if middle.isdigit():
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
    durations: dict[str, int] = {}
    exclusion_reasons: dict[str, str] = {}

    if args.transcription_file:
        if not args.transcription_file.exists():
            print(
                f"Error: Transcription file not found: {args.transcription_file}"
            )
            return

        exclude_texts = args.exclude_text if args.exclude_text else None
        (
            excluded_by_text,
            excluded_by_word_count,
            durations,
            exclusion_reasons,
        ) = load_transcription_data(
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
    excluded_batch_list: list[
        tuple[str, str, int, int]
    ] = []  # (batch_key, core_key, duration, segment_count)
    included_batch_list: list[
        tuple[str, str, int, int]
    ] = []  # (batch_key, core_key, duration, segment_count)
    short_included_batch_names: list[
        str
    ] = []  # Included batches with < 3 segments (skipped by autofill)

    if excluded_keys:
        print("\nApplying exclusions based on transcription text...")
        for batch_key, batch_entries in batches.items():
            # Find matching core key
            matched_core_key = None
            for core_key in excluded_keys:
                if batch_key == core_key:
                    matched_core_key = core_key
                    break
                # Check suffix variations
                first_underscore = core_key.find("_")
                if first_underscore > 0:
                    prefix = core_key[:first_underscore]
                    rest = core_key[first_underscore:]
                    if batch_key.startswith(
                        prefix + " "
                    ) and batch_key.endswith(rest):
                        middle = batch_key[len(prefix) + 1 : -len(rest)]
                        if middle.isdigit():
                            matched_core_key = core_key
                            break

            if matched_core_key:
                duration = durations.get(matched_core_key, 0)
                segment_count = len(batch_entries)
                excluded_batch_list.append(
                    (batch_key, matched_core_key, duration, segment_count)
                )
                for entry_idx, segment_index, entry in batch_entries:
                    # Set endpoint_bool to null for ALL segments to exclude from training
                    entries[entry_idx]["endpoint_bool"] = None
                    excluded_count += 1
            else:
                # Try to find duration for included batches
                # Match batch_key to a core_key in durations
                segment_count = len(batch_entries)
                if segment_count < 3:
                    short_included_batch_names.append(batch_key)
                for core_key, dur in durations.items():
                    if batch_key == core_key:
                        included_batch_list.append(
                            (batch_key, core_key, dur, segment_count)
                        )
                        break
                    first_underscore = core_key.find("_")
                    if first_underscore > 0:
                        prefix = core_key[:first_underscore]
                        rest = core_key[first_underscore:]
                        if batch_key.startswith(
                            prefix + " "
                        ) and batch_key.endswith(rest):
                            middle = batch_key[len(prefix) + 1 : -len(rest)]
                            if middle.isdigit():
                                included_batch_list.append(
                                    (batch_key, core_key, dur, segment_count)
                                )
                                break

        print(
            f"Excluded {len(excluded_batch_list)} videos ({excluded_count} segments, all set to endpoint_bool=null)"
        )
    else:
        # No exclusions - all batches with matching durations are included
        for batch_key, batch_entries in batches.items():
            segment_count = len(batch_entries)
            if segment_count < 3:
                short_included_batch_names.append(batch_key)
            for core_key, dur in durations.items():
                if batch_key == core_key:
                    included_batch_list.append(
                        (batch_key, core_key, dur, segment_count)
                    )
                    break
                first_underscore = core_key.find("_")
                if first_underscore > 0:
                    prefix = core_key[:first_underscore]
                    rest = core_key[first_underscore:]
                    if batch_key.startswith(
                        prefix + " "
                    ) and batch_key.endswith(rest):
                        middle = batch_key[len(prefix) + 1 : -len(rest)]
                        if middle.isdigit():
                            included_batch_list.append(
                                (batch_key, core_key, dur, segment_count)
                            )
                            break

    # Write back
    with open(metadata_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved to {metadata_path}")

    # Generate log file if transcription file was provided
    if args.transcription_file and (excluded_batch_list or included_batch_list):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = metadata_path.parent / f"autofill_log_{timestamp}.md"

        # Calculate durations and segment counts
        excluded_duration_ms = sum(d for _, _, d, _ in excluded_batch_list)
        included_duration_ms = sum(d for _, _, d, _ in included_batch_list)
        excluded_segments = sum(s for _, _, _, s in excluded_batch_list)
        included_segments = sum(s for _, _, _, s in included_batch_list)

        def ms_to_human(ms: int) -> str:
            seconds = ms / 1000
            minutes = seconds / 60
            hours = minutes / 60
            if hours >= 1:
                return f"{hours:.2f} hours"
            elif minutes >= 1:
                return f"{minutes:.2f} minutes"
            else:
                return f"{seconds:.2f} seconds"

        with open(log_path, "w") as f:
            f.write("# Autofill Endpoint Bool Log\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Metadata file:** `{metadata_path}`\n\n")
            f.write(f"**Transcription file:** `{args.transcription_file}`\n\n")

            # Exclusion settings
            f.write("## Exclusion Settings\n\n")
            if args.exclude_text:
                f.write(f"- **Exclude text patterns:** {args.exclude_text}\n")
            f.write(f"- **Minimum word count:** {args.min_word_count}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write("| Category | Batches | Segments | Duration |\n")
            f.write("|----------|---------|----------|----------|\n")
            f.write(
                f"| Excluded | {len(excluded_batch_list)} | {excluded_segments:,} | {ms_to_human(excluded_duration_ms)} ({excluded_duration_ms:,} ms) |\n"
            )
            f.write(
                f"| Included | {len(included_batch_list)} | {included_segments:,} | {ms_to_human(included_duration_ms)} ({included_duration_ms:,} ms) |\n"
            )
            total_duration = excluded_duration_ms + included_duration_ms
            total_batches = len(excluded_batch_list) + len(included_batch_list)
            total_segments = excluded_segments + included_segments
            f.write(
                f"| **Total** | {total_batches} | {total_segments:,} | {ms_to_human(total_duration)} ({total_duration:,} ms) |\n\n"
            )

            if short_included_batch_names:
                f.write(
                    f"**Note:** {len(short_included_batch_names)} included batch(es) have fewer than 3 segments "
                    f"and were skipped by autofill (no `true` label assigned):\n"
                )
                for batch_name in sorted(short_included_batch_names):
                    f.write(f"- `{batch_name}`\n")
                f.write("\n")

            # Final endpoint_bool distribution
            true_count = sum(
                1 for e in entries if e.get("endpoint_bool") is True
            )
            false_count = sum(
                1 for e in entries if e.get("endpoint_bool") is False
            )
            null_count = sum(
                1 for e in entries if e.get("endpoint_bool") is None
            )

            f.write("## Final Label Distribution\n\n")
            f.write("| endpoint_bool | Count | Percentage |\n")
            f.write("|---------------|-------|------------|\n")
            total_entries = len(entries)
            f.write(
                f"| `true` | {true_count:,} | {100 * true_count / total_entries:.1f}% |\n"
            )
            f.write(
                f"| `false` | {false_count:,} | {100 * false_count / total_entries:.1f}% |\n"
            )
            f.write(
                f"| `null` | {null_count:,} | {100 * null_count / total_entries:.1f}% |\n"
            )
            f.write(f"| **Total** | {total_entries:,} | 100% |\n\n")

            # Excluded batches list
            if excluded_batch_list:
                f.write("## Excluded Batches\n\n")
                f.write("| Batch Key | Segments | Duration (ms) | Reason |\n")
                f.write("|-----------|----------|---------------|--------|\n")
                for batch_key, core_key, duration, seg_count in sorted(
                    excluded_batch_list
                ):
                    reason = exclusion_reasons.get(core_key, "unknown")
                    f.write(
                        f"| {batch_key} | {seg_count} | {duration:,} | {reason} |\n"
                    )
                f.write("\n")

        print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
