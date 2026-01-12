#!/usr/bin/env python3
"""
Auto-fill endpoint_bool in metadata.jsonl based on segment position within each batch.

Rules (only applied when endpoint_bool is null):
- Indexes 0 and 1: Leave as null
- Indexes 2 to N-2 (inclusive): Set to false
- Index N-1 (last): Set to true
- Batches with fewer than 3 segments are ignored
"""

import json
import re
from collections import defaultdict
from pathlib import Path


def extract_batch_key_and_segment_index(file_name: str) -> tuple[str, int] | None:
    """
    Extract the batch key (part before '_segment_') and segment index from file name.
    
    Example: 'audio/CasualConversationsA_1140_1140_00_segment_0000_0.0s_to_4.8s.wav'
    Returns: ('CasualConversationsA_1140_1140_00', 0)
    """
    # Remove the 'audio/' prefix if present
    base_name = file_name.replace('audio/', '').replace('video/', '')
    
    # Match pattern: {batch_key}_segment_{index}_{rest}
    match = re.match(r'^(.+?)_segment_(\d+)_.*', base_name)
    if match:
        batch_key = match.group(1)
        segment_index = int(match.group(2))
        return batch_key, segment_index
    return None


def main():
    metadata_path = Path(__file__).parent / 'smart_turn_multimodal_dataset_v2' / 'metadata.jsonl'
    
    # Read all entries
    entries = []
    with open(metadata_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    print(f"Loaded {len(entries)} entries")
    
    # Group entries by batch key
    batches = defaultdict(list)
    for idx, entry in enumerate(entries):
        result = extract_batch_key_and_segment_index(entry['file_name'])
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
            if entry['endpoint_bool'] is not None:
                continue
            
            # Apply rules based on position in batch
            if i == 0 or i == 1:
                # Indexes 0 and 1: leave as null
                pass
            elif i == n - 1:
                # Last index: set to true
                entries[entry_idx]['endpoint_bool'] = True
                modified_count += 1
            else:
                # Indexes 2 to N-2: set to false
                entries[entry_idx]['endpoint_bool'] = False
                modified_count += 1
    
    print(f"Skipped {skipped_batches} batches with fewer than 3 segments")
    print(f"Modified {modified_count} entries")
    
    # Write back
    with open(metadata_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved to {metadata_path}")


if __name__ == '__main__':
    main()
