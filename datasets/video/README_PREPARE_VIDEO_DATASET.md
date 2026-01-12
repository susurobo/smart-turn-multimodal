# Video Dataset Preparation Script

Unified pipeline for preparing video datasets for the Smart Turn multimodal model. Combines face cropping, VAD-based segmentation, and HuggingFace-compatible metadata generation into a single script.

## Features

- **Single face detection per video** — Detects face once on the original video (middle frame), applies the same crop to all segments. This ensures consistent cropping and avoids per-segment detection errors.
- **VAD-based segmentation** — Uses Silero VAD to detect speech and segment videos at silence boundaries.
- **HuggingFace-compatible output** — Generates `metadata.jsonl` compatible with HF datasets.
- **Incremental processing** — Can append new videos to an existing dataset.
- **Nested directory support** — Recursively finds videos in nested folder structures.

## Requirements

```bash
pip install torch torchaudio opencv-python facenet-pytorch tqdm
```

FFmpeg must be installed:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Usage

```bash
python prepare_video_dataset.py <input_path> <output_dir> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `input_path` | Path to a video file or directory containing videos |
| `output_dir` | Output directory for the dataset |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--silence-ms` | 500 | Minimum silence duration (ms) to trigger a segment cut |
| `--face-size` | 224 | Output video size in pixels (square, e.g., 224x224) |
| `--margin` | 40 | Margin around detected face in pixels |
| `--dataset-name` | CasualConversations_Video | Dataset name tag in metadata |

## Examples

### Process a single video

```bash
python prepare_video_dataset.py /path/to/video.mp4 ./output_dataset/
```

### Process a directory of videos

```bash
python prepare_video_dataset.py /path/to/videos/ ./output_dataset/
```

### Process nested directory structure

```bash
# Input structure:
# CasualConversations/
# ├── CasualConversationsA/
# │   └── 1223/
# │       ├── 1223_09.mp4
# │       └── 1223_02.mp4
# └── CasualConversationsB/
#     └── 2333/
#         └── 2333_01.mp4

python prepare_video_dataset.py CasualConversations/ ./output_dataset/

# Output segments will be named:
# CasualConversationsA_1223_1223_09_segment_0000_4.8s_to_9.7s.mp4
# CasualConversationsA_1223_1223_02_segment_0000_...
# CasualConversationsB_2333_2333_01_segment_0000_...
```

### Add more videos to existing dataset

```bash
# First batch
python prepare_video_dataset.py batch1_videos/ ./output_dataset/

# Add more videos (appends to existing metadata.jsonl)
python prepare_video_dataset.py batch2_videos/ ./output_dataset/
```

### Custom settings

```bash
python prepare_video_dataset.py videos/ ./output/ \
    --silence-ms 750 \
    --face-size 112 \
    --margin 60 \
    --dataset-name "MyCustomDataset"
```

## Output Structure

```
output_dir/
├── video/
│   ├── VideoName_segment_0000_0.0s_to_4.8s.mp4
│   ├── VideoName_segment_0001_4.8s_to_9.7s.mp4
│   └── ...
├── audio/
│   ├── VideoName_segment_0000_0.0s_to_4.8s.wav
│   ├── VideoName_segment_0001_4.8s_to_9.7s.wav
│   └── ...
└── metadata.jsonl
```

### Video Output

- Face-cropped and scaled to `--face-size` (default 224x224)
- H.264 codec with AAC audio
- Maintains original frame rate

### Audio Output

- 16kHz mono WAV
- PCM 16-bit

## Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. FACE DETECTION (once per video)                             │
│     - Read middle frame of original video                       │
│     - Detect face using MTCNN                                   │
│     - Calculate square crop box with margin                     │
│     - Same crop applied to ALL segments of this video           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. VAD SEGMENTATION                                            │
│     - Extract audio from original video (16kHz mono)            │
│     - Run Silero VAD to detect speech segments                  │
│     - Find silence gaps ≥ --silence-ms                          │
│     - Generate cut points at silence boundaries                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. SEGMENT EXTRACTION                                          │
│     For each segment:                                           │
│     - Crop face region (same box for all segments)              │
│     - Scale to --face-size × --face-size                        │
│     - Save video (.mp4) and audio (.wav)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. METADATA GENERATION                                         │
│     - Generate metadata.jsonl with placeholder labels           │
│     - endpoint_bool: null (to be manually labeled)              │
│     - visual_label: null (optional, for visual cues)            │
└─────────────────────────────────────────────────────────────────┘
```

## Metadata Format

Each line in `metadata.jsonl` is a JSON object:

```json
{
  "file_name": "audio/VideoName_segment_0001_4.8s_to_9.7s.wav",
  "id": "uuid-string",
  "language": "eng",
  "endpoint_bool": null,
  "midfiller": false,
  "endfiller": false,
  "synthetic": false,
  "spoken_text": null,
  "dataset": "CasualConversations_Video",
  "video_path": "video/VideoName_segment_0001_4.8s_to_9.7s.mp4",
  "visual_label": null
}
```

### Fields to Label Manually

After processing, edit `metadata.jsonl` to fill in:

| Field | Type | Description |
|-------|------|-------------|
| `endpoint_bool` | `true` / `false` | Is this a complete turn (speaker finished speaking)? |
| `visual_label` | `string` or `null` | Optional visual cues: `"expressive"`, `"breath-in"`, `"mouth-open"`, etc. |

### Example: Before and After Labeling

**Before (generated):**
```json
{"endpoint_bool": null, "visual_label": null, ...}
```

**After (manually labeled):**
```json
{"endpoint_bool": false, "visual_label": "expressive", ...}
```

## Face Cropping Details

The face cropping works as follows:

1. **Detection**: MTCNN detects the face bounding box in the middle frame
2. **Margin**: Adds `--margin` pixels around the detected face (default 40px)
3. **Square crop**: Expands the box to a square (max of width/height)
4. **Scaling**: Scales the cropped region to `--face-size` × `--face-size`

This means:
- A detected 200x200 face box → scaled UP to 224x224
- A detected 300x300 face box → scaled DOWN to 224x224

The face always fills the frame consistently, regardless of camera distance.

## Troubleshooting

### "No face detected, skipping video"

The script couldn't detect a face in the middle frame. Try:
- Check if the video has a visible face in the middle
- The person might be looking away or face is occluded

### "No speech detected, skipping video"

Silero VAD found no speech. The video might:
- Have no audio track
- Contain only background noise
- Have very quiet speech

### FFmpeg errors

Ensure FFmpeg is installed and in your PATH:
```bash
ffmpeg -version
```

## Performance

- Uses GPU for face detection if CUDA is available
- Progress bars show ETA for both video-level and segment-level processing
- Typical speed: ~5-10 seconds per video (varies by video length and hardware)




# sequence

python prepare_video_dataset.py ~/susurobo/data/fb_casual_conv/mini/test  ./smart_turn_multimodal_test_dataset --silence-ms 200

