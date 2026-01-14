# Smart Turn Multimodal v0.0.1

An open source, community-driven, native audio-visual turn detection fork of [Pipecat's Smart Turn](https://github.com/pipecat-ai/smart-turn).

* [HuggingFace page with model weights](https://huggingface.co/susuROBO/smart-turn-multimodal)

Deciding when to speak in a multi-party conversations is an ill-defined problem: people navigate it with various levels of success that in part defines their personality. Visual cues appear to be an important source of data: we tend to give people floor if they look like they are about to start or continue speaking. 

Yet, current voice frameworks like Pipecat do not incorporate visual cues.
Smart Turn Multimodal is intended as a step towards filling this gap.

As with Smart Turn, we continue with the BSD 2-clause license. Anyone can use, fork, and contribute to this project. 

 ## Features

* **Languages**
   Trained on English samples from a subset of Meta Casual Conversations.
* **Fast inference time**
   Slower than the original Smart Turn but still suitable for real-time applications.
 * Works in conjunction with a lightweight VAD model like Silero: like Smart Turn it only needs to run during periods of silence which also
 means it is not trained to be predictive before silence happens.
* **As Smart Turn, Smart Turn Multimodal is available in CPU (8MB quantized) and GPU (32MB unquantized) versions**
  * The GPU version uses `fp32` weights, meaning it runs slightly faster on GPUs, and has slightly improved accuracy by around 1%
  * The CPU version is quantized to `int8`, making it significantly smaller and faster for CPU inference, at a slight accuracy cost
* **Audio and video native**
  * The model works directly with PCM audio samples and video frames, rather than text transcriptions, allowing it to take into account not just prosody cues in the user's speech but also facial expressions such as gaze, mouth and nods.
* **Open source**
  * Training script, and model weights are all open source.
  * The [Meta Casual Conversations Dataset](https://ai.meta.com/datasets/casual-conversations-dataset/) allows commercial use but limits redistribution, so training requires downloading and pre-processing the data from Meta.

## Inference Performance

### Direct Inference Performance
*Using pre-computed zero features (inference only)*

| Provider | P50 Latency (ms) | P90 Latency (ms) | Mean Latency (ms) | Throughput (samples/sec) |
| :------- | ---------------: | ---------------: | ----------------: | -----------------------: |
| CPU      |           769.32 |          1188.00 |            807.65 |                      1.2 |
| T4       |            69.74 |            71.54 |             69.19 |                     14.5 |

### Audio Feature Extraction Performance
*Whisper feature extraction from 8-second audio*

| Component               | P50 Latency (ms) | P90 Latency (ms) | Mean Latency (ms) | Throughput (samples/sec) |
| :---------------------- | ---------------: | ---------------: | ----------------: | -----------------------: |
| Audio Feature Extractor |            33.78 |            37.29 |             34.53 |                     29.0 |

### Video Preprocessing Performance
*Video decoding, resizing, and normalization*

| Component          | P50 Latency (ms) | P90 Latency (ms) | Mean Latency (ms) | Throughput (samples/sec) |
| :----------------- | ---------------: | ---------------: | ----------------: | -----------------------: |
| Video Preprocessor |            63.00 |            72.27 |             64.82 |                     15.4 |

### End-to-End Performance
*Audio feature extraction + video preprocessing + inference*

| Provider | P50 Latency (ms) | P90 Latency (ms) | Mean Latency (ms) | Throughput (samples/sec) |
| :------- | ---------------: | ---------------: | ----------------: | -----------------------: |
| CPU      |           716.00 |          1005.46 |            739.80 |                      1.4 |
| T4       |            96.76 |           111.24 |            100.17 |                     10.0 |



 ## Run the model locally

**Set up the environment:**

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You may need to install PortAudio development libraries if not already installed as those are required for PyAudio:

**Ubuntu/Debian**

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```

**macOS (using Homebrew)**

```bash
brew install portaudio
```

**Run the web demo**

Run a web-server that streams audio and video from the system microphone and camera, detects segment start/stop using VAD, and sends each segment to the model for a phrase endpoint prediction.

```
python app_multimodal.py
```

## Model usage

### With local inference

From the Smart Turn Multimodal source repository, obtain the file  `inference_multimodal.py`. Invoke the `predict_endpoint()` function with your audio and video. For an example, please see `app_multimodal.py`:


## Project goals


The current version of this model is based on the audio pipeline from Smart Turn Whisper Tiny backbone and late fusion with the R3D-18 video branch. More on the architecture below.

The high-level goal of this project is to encourage and enable the usage of video in conversational pipelines.

Medium-term goals:
  - Access to more diverse datasets (video conferences, face-to-face conversations)
  - Support for additional languages
  - Experiment with further optimizations and architecture improvements

## Model architecture

* **Audio branch:** Whisper Tiny encoder (8s context) with cross-attention pooling → 384-dim embedding
* **Video branch:** R3D-18 (Kinetics-400 pretrained) processing last 32 frames (~1s) → 256-dim embedding  
* **Fusion:** Late fusion via concatenation + linear projection back to 384-dim

More details on the architecture and training are in the [blog](https://susurobo.jp/blog/smart_turn_multimodal.html).


## Inference

Sample code for inference is included in `inference_multimodal.py`. See `app_multimodal.py` for usage examples.

## Training

All training code is defined in `train_multimodal.py`.

The training code uses Smart Turn's audio datasets from the [pipecat-ai](https://huggingface.co/pipecat-ai) HuggingFace repository and also expects the audio-visual dataset available in a specific format (more to follow.)

You can run training locally or using [Modal](https://modal.com) (using `train_multimodal.py`). Training runs are logged to [Weights & Biases](https://www.wandb.ai) unless you disable logging.

```
# To run a training job on Modal, run:
modal volume create endpointing
modal run --detach train_modal.py --multimodal casual_conversations_v3
```

### Collecting and contributing data

Currently, the following datasets are used for training and evaluation:

* [Meta Casual Conversations](https://ai.meta.com/datasets/casual-conversations-dataset/) that consists mostly of prompted monologues

More datasets will help to generalize the model to various conversational scenarios.

## Things to do

### Provide additional datasets

### Architecture experiments

## Contributors

- [Maxim](https://github.com/maxipesfix)

Thank you to Pipecat and Daily for leading the open-source voice ecosystem. Thank you to Meta for the Casual Conversations Dataset.
