Day-to-day workflow

git push → goes to your private repo (origin)
git push public main → pushes to your public fork when ready to release
git fetch upstream && git merge upstream/main → pulls in updates from the original project


## Quantize for CPU

modal run train_modal.py --quantize "/data/output/v3.1-2026-01-01_12:33_run1/final_model/exports/model_fp32.onnx"

modal volume get endpointing /output/v3.1-2026-01-01_12:33_run1/final_model/exports/model_int8_static_calib1024.onnx



# copy dataset to modal
modal volume put endpointing ./datasets/video/smart_turn_multimodal_dataset /datasets/casual_conversations


# run training
Mode 1: The Shortcut (Quick Experiment)

Bash

modal run --detach train_modal.py --multimodal casual_conversations
# Defaults to openai/whisper-tiny -> Slices weights
Mode 2: The Gold Standard (High Performance)

Bash

modal run --detach train_modal.py \
    --multimodal casual_conversations \
    --audio-checkpoint "/data/output/audio_baseline_v1/final_model"
# Points to local volume -> Direct loads



modal run --detach benchmark_multimodal.py \
    --onnx-path /data/output/mm_run_20260111_1904/model_fixed.onnx \
    --dataset-path /data/datasets/casual_conversations \
    --run-description "Multimodal_FIXED_WEIGHTS"



## benchmark runs on test data

### Multimodal benchmark
modal run benchmark_multimodal.py \
  --onnx-path "/data/output/mm_run_20260111_1904/model_fixed.onnx" \
  --dataset-path "/data/datasets/smart_turn_multimodal_casual_conv_mini_a3_test" \
  --run-description "CasualConv_A3_Test"

### Audio-only benchmark  
modal run benchmark.py \
  --onnx-path "hf-audio-only" \
  --dataset-path "/data/datasets/smart_turn_multimodal_casual_conv_mini_a3_test" \
  --run-description "CasualConv_A3_Test"



## performance multimodal 
  modal run --detach benchmark_multimodal.py --onnx-path "/data/output/mm_run_20260111_1904/model_fixed.onnx"  \
    --dataset-path "/data/datasets/smart_turn_multimodal_casual_conv_mini_a3_test" --run-description "multimodal_perf_benchmark" --perf-runs 100



audio run on A3
-0.9365807771682739, 'prob_std': 3.565020799636841, 'csv_path':
'/data/benchmark/model/CasualConv_A3_Test_20260112_131334.csv', 'plot_path':
'/data/benchmark/model/CasualConv_A3_Test_20260112_131334.png'}}