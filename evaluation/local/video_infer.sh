#!/bin/bash
# ============================================================================
# Usage example for video_inference.py
#
# Generation parameters (optional):
#   --temperature 0.7    (default: 0.7, 0.0 = deterministic)
#   --top_p 0.9          (default: 0.9)
#   --top_k 0            (default: 0, 0 = no limit)
# ============================================================================

# InternVL
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python video_inference.py \
    --model intern \
    --model_path OpenGVLab/InternVL3_5-8B-Instruct \
    --csv /path/to/input.csv \
    --video_dir /path/to/video_dir \
    --num_frames 32 \
    --temperature 0.7 \
    --top_p 1 \
    --top_k 40 \
    --output result.csv

# Qwen
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python video_inference.py \
    --model qwen \
    --model_path Qwen/Qwen3-VL-7B-Instruct \
    --csv /path/to/input.csv \
    --video_dir /path/to/video_dir \
    --num_frames 32 \
    --temperature 0.7 \
    --top_p 1 \
    --top_k 40 \
    --output result.csv
