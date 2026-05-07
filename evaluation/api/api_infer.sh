#!/bin/bash
# ============================================================================
# Usage examples for api_inference.py
#
# Providers: claude, gemini, gpt, openrouter
#
# Generation parameters (optional):
#   --temperature 0.0    (default: 0.0)
#   --top_p 1.0          (default: 1.0)
#   --top_k 0            (default: 0 = disabled)
#   --max_tokens 512     (default: 512)
#   --prompt_type base   (base: validate answer format, open: return raw response)
# ============================================================================

# Claude (AWS Bedrock)
python api_inference.py \
    --provider claude \
    --aws_access_key YOUR_AWS_ACCESS_KEY \
    --aws_secret_key YOUR_AWS_SECRET_KEY \
    --aws_region us-east-1 \
    --model anthropic.claude-3-5-sonnet-20241022-v2:0 \
    --csv /path/to/input.csv \
    --video_dir /path/to/video_dir \
    --num_frames 8 \
    --temperature 0.7 \
    --max_tokens 512 \
    --prompt_type base \
    --output claude_result.csv

# Gemini
# python api_inference.py \
#     --provider gemini \
#     --api_key YOUR_GEMINI_API_KEY \
#     --model gemini-2.0-flash \
#     --csv /path/to/input.csv \
#     --video_dir /path/to/video_dir \
#     --temperature 0.7 \
#     --max_tokens 512 \
#     --prompt_type base \
#     --output gemini_result.csv

# GPT
# python api_inference.py \
#     --provider gpt \
#     --api_key YOUR_OPENAI_API_KEY \
#     --model gpt-4o \
#     --csv /path/to/input.csv \
#     --video_dir /path/to/video_dir \
#     --num_frames 8 \
#     --temperature 0.7 \
#     --max_tokens 512 \
#     --prompt_type base \
#     --output gpt_result.csv

# OpenRouter
# python api_inference.py \
#     --provider openrouter \
#     --api_key YOUR_OPENROUTER_API_KEY \
#     --model qwen/qwen3-vl-235b-a22b-instruct \
#     --csv /path/to/input.csv \
#     --video_dir /path/to/video_dir \
#     --num_frames 32 \
#     --temperature 0.7 \
#     --top_k 40 \
#     --max_tokens 1024 \
#     --prompt_type base \
#     --output openrouter_result.csv
