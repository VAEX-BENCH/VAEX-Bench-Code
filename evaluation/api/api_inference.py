import os
import sys
import argparse
import base64
import io
import re
import time
import json
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu
from PIL import Image

# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='API-based Video Inference (Claude / Gemini / GPT / OpenRouter)')
    parser.add_argument('--provider', type=str, required=True,
                        choices=['claude', 'gemini', 'gpt', 'openrouter'],
                        help='API provider to use')

    # Common arguments
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (required for gemini, gpt, openrouter)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name/ID (defaults vary by provider)')
    parser.add_argument('--csv', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (default: auto-generated)')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames to extract (not used for gemini, default: 8)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Generation temperature (default: 0.0)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p sampling (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k sampling (default: 0 = disabled)')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--prompt_type', type=str, default='base',
                        choices=['base', 'open'],
                        help='base: validate answer format and retry. open: return raw response (default: base)')

    # Claude (AWS Bedrock) specific
    parser.add_argument('--aws_access_key', type=str, default=None,
                        help='AWS Access Key ID (required for claude)')
    parser.add_argument('--aws_secret_key', type=str, default=None,
                        help='AWS Secret Access Key (required for claude)')
    parser.add_argument('--aws_region', type=str, default='us-east-1',
                        help='AWS Region (default: us-east-1)')

    return parser.parse_args()


# ============================================================================
# Shared Utilities
# ============================================================================

DEFAULT_MODELS = {
    'claude':      'anthropic.claude-3-5-sonnet-20241022-v2:0',
    'gemini':      'gemini-2.0-flash',
    'gpt':         'gpt-4o',
    'openrouter':  'qwen/qwen3-vl-235b-a22b-instruct',
}

def get_index(max_frame, num_segments):
    """Uniformly sample frame indices from a video."""
    if num_segments == 1:
        return [max_frame // 2]
    seg_size = max_frame / num_segments
    return [int(seg_size / 2 + seg_size * i) for i in range(num_segments)]


def extract_frames_as_base64(video_path, num_frames):
    """Extract frames from a video and return them as base64-encoded JPEG strings."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        frame_indices = get_index(len(vr) - 1, num_frames)
        frames_b64 = []
        for idx in frame_indices:
            img = Image.fromarray(vr[idx].asnumpy())
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        return frames_b64
    except Exception as e:
        print(f"  Frame extraction failed: {e}")
        return None


def validate_answer(answer, question_type):
    """Validate and clean model answer based on question type.

    Returns (is_valid, cleaned_answer).
    """
    answer = str(answer).strip()

    if question_type in ['Object Count', 'Quantative reasoning']:
        numbers = re.findall(r'\b\d+\b', answer)
        if numbers and len(numbers) == 1:
            cleaned = numbers[0]
            if answer.strip() == cleaned or re.match(r'^\d+$', answer):
                return True, cleaned
        return False, answer

    match = re.match(r'^([A-E])\.?$', answer.upper().strip())
    if match:
        return True, match.group(1)

    return False, answer


def stronger_prompt(question, question_type, attempt):
    """Return a stricter prompt for retry attempts."""
    if question_type in ['Object Count', 'Quantative reasoning']:
        return question + "\n\nProvide ONLY a single number as your answer. No text, explanation, or extra characters."
    return question + "\n\nProvide ONLY a single letter (A, B, C, D, or E) as your answer. No explanation or punctuation."


# ============================================================================
# Claude (AWS Bedrock)
# ============================================================================

def build_claude_client(args):
    import boto3
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=args.aws_region,
        aws_access_key_id=args.aws_access_key,
        aws_secret_access_key=args.aws_secret_key,
    )


def inference_claude(client, video_path, question, question_type, args, cached_frames=None, max_retries=3):
    """Run Claude inference via AWS Bedrock."""
    try:
        if cached_frames:
            frames_b64 = cached_frames
        else:
            frames_b64 = extract_frames_as_base64(video_path, args.num_frames)
            if frames_b64 is None:
                raise ValueError("Frame extraction failed")

        image_content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f}}
            for f in frames_b64
        ]

        base_prompt = "\n\nProvide only one answer without any explanation. If it's a multiple choice question, respond with only the letter (A, B, C, D, or E). If it's a counting question, respond with only the number."
        use_base = (args.prompt_type == 'base')
        n_retries = max_retries if use_base else 1

        for attempt in range(n_retries):
            content = image_content.copy()
            if attempt == 0:
                full_question = question + base_prompt if use_base else question
            else:
                full_question = stronger_prompt(question, question_type, attempt)
            content.append({"type": "text", "text": full_question})

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
            if args.temperature > 0:
                body["temperature"] = args.temperature
            else:
                body["top_p"] = args.top_p
            if 1 <= args.top_k <= 500:
                body["top_k"] = args.top_k

            response = client.invoke_model(modelId=args.model, body=json.dumps(body))
            answer_text = json.loads(response['body'].read())['content'][0]['text'].strip()

            if not use_base:
                return answer_text, frames_b64 if not cached_frames else None

            is_valid, cleaned = validate_answer(answer_text, question_type)
            if is_valid:
                return cleaned, frames_b64 if not cached_frames else None
            if attempt < n_retries - 1:
                time.sleep(2)

        return answer_text, frames_b64 if not cached_frames else None

    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {e}", None


# ============================================================================
# Gemini
# ============================================================================

def build_gemini_model(args):
    import google.generativeai as genai
    genai.configure(api_key=args.api_key)
    gen_cfg = {"temperature": args.temperature, "top_p": args.top_p, "max_output_tokens": args.max_tokens}
    if args.top_k > 0:
        gen_cfg["top_k"] = args.top_k
    return genai.GenerativeModel(model_name=args.model, generation_config=gen_cfg)


def inference_gemini(model, video_path, question, question_type, args, cached_video_file=None, max_retries=3):
    """Run Gemini inference. Uploads the full video file (no frame extraction)."""
    import google.generativeai as genai

    new_video_file = None
    try:
        if cached_video_file:
            video_content = cached_video_file
        else:
            print(f"  Uploading video: {video_path}")
            video_file = genai.upload_file(video_path, mime_type="video/mp4")
            while video_file.state.name == "PROCESSING":
                time.sleep(1)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_file.state.name}")
            video_content = video_file
            new_video_file = video_file

        base_prompt = "Provide only one answer without any explanation. If it's a multiple choice question, respond with only the letter (A, B, C, D, or E). If it's a counting question, respond with only the number.\n\n"
        use_base = (args.prompt_type == 'base')
        n_retries = max_retries if use_base else 1

        full_question = (base_prompt + question) if use_base else question

        for attempt in range(n_retries):
            response = model.generate_content([video_content, full_question])
            answer_text = response.text.strip()

            if not use_base:
                return answer_text, new_video_file

            is_valid, cleaned = validate_answer(answer_text, question_type)
            if is_valid:
                return cleaned, new_video_file
            if attempt < n_retries - 1:
                full_question = stronger_prompt(question, question_type, attempt + 1)
                time.sleep(2)

        return answer_text, new_video_file

    except Exception as e:
        if new_video_file:
            try:
                genai.delete_file(new_video_file.name)
            except Exception:
                pass
        traceback.print_exc()
        return f"ERROR: {e}", None


# ============================================================================
# GPT (OpenAI)
# ============================================================================

def build_gpt_client(args):
    from openai import OpenAI
    return OpenAI(api_key=args.api_key, timeout=600.0)


def inference_gpt(client, video_path, question, question_type, args, cached_frames=None, max_retries=3):
    """Run GPT inference via OpenAI API."""
    try:
        if cached_frames:
            frames_b64 = cached_frames
        else:
            frames_b64 = extract_frames_as_base64(video_path, args.num_frames)
            if frames_b64 is None:
                raise ValueError("Frame extraction failed")

        image_msgs = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "high"}}
            for f in frames_b64
        ]

        use_base = (args.prompt_type == 'base')
        base_prompt = "\n\nProvide only one answer without any explanation. If it's a multiple choice question, respond with only the letter (A, B, C, D, or E). If it's a counting question, respond with only the number."
        n_retries = max_retries if use_base else 1

        is_reasoning = any(x in args.model.lower() for x in ['gpt-5', 'o3', 'o1'])

        for attempt in range(n_retries):
            content = image_msgs.copy()
            if attempt == 0:
                full_question = question + base_prompt if use_base else question
            else:
                full_question = stronger_prompt(question, question_type, attempt)
            content.append({"type": "text", "text": full_question})

            kwargs = {"model": args.model, "messages": [{"role": "user", "content": content}]}
            if is_reasoning:
                kwargs["max_completion_tokens"] = args.max_tokens
            else:
                kwargs["max_tokens"] = args.max_tokens
                kwargs["temperature"] = args.temperature
                kwargs["top_p"] = args.top_p

            response = client.chat.completions.create(**kwargs)
            answer_text = response.choices[0].message.content.strip()

            if not use_base:
                return answer_text, frames_b64 if not cached_frames else None

            is_valid, cleaned = validate_answer(answer_text, question_type)
            if is_valid:
                return cleaned, frames_b64 if not cached_frames else None
            if attempt < n_retries - 1:
                time.sleep(2)

        return answer_text, frames_b64 if not cached_frames else None

    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {e}", None


# ============================================================================
# OpenRouter
# ============================================================================

def build_openrouter_client(args):
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key, timeout=600.0)


def inference_openrouter(client, video_path, question, question_type, args, cached_frames=None, max_retries=3):
    """Run inference via OpenRouter API."""
    try:
        if cached_frames:
            frames_b64 = cached_frames
        else:
            frames_b64 = extract_frames_as_base64(video_path, args.num_frames)
            if frames_b64 is None:
                raise ValueError("Frame extraction failed")

        image_msgs = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
            for f in frames_b64
        ]

        use_base = (args.prompt_type == 'base')
        base_prompt = "\n\nProvide only one answer without any explanation. If it's a multiple choice question, respond with only the letter (A, B, C, D, or E). If it's a counting question, respond with only the number."
        n_retries = max_retries if use_base else 1

        extra_body = {"top_k": args.top_k} if args.top_k > 0 else {}

        for attempt in range(n_retries):
            content = image_msgs.copy()
            if attempt == 0:
                full_question = question + base_prompt if use_base else question
            else:
                full_question = stronger_prompt(question, question_type, attempt)
            content.append({"type": "text", "text": full_question})

            response = client.chat.completions.create(
                extra_body=extra_body,
                model=args.model,
                messages=[{"role": "user", "content": content}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            answer_text = response.choices[0].message.content.strip()

            if not use_base:
                return answer_text, frames_b64 if not cached_frames else None

            is_valid, cleaned = validate_answer(answer_text, question_type)
            if is_valid:
                return cleaned, frames_b64 if not cached_frames else None
            if attempt < n_retries - 1:
                time.sleep(2)

        return answer_text, frames_b64 if not cached_frames else None

    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {e}", None


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Set default model
    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]

    # Set default output path
    if args.output is None:
        args.output = f"{args.provider}_{Path(args.model).name}.csv"

    print("=" * 70)
    print(f"Provider:    {args.provider}")
    print(f"Model:       {args.model}")
    print(f"Input CSV:   {args.csv}")
    print(f"Video dir:   {args.video_dir}")
    print(f"Output:      {args.output}")
    print(f"Frames:      {args.num_frames}" + (" (ignored for gemini)" if args.provider == "gemini" else ""))
    print(f"Temperature: {args.temperature}  Top-p: {args.top_p}  Top-k: {args.top_k}  Max tokens: {args.max_tokens}")
    print(f"Prompt type: {args.prompt_type}")
    print("=" * 70)

    # Validate credentials
    if args.provider == 'claude':
        if not args.aws_access_key or not args.aws_secret_key:
            print("ERROR: --aws_access_key and --aws_secret_key are required for claude provider.")
            sys.exit(1)
    else:
        if not args.api_key:
            print(f"ERROR: --api_key is required for {args.provider} provider.")
            sys.exit(1)

    # Build client / model
    if args.provider == 'claude':
        client = build_claude_client(args)
        inference_fn = lambda *a, **kw: inference_claude(client, *a, **kw)
    elif args.provider == 'gemini':
        client = build_gemini_model(args)
        inference_fn = lambda *a, **kw: inference_gemini(client, *a, **kw)
    elif args.provider == 'gpt':
        client = build_gpt_client(args)
        inference_fn = lambda *a, **kw: inference_gpt(client, *a, **kw)
    elif args.provider == 'openrouter':
        client = build_openrouter_client(args)
        inference_fn = lambda *a, **kw: inference_openrouter(client, *a, **kw)

    # Load CSV
    df = pd.read_csv(args.csv, dtype={'answer': str})
    print(f"\nLoaded {len(df)} entries\n")

    video_cache = {}

    for idx, row in df.iterrows():
        video_id = row['video_id']
        question = row['question']
        question_type = row.get('question_type', 'Unknown')
        video_path = os.path.join(args.video_dir, f"video{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"[{idx+1}/{len(df)}] ERROR: Video not found — {video_path}")
            df.at[idx, 'answer'] = "ERROR: Video file not found"
            continue

        cached = video_cache.get(video_id)

        try:
            answer, new_cache = inference_fn(
                video_path=video_path,
                question=question,
                question_type=question_type,
                args=args,
                cached_frames=cached if args.provider != 'gemini' else None,
                cached_video_file=cached if args.provider == 'gemini' else None,
                max_retries=3,
            )
            if new_cache is not None:
                video_cache[video_id] = new_cache
        except Exception as e:
            traceback.print_exc()
            answer = f"ERROR: {e}"

        df.at[idx, 'answer'] = answer
        print(f"[{idx+1}/{len(df)}] {answer}")

        # Rate limiting for Gemini
        if args.provider == 'gemini':
            wait = 32 if 'pro' in args.model.lower() else 5
            time.sleep(wait)

    # Cleanup Gemini uploaded files
    if args.provider == 'gemini':
        import google.generativeai as genai
        for vid_id, vf in video_cache.items():
            try:
                genai.delete_file(vf.name)
            except Exception:
                pass

    # Check for errors
    has_error = df['answer'].astype(str).str.startswith('ERROR', na=False).any()
    if has_error:
        base, ext = os.path.splitext(args.output)
        args.output = f"{base}_error{ext}"
        print("\nErrors detected — appended '_error' to output filename.")

    df.to_csv(args.output, index=False)
    print(f"\nDone! Results saved to '{args.output}'.")


if __name__ == "__main__":
    main()
