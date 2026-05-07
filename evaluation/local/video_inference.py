import warnings
warnings.filterwarnings('ignore')
import os
import sys
import argparse
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import pandas as pd
import numpy as np
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# ============================================================================
# Argument Parsing
# ============================================================================
parser = argparse.ArgumentParser(description='VLM Inference')
parser.add_argument('--model', type=str, required=True,
                    choices=['intern', 'qwen'],
                    help='Model selection')
parser.add_argument('--model_path', type=str, required=True,
                    help='Model path')
parser.add_argument('--csv', type=str, required=True,
                    help='Input CSV file path')
parser.add_argument('--video_dir', type=str, required=None,
                    help='Directory containing video files')
parser.add_argument('--num_frames', type=int, required=True,
                    help='Number of frames to extract')
parser.add_argument('--output', type=str, default=None,
                    help='Output CSV filename (optional, default: auto-generated)')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Generation temperature (default: 0.7)')
parser.add_argument('--top_p', type=float, default=0.9,
                    help='Top-p sampling (default: 0.9)')
parser.add_argument('--top_k', type=int, default=0,
                    help='Top-k sampling (default: 0, no limit)')

args = parser.parse_args()

MODEL_PATH = args.model_path

# Extract model parameter count (e.g., "8B" -> "8")
model_size_match = re.search(r'(\d+)[bB]', MODEL_PATH, re.IGNORECASE)
if model_size_match:
    model_size = model_size_match.group(1)
else:
    model_size = 'unknown'

# Create result directory
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

# Auto-generate output filename
if args.output:
    output_file = os.path.join(result_dir, args.output)
else:
    output_file = os.path.join(result_dir, f"{args.model}_{model_size}b.csv")

print("=" * 70)
print(f"Model type: {args.model}")
print(f"Model path: {MODEL_PATH}")
print(f"Model size: {model_size}B")
print(f"Input CSV: {args.csv}")
if args.video_dir:
    print(f"Video directory: {args.video_dir}")
print(f"Number of frames: {args.num_frames}")
print(f"Generation params: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
print(f"Output file: {output_file}")
print("=" * 70)

# ============================================================================
# Model Loading and Inference Functions
# ============================================================================

def load_intern_model():
    """Load InternVL model"""
    from transformers import AutoTokenizer, AutoModel
    from transformers import logging as hf_logging
    from decord import VideoReader, cpu
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    print("\nLoading InternVL model...")

    # Temporarily change logging level to show checkpoint loading progress
    original_log_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_info()

    # Helper functions
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Restore logging level
    hf_logging.set_verbosity(original_log_level)

    print("\nInternVL model loaded successfully!")

    def inference_fn(video_path, question, num_frames, cached_video_data=None):
        # Use cached data if available, otherwise load video
        if cached_video_data is not None:
            pixel_values, num_patches_list = cached_video_data
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        else:
            # Load video following official code style
            pixel_values, num_patches_list = load_video(
                video_path,
                bound=None,
                input_size=448,
                max_num=1,
                num_segments=num_frames
            )
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # Add frame prefix
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        full_question = video_prefix + question + "\n\nProvide only your answer without any explanation. If it's a multiple choice question, respond with only the letter (A, B, C, D, or E). If it's a counting question, respond with only the number."
        # Set generation parameters
        generation_config = dict(
            max_new_tokens=512,
            do_sample=(args.temperature > 0),
            temperature=args.temperature,
            top_p=args.top_p
        )
        if args.top_k > 0:
            generation_config['top_k'] = args.top_k
        response = model.chat(
            tokenizer,
            pixel_values,
            full_question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None
        )

        # Only clean up pixel values when not using cached data (preserve original cache)
        if cached_video_data is None:
            return response, (pixel_values.cpu(), num_patches_list)
        else:
            return response, None

    return inference_fn


def load_qwen_model():
    """Load Qwen3-VL model"""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print("\nLoading Qwen model...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    print("Qwen model loaded successfully!")

    def inference_fn(video_path, question, num_frames):
        content = []

        # Read video length and compute fps
        target_fps = 1.0  # default value
        if video_path:
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                fps = vr.get_avg_fps()
                duration = total_frames / fps  # video duration in seconds

                # Compute fps to obtain the desired number of frames
                # Qwen uses fewer frames in practice, so apply a scaling factor
                target_fps = (num_frames * 2.0) / duration if duration > 0 else 1.0
            except Exception as e:
                target_fps = num_frames / 30.0  # fallback

        # Add video
        if video_path:
            content.append({
                "type": "video",
                "video": video_path,
                "max_frames": num_frames,
                "max_pixels": 640 * 480 * num_frames,  # required to control frame count
            })

        final_question = question + "\n\nProvide only your answer without any explanation. If it's a multiple choice question, respond with only the letter (A, B, C, D, or E). If it's a counting question, respond with only the number."
        content.append({
            "type": "text",
            "text": final_question
        })

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            fps=target_fps,
        ).to(model.device)

        # Set generation parameters
        gen_kwargs = {
            'max_new_tokens': 512,
            'do_sample': (args.temperature > 0),
            'temperature': args.temperature,
            'top_p': args.top_p,
        }
        if args.top_k > 0:
            gen_kwargs['top_k'] = args.top_k

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
            response = processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

        # Remove "Answer only."
        response = response.strip()
        if "Answer only." in response:
            response = response.replace("Answer only.", "").strip()

        torch.cuda.empty_cache()
        return response

    return inference_fn


# ============================================================================
# Model Loading
# ============================================================================
print(f"\nGPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")

try:
    if args.model == 'intern':
        inference_function = load_intern_model()
    elif args.model == 'qwen':
        inference_function = load_qwen_model()
    else:
        raise ValueError(f"Unsupported model: {args.model}")
except Exception as e:
    print(f"Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Read CSV File
# ============================================================================
print(f"\nLoading CSV file: {args.csv}")
df = pd.read_csv(args.csv)
print(f"Found {len(df)} entries\n")

# ============================================================================
# Dataset Processing
# ============================================================================
print("Starting dataset processing\n")

# Video decoding cache (each video_id decoded only once)
video_cache = {}
cache_hits = 0
cache_misses = 0

for idx, row in df.iterrows():
    video_id = row['video_id']
    query = row['question']

    video_path = os.path.join(args.video_dir, f"video{video_id}.mp4") if args.video_dir else None

    # Check file existence
    if video_path and not os.path.exists(video_path):
        df.at[idx, 'answer'] = "ERROR: Video file not found"
        continue

    if not video_path:
        df.at[idx, 'answer'] = "ERROR: No video provided"
        continue

    try:
        # Video caching logic (InternVL only)
        cached_data = None
        new_cache_data = None

        if args.model == 'intern' and video_id in video_cache:
            cached_data = video_cache[video_id]
            cache_hits += 1
        elif args.model == 'intern':
            cache_misses += 1

        if args.model == 'qwen':
            response = inference_function(video_path, query, args.num_frames)
        elif args.model == 'intern':
            response, new_cache_data = inference_function(video_path, query, args.num_frames, cached_video_data=cached_data)
            # Cache newly decoded video data
            if new_cache_data is not None:
                video_cache[video_id] = new_cache_data
        else:
            response = inference_function(video_path, query, args.num_frames)

        df.at[idx, 'answer'] = response
        print(f"[{idx+1}/{len(df)}] {response}")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[{idx+1}/{len(df)}] ERROR: {str(e)}")
        print(error_details)
        df.at[idx, 'answer'] = f"ERROR: {str(e)}"

# ============================================================================
# Save Results
# ============================================================================
# Print cache statistics
if args.model == 'intern':
    total_requests = cache_hits + cache_misses
    print(f"\n{'='*70}")
    print(f"Video caching statistics:")
    print(f"  - Total requests: {total_requests}")
    print(f"  - Cache hits: {cache_hits} ({cache_hits/total_requests*100:.1f}%)" if total_requests > 0 else "  - Cache hits: 0")
    print(f"  - Cache misses: {cache_misses} ({cache_misses/total_requests*100:.1f}%)" if total_requests > 0 else "  - Cache misses: 0")
    print(f"  - Unique videos: {len(video_cache)}")
    print(f"{'='*70}\n")

# Check for ERRORs
has_error = False
if 'answer' in df.columns:
    has_error = df['answer'].astype(str).str.startswith('ERROR', na=False).any()

# Append '_error' to filename if errors are detected
if has_error:
    base_name = os.path.splitext(output_file)[0]
    ext = os.path.splitext(output_file)[1]
    output_file = f"{base_name}_error{ext}"
    print("\n Errors detected — appended '_error' to output filename.")

df.to_csv(output_file, index=False)
print(f"\nDone! Results saved to '{output_file}'.")
