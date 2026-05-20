# -*- coding: utf-8 -*-
"""
Lumina-DiMOO GenEval 图像生成评估脚本
用于批量生成 GenEval 评估所需的图像
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

import json
import argparse
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoTokenizer
from diffusers import VQModel

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates
from utils.en_segmentation import en_tristate_segmentation, save_en_tristate_overlay
from generators.image_generation_generator import generate_image

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Lumina-DiMOO GenEval image generation evaluation',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  # 原始 Lumina cosine 采样
  CUDA_VISIBLE_DEVICES=0 conda run -n lumina_dimoo python evaluation/gen_eval/geneval_lumina_dimoo.py \\
    --metadata_file prompts/evaluation_metadata.jsonl \\
    --output_root output/geneval_original \\
    --height 512 --width 512 --timesteps 64 \\
    --no-en_region_sampling

  # EN 区域独立采样（算法超参数使用本脚本默认值）
  CUDA_VISIBLE_DEVICES=0 conda run -n lumina_dimoo python evaluation/gen_eval/geneval_lumina_dimoo.py \\
    --metadata_file prompts/evaluation_metadata.jsonl \\
    --output_root output/geneval_en_region \\
    --height 512 --width 512 \\
    --en_region_sampling

  # EN 区域独立采样，并额外保存 EN 可视化图
  CUDA_VISIBLE_DEVICES=0 conda run -n lumina_dimoo python evaluation/gen_eval/geneval_lumina_dimoo.py \\
    --metadata_file prompts/evaluation_metadata.jsonl \\
    --output_root output/geneval_en_region_heatmap \\
    --height 512 --width 512 \\
    --en_region_sampling --en_heatmap
""",
    )

    # Model and data paths
    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights',
                        help='Model checkpoint path')
    parser.add_argument('--metadata_file', type=str,
                        default='prompts/evaluation_metadata.jsonl',
                        help='Path to evaluation_metadata.jsonl file')
    parser.add_argument('--output_root', type=str,
                        default='output/geneval_results',
                        help='Root directory for output')
    parser.add_argument('--output_dir_prefix', type=str, default='',
                        help='Optional prefix for the generated output directory name')

    # Generation parameters
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of samples to generate per prompt')
    parser.add_argument('--height', type=int, default=512,
                        help='Generated image height')
    parser.add_argument('--width', type=int, default=512,
                        help='Generated image width')
    parser.add_argument('--timesteps', type=int, default=64,
                        help='Number of sampling timesteps')
    parser.add_argument('--cfg_scale', type=float, default=4.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for first sample')

    # Cache parameters (for acceleration)
    parser.add_argument('--use_cache', action='store_true', default=True,
                        help='Enable CFG cache to reduce computation')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false',
                        help='Disable CFG cache')
    parser.add_argument('--cache_ratio', type=float, default=0.9,
                        help='Ratio of reused tokens (0-1); higher = faster')
    parser.add_argument('--warmup_ratio', type=float, default=0.3,
                        help='Warmup ratio for caching [0-1); lower = faster')
    parser.add_argument('--refresh_interval', type=int, default=5,
                        help='Refresh cache every N steps; higher = faster')

    # EN region sampling parameters
    parser.add_argument('--en_region_sampling',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Enable EN region-independent sampling (default: enabled; use --no-en_region_sampling for original Lumina)')
    parser.add_argument('--en_heatmap', action='store_true',
                        help='Save EN tri-state visualizations for generated samples')
    parser.add_argument('--en_snapshot_step', type=int, default=8,
                        help='Zero-based earlier EN snapshot step')
    parser.add_argument('--en_snapshot_step_b', type=int, default=9,
                        help='Zero-based later EN snapshot step')
    parser.add_argument('--en_threshold_pair', type=str, default='0.18:0.48',
                        help='EN tri-state thresholds, e.g. "0.18:0.48"')
    parser.add_argument('--en_alpha', type=float, default=0.45,
                        help='Red overlay alpha for EN heatmap')
    parser.add_argument('--en_region_steps', type=str, default='4,12,20',
                        help='Remaining sampling steps for EN labels 0,1,2')
    parser.add_argument('--en_region_cache_start_step', type=int, default=10,
                        help='First zero-based step that may use cache when EN region sampling is enabled')
    parser.add_argument('--cache_prune_ratio', type=float, default=0.0,
                        help='Fraction of region 0+1 tokens to prune from cache after they finish sampling (0 disables pruning)')
    parser.add_argument('--cache_prune_context_radius', type=int, default=2,
                        help='Boundary protection radius around target region-2 tokens during cache pruning')
    parser.add_argument('--cache_prune_anchor_stride', type=int, default=0,
                        help='Deterministic anchor stride kept during cache pruning (0 disables anchors)')
    parser.add_argument('--cache_prune_anchor_ratio', type=float, default=0.0,
                        help='Random anchor ratio kept during cache pruning')
    parser.add_argument('--cache_prune_min_keep', type=int, default=0,
                        help='Minimum candidate tokens to keep after cache pruning')

    return parser


def parse_en_threshold_pair(threshold_text):
    left, right = threshold_text.replace("-", ":").split(":")
    threshold_low = float(left)
    threshold_high = float(right)
    if threshold_high <= threshold_low:
        raise ValueError(f"EN threshold pair must be increasing, got {threshold_low}:{threshold_high}")
    return threshold_low, threshold_high


def parse_en_region_steps(steps_text):
    parts = steps_text.replace(":", ",").split(",")
    if len(parts) != 3:
        raise ValueError(f"EN region steps must have 3 integers for labels 0,1,2, got {steps_text!r}")
    steps = [int(part.strip()) for part in parts]
    if any(step_count < 0 for step_count in steps):
        raise ValueError(f"EN region steps must be non-negative, got {steps}")
    return steps


def format_dir_value(value):
    return f"{value:g}".replace(".", "p")


def build_run_dir_name(args, timestamp, effective_timesteps):
    if args.en_region_sampling:
        en_region_steps = parse_en_region_steps(args.en_region_steps)
        en_part = (
            f"en1_s{args.en_snapshot_step}-{args.en_snapshot_step_b}"
            f"_thr{args.en_threshold_pair.replace(':', '-')}"
            f"_rs{'-'.join(str(step) for step in en_region_steps)}"
            f"_cs{args.en_region_cache_start_step}"
        )
    else:
        en_part = "en0"

    cache_part = (
        f"cache{int(args.use_cache)}"
        f"_cr{format_dir_value(args.cache_ratio)}"
        f"_wu{format_dir_value(args.warmup_ratio)}"
        f"_ri{args.refresh_interval}"
    )
    output_name = (
        f"lumina_dimoo_geneval_{timestamp}"
        f"_h{args.height}_w{args.width}"
        f"_t{args.timesteps}_eff{effective_timesteps}"
        f"_cfg{format_dir_value(args.cfg_scale)}"
        f"_temp{format_dir_value(args.temperature)}"
        f"_seed{args.seed}"
        f"_n{args.n_samples}"
        f"_{cache_part}_{en_part}"
    )
    if args.output_dir_prefix:
        output_name = f"{args.output_dir_prefix}_{output_name}"
    return output_name


def print_generation_config(args, effective_timesteps, en_region_steps=None, finish_steps=None):
    print("📋 生成参数详情:")
    print(f"  image_size={args.height}x{args.width}")
    print(f"  n_samples={args.n_samples}")
    print(f"  timesteps={args.timesteps}")
    print(f"  effective_timesteps={effective_timesteps}")
    print(f"  cfg_scale={args.cfg_scale}")
    print(f"  temperature={args.temperature}")
    print(f"  seed={args.seed}")
    print(
        "  cache="
        f"{args.use_cache} cache_ratio={args.cache_ratio} "
        f"warmup_ratio={args.warmup_ratio} refresh_interval={args.refresh_interval}"
    )
    print(
        "  pruning="
        f"ratio={args.cache_prune_ratio} context_radius={args.cache_prune_context_radius} "
        f"anchor_stride={args.cache_prune_anchor_stride} anchor_ratio={args.cache_prune_anchor_ratio} "
        f"min_keep={args.cache_prune_min_keep}"
    )
    if args.en_region_sampling:
        print("  en_region_sampling=True")
        print(f"  en_snapshots=step{args.en_snapshot_step}->step{args.en_snapshot_step_b}")
        print(f"  en_threshold_pair={args.en_threshold_pair}")
        print(f"  en_region_steps={en_region_steps}")
        print(f"  en_finish_steps={finish_steps}")
        print(f"  en_region_cache_start_step={args.en_region_cache_start_step}")
    else:
        print("  en_region_sampling=False")


def convert_torch_to_int(data):
    """Convert torch tensors to integers for JSON serialization"""
    if isinstance(data, torch.Tensor):
        return int(data.item())
    elif isinstance(data, list):
        return [convert_torch_to_int(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_torch_to_int(value) for key, value in data.items()}
    else:
        return data


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects"""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def generate_single_image(
        model,
        tokenizer,
        vqvae,
        prompt_text,
        height,
        width,
        timesteps,
        cfg_scale,
        temperature,
        seed,
        use_cache,
        cache_ratio,
        warmup_ratio,
        refresh_interval,
        device,
        en_region_sampling=False,
        en_heatmap=False,
        en_snapshot_step=8,
        en_snapshot_step_b=9,
        en_threshold_pair=(0.18, 0.48),
        en_region_steps=None,
        en_region_cache_start_step=10,
        en_alpha=0.45,
        en_heatmap_path=None,
        cache_prune_ratio=0.0,
        cache_prune_context_radius=2,
        cache_prune_anchor_stride=0,
        cache_prune_anchor_ratio=0.0,
        cache_prune_min_keep=0,
        return_stats=False,
):
    """生成单张图像"""
    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]

    # Set seed
    if seed > 0:
        setup_seed(seed)

    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(height, width)

    # Get prompt templates
    templates = create_prompt_templates()

    # Generate prompts
    input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)

    # Tokenize prompts
    con_prompt_token = tokenizer(input_prompt)["input_ids"]
    uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]

    # Build image mask prediction
    img_mask_token = add_break_line(
        [MASK] * seq_len,
        token_grid_height,
        token_grid_width,
        new_number=NEW_LINE
    )
    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

    # Create input tensors
    prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
    uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

    # Image start index
    code_start = len(con_prompt_token) + 2

    need_en_snapshots = en_region_sampling or en_heatmap
    if need_en_snapshots:
        if en_snapshot_step_b <= en_snapshot_step:
            raise ValueError("en_snapshot_step_b must be later than en_snapshot_step")
        en_snapshot_steps = [en_snapshot_step, en_snapshot_step_b]
        threshold_low, threshold_high = en_threshold_pair
    else:
        en_snapshot_steps = []
        threshold_low, threshold_high = en_threshold_pair

    if en_region_sampling:
        en_region_steps = list(en_region_steps or [4, 12, 20])
    else:
        en_region_steps = None

    en_context = {}

    def compute_en_region_labels(snapshots):
        pair_start = time.time()
        image_a = decode_vq_to_image(
            snapshots[en_snapshot_step],
            "/tmp/geneval_en_step_a.png",
            vae_ckpt=None,
            image_height=height,
            image_width=width,
            vqvae=vqvae,
        )
        image_b = decode_vq_to_image(
            snapshots[en_snapshot_step_b],
            "/tmp/geneval_en_step_b.png",
            vae_ckpt=None,
            image_height=height,
            image_width=width,
            vqvae=vqvae,
        )
        compute_start = time.time()
        labels, _ = en_tristate_segmentation(
            image_a,
            image_b,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            grid_size=(token_grid_height, token_grid_width),
        )
        en_context.update(
            image_a=image_a,
            labels=labels,
            compute_time=time.time() - compute_start,
            total_time=time.time() - pair_start,
        )
        return labels

    # Generate VQ tokens
    generation_result = generate_image(
        model,
        prompt_ids,
        seq_len=seq_len,
        newline_every=newline_every,
        timesteps=timesteps,
        temperature=temperature,
        cfg_scale=cfg_scale,
        uncon_ids=uncon_ids,
        code_start=code_start,
        use_cache=use_cache,
        cache_ratio=cache_ratio,
        refresh_interval=refresh_interval,
        warmup_ratio=warmup_ratio,
        snapshot_steps=en_snapshot_steps if need_en_snapshots else None,
        en_region_steps=en_region_steps if en_region_sampling else None,
        en_region_snapshot_steps=en_snapshot_steps if en_region_sampling else None,
        en_region_label_callback=compute_en_region_labels if en_region_sampling else None,
        en_region_cache_start_step=en_region_cache_start_step if en_region_sampling else None,
        cache_prune_ratio=cache_prune_ratio,
        cache_prune_context_radius=cache_prune_context_radius,
        cache_prune_anchor_stride=cache_prune_anchor_stride,
        cache_prune_anchor_ratio=cache_prune_anchor_ratio,
        cache_prune_min_keep=cache_prune_min_keep,
        return_stats=return_stats,
    )

    run_stats = None
    if need_en_snapshots:
        if return_stats:
            vq_tokens, snapshots, run_stats = generation_result
        else:
            vq_tokens, snapshots = generation_result
    else:
        if return_stats:
            vq_tokens, run_stats = generation_result
        else:
            vq_tokens = generation_result
        snapshots = {}


    # Decode VQ codes to image (返回 PIL Image 而不是保存)
    # 临时保存路径
    temp_path = "/tmp/temp_image.png"
    out_img = decode_vq_to_image(
        vq_tokens,
        temp_path,
        vae_ckpt=None,
        image_height=height,
        image_width=width,
        vqvae=vqvae
    )

    if en_heatmap:
        if not en_context:
            compute_en_region_labels(snapshots)
        if en_heatmap_path is not None:
            save_en_tristate_overlay(
                out_img,
                en_context["labels"],
                en_heatmap_path,
                alpha=en_alpha,
            )

    if return_stats:
        return out_img, run_stats
    return out_img


def main(args):
    print(f"🚀 启动 Lumina-DiMOO GenEval 评估")
    print(f"📁 模型路径: {args.checkpoint}")
    print(f"📊 评估数据: {args.metadata_file}")
    print(f"🎯 每个提示词生成样本数: {args.n_samples}")
    print(f"📐 图像尺寸: {args.height}x{args.width}")
    print(f"⚙️  CFG scale: {args.cfg_scale}")
    print(f"🔥 采样步数: {args.timesteps}")
    print(f"🌡️  温度: {args.temperature}")

    en_threshold_pair = parse_en_threshold_pair(args.en_threshold_pair)
    en_region_steps = parse_en_region_steps(args.en_region_steps)
    if args.en_region_sampling:
        finish_steps = [args.en_snapshot_step_b + step_count for step_count in en_region_steps]
        effective_timesteps = max(finish_steps) + 1
        print(
            "🧩 EN 区域独立采样: 开启 "
            f"step{args.en_snapshot_step}->step{args.en_snapshot_step_b}, "
            f"region_steps={en_region_steps}, finish_steps={finish_steps}, "
            f"effective_timesteps={effective_timesteps}, "
            f"cache_start_step={args.en_region_cache_start_step}"
        )
    else:
        effective_timesteps = args.timesteps
        print("🧩 EN 区域独立采样: 关闭，使用原始 Lumina cosine schedule")
    if effective_timesteps != args.timesteps:
        print(f"🔥 区域采样实际步数: {effective_timesteps} (忽略命令行 timesteps={args.timesteps})")
    if args.en_heatmap:
        print("🖼️  EN 可视化: 开启")
    print_generation_config(args, effective_timesteps, en_region_steps, finish_steps if args.en_region_sampling else None)

    # Set random seed
    setup_seed(args.seed)

    # Create output directory with timestamp.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = build_run_dir_name(args, timestamp, effective_timesteps)
    output_dir = os.path.join(args.output_root, output_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"💾 输出目录: {output_dir}")
    print(f"OUTPUT_DIR_FINAL={output_dir}", flush=True)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  设备: {device}")

    # Load model and tokenizer
    print("📥 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Load VQ-VAE
    print("📥 加载 VQ-VAE...")
    vqvae = VQModel.from_pretrained(args.checkpoint, subfolder="vqvae").to(device)
    vqvae.eval()

    print("✅ 模型加载完成")

    # Load GenEval metadata
    print(f"📖 加载评估数据...")
    with open(args.metadata_file, 'r') as fp:
        metadatas = [json.loads(line) for line in fp]

    print(f"📊 共有 {len(metadatas)} 个评估样本")

    # Process each sample
    total_time = 0
    total_cond_forward_time = 0.0
    total_uncond_forward_time = 0.0
    total_gpu_time = 0.0
    total_cond_forward_steps = 0
    total_uncond_forward_steps = 0
    executed_steps_values = []
    total_images_ok = 0
    total_images_all = len(metadatas) * args.n_samples
    for index, metadata in enumerate(tqdm(metadatas, desc="生成图像")):
        # Create output directory for this sample
        sample_output_dir = os.path.join(output_dir, f"{index:05d}")
        os.makedirs(sample_output_dir, exist_ok=True)

        # Create samples subdirectory
        samples_dir = os.path.join(sample_output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        if args.en_heatmap:
            en_heatmap_dir = os.path.join(sample_output_dir, "en_heatmaps")
            os.makedirs(en_heatmap_dir, exist_ok=True)
        else:
            en_heatmap_dir = None

        # Get prompt
        prompt = metadata['prompt']
        print(f"\n[{index + 1}/{len(metadatas)}] 处理: {prompt[:60]}...")

        # Save metadata
        metadata_converted = convert_torch_to_int(metadata)
        with open(os.path.join(sample_output_dir, "metadata.jsonl"), "w") as fp:
            json.dump(metadata_converted, fp, indent=4, default=default_dump)

        # Generate n_samples images
        sample_images = []
        sample_start_time = time.time()

        for sample_idx in range(args.n_samples):
            # Use different seed for each sample
            sample_seed = args.seed + index * args.n_samples + sample_idx

            try:
                # Generate image
                en_heatmap_path = (
                    os.path.join(en_heatmap_dir, f"{sample_idx:04d}_en_tristate_step{args.en_snapshot_step}_to_step{args.en_snapshot_step_b}.png")
                    if en_heatmap_dir is not None
                    else None
                )
                img, run_stats = generate_single_image(
                    model=model,
                    tokenizer=tokenizer,
                    vqvae=vqvae,
                    prompt_text=prompt,
                    height=args.height,
                    width=args.width,
                    timesteps=args.timesteps,
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature,
                    seed=sample_seed,
                    use_cache=args.use_cache,
                    cache_ratio=args.cache_ratio,
                    warmup_ratio=args.warmup_ratio,
                    refresh_interval=args.refresh_interval,
                    device=device,
                    en_region_sampling=args.en_region_sampling,
                    en_heatmap=args.en_heatmap,
                    en_snapshot_step=args.en_snapshot_step,
                    en_snapshot_step_b=args.en_snapshot_step_b,
                    en_threshold_pair=en_threshold_pair,
                    en_region_steps=en_region_steps,
                    en_region_cache_start_step=args.en_region_cache_start_step,
                    en_alpha=args.en_alpha,
                    en_heatmap_path=en_heatmap_path,
                    cache_prune_ratio=args.cache_prune_ratio,
                    cache_prune_context_radius=args.cache_prune_context_radius,
                    cache_prune_anchor_stride=args.cache_prune_anchor_stride,
                    cache_prune_anchor_ratio=args.cache_prune_anchor_ratio,
                    cache_prune_min_keep=args.cache_prune_min_keep,
                    return_stats=True,
                )

                # Save individual sample
                sample_path = os.path.join(samples_dir, f"{sample_idx:04d}.png")
                img.save(sample_path)

                # Collect for grid
                sample_images.append(img)

                executed_steps = int(run_stats.get("executed_steps", effective_timesteps)) if run_stats else effective_timesteps
                cond_forward_time_seconds = float(run_stats.get("cond_forward_time_seconds", 0.0)) if run_stats else 0.0
                uncond_forward_time_seconds = float(run_stats.get("uncond_forward_time_seconds", 0.0)) if run_stats else 0.0
                gpu_time_seconds = float(run_stats.get("gpu_time_seconds", 0.0) or 0.0) if run_stats else 0.0
                cond_forward_steps = int(run_stats.get("cond_forward_steps", 0)) if run_stats else 0
                uncond_forward_steps = int(run_stats.get("uncond_forward_steps", 0)) if run_stats else 0
                total_cond_forward_time += cond_forward_time_seconds
                total_uncond_forward_time += uncond_forward_time_seconds
                total_gpu_time += gpu_time_seconds
                total_cond_forward_steps += cond_forward_steps
                total_uncond_forward_steps += uncond_forward_steps
                executed_steps_values.append(executed_steps)
                total_images_ok += 1

                print(f"  ✅ 样本 {sample_idx + 1}/{args.n_samples} 已生成 ({executed_steps} steps)")

            except Exception as e:
                print(f"  ❌ 样本 {sample_idx + 1} 生成失败: {e}")
                # Create a blank image as placeholder
                blank_img = Image.new('RGB', (args.width, args.height), color='gray')
                sample_path = os.path.join(samples_dir, f"{sample_idx:04d}.png")
                blank_img.save(sample_path)
                sample_images.append(blank_img)

        # Create grid image
        if sample_images:
            # Calculate grid layout
            grid_cols = int(np.ceil(np.sqrt(args.n_samples)))
            grid_rows = int(np.ceil(args.n_samples / grid_cols))

            # Create grid
            grid_width = args.width * grid_cols
            grid_height = args.height * grid_rows
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')

            for idx, img in enumerate(sample_images):
                row = idx // grid_cols
                col = idx % grid_cols
                x = col * args.width
                y = row * args.height
                grid_img.paste(img, (x, y))

            grid_path = os.path.join(sample_output_dir, "grid.png")
            grid_img.save(grid_path)

        sample_time = time.time() - sample_start_time
        total_time += sample_time
        avg_time_per_image = sample_time / args.n_samples

        print(f"  ⏱️  本组用时: {sample_time:.2f}s (平均每张: {avg_time_per_image:.2f}s)")

    print(f"\n✅ 生成完成！")
    print(f"📊 总计生成: {len(metadatas) * args.n_samples} 张图像")
    print(f"⏱️  总用时: {total_time:.2f}s")
    print(f"📈 平均每张: {total_time / (len(metadatas) * args.n_samples):.2f}s")
    print(f"💾 结果保存至: {output_dir}")

    avg_time_per_image_all = total_time / total_images_all
    executed_steps_summary = {
        "total_images": total_images_all,
        "successful_images": total_images_ok,
        "failed_images": total_images_all - total_images_ok,
        "avg_executed_steps": (sum(executed_steps_values) / len(executed_steps_values)) if executed_steps_values else None,
        "min_executed_steps": min(executed_steps_values) if executed_steps_values else None,
        "max_executed_steps": max(executed_steps_values) if executed_steps_values else None,
    }
    timing_summary = {
        "avg_generation_time_per_image_seconds": avg_time_per_image_all,
        "avg_executed_steps_per_image": executed_steps_summary["avg_executed_steps"],
        "avg_cond_forward_time_per_image_seconds": (total_cond_forward_time / total_images_ok) if total_images_ok else None,
        "avg_uncond_forward_time_per_image_seconds": (total_uncond_forward_time / total_images_ok) if total_images_ok else None,
        "avg_gpu_time_per_image_seconds": (total_gpu_time / total_images_ok) if total_images_ok else None,
        "avg_cond_forward_time_per_step_seconds": (total_cond_forward_time / total_cond_forward_steps) if total_cond_forward_steps else None,
        "avg_uncond_forward_time_per_step_seconds": (total_uncond_forward_time / total_uncond_forward_steps) if total_uncond_forward_steps else None,
        "total_cond_forward_steps": total_cond_forward_steps,
        "total_uncond_forward_steps": total_uncond_forward_steps,
    }

    print(f"📉 实际步数: avg={executed_steps_summary['avg_executed_steps']}, min={executed_steps_summary['min_executed_steps']}, max={executed_steps_summary['max_executed_steps']}")
    print(f"[Timing] cond/img={timing_summary['avg_cond_forward_time_per_image_seconds']}, uncond/img={timing_summary['avg_uncond_forward_time_per_image_seconds']}, gpu/img={timing_summary['avg_gpu_time_per_image_seconds']}, cond/step={timing_summary['avg_cond_forward_time_per_step_seconds']}")

    # Save generation parameters
    params = {
        'checkpoint': args.checkpoint,
        'metadata_file': args.metadata_file,
        'output_root': args.output_root,
        'output_dir_prefix': args.output_dir_prefix,
        'n_samples': args.n_samples,
        'height': args.height,
        'width': args.width,
        'timesteps': args.timesteps,
        'effective_timesteps': effective_timesteps,
        'cfg_scale': args.cfg_scale,
        'temperature': args.temperature,
        'seed': args.seed,
        'use_cache': args.use_cache,
        'cache_ratio': args.cache_ratio,
        'warmup_ratio': args.warmup_ratio,
        'refresh_interval': args.refresh_interval,
        'en_region_sampling': args.en_region_sampling,
        'en_heatmap': args.en_heatmap,
        'en_snapshot_step': args.en_snapshot_step,
        'en_snapshot_step_b': args.en_snapshot_step_b,
        'en_threshold_pair': args.en_threshold_pair,
        'en_region_steps': args.en_region_steps,
        'en_region_cache_start_step': args.en_region_cache_start_step,
        'en_alpha': args.en_alpha,
        'cache_prune_ratio': args.cache_prune_ratio,
        'cache_prune_context_radius': args.cache_prune_context_radius,
        'cache_prune_anchor_stride': args.cache_prune_anchor_stride,
        'cache_prune_anchor_ratio': args.cache_prune_anchor_ratio,
        'cache_prune_min_keep': args.cache_prune_min_keep,
        'total_samples': len(metadatas),
        'total_images': len(metadatas) * args.n_samples,
        'total_time': total_time,
        'avg_time_per_image': avg_time_per_image_all,
        'executed_steps_summary': executed_steps_summary,
        'timing_summary': timing_summary,
        'rope_encoding_mode': 'original',
        'timestamp': timestamp
    }

    with open(os.path.join(output_dir, "generation_params.json"), "w") as f:
        json.dump(params, f, indent=4)

    print(f"📝 参数已保存至: {os.path.join(output_dir, 'generation_params.json')}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

