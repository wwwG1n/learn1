# -*- coding: utf-8 -*-
"""
Text-to-image inference script
"""
import os
import argparse
import time
from datetime import datetime
import torch
from transformers import AutoTokenizer
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line, encode_img_with_paint
from utils.en_segmentation import en_tristate_segmentation, save_en_tristate_overlay
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates


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


def build_run_dir_name(args, height, width):
    if args.en_region_sampling:
        en_region_steps = parse_en_region_steps(args.en_region_steps)
        finish_steps = [args.en_snapshot_step_b + step_count for step_count in en_region_steps]
        effective_timesteps = max(finish_steps) + 1
        en_part = (
            f"en1_s{args.en_snapshot_step}-{args.en_snapshot_step_b}"
            f"_thr{args.en_threshold_pair.replace(':', '-')}"
            f"_rs{'-'.join(str(step) for step in en_region_steps)}"
            f"_cs{args.en_region_cache_start_step}"
        )
    else:
        effective_timesteps = args.timesteps
        en_part = "en0"

    cache_part = (
        f"cache{int(args.use_cache)}"
        f"_cr{format_dir_value(args.cache_ratio)}"
        f"_wu{format_dir_value(args.warmup_ratio)}"
        f"_ri{args.refresh_interval}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"t2i_{timestamp}"
        f"_h{height}_w{width}"
        f"_t{args.timesteps}_eff{effective_timesteps}"
        f"_cfg{format_dir_value(args.cfg_scale)}"
        f"_temp{format_dir_value(args.temperature)}"
        f"_seed{args.seed}"
        f"_{cache_part}_{en_part}"
    )


def print_generation_config(args, height, width):
    print("Generation config:")
    print(f"  image_size={height}x{width}")
    print(f"  timesteps={args.timesteps}")
    print(f"  cfg_scale={args.cfg_scale}")
    print(f"  temperature={args.temperature}")
    print(f"  seed={args.seed}")
    print(
        "  cache="
        f"{args.use_cache} cache_ratio={args.cache_ratio} "
        f"warmup_ratio={args.warmup_ratio} refresh_interval={args.refresh_interval}"
    )
    if args.en_region_sampling:
        en_region_steps = parse_en_region_steps(args.en_region_steps)
        finish_steps = [args.en_snapshot_step_b + step_count for step_count in en_region_steps]
        effective_timesteps = max(finish_steps) + 1
        print("  en_region_sampling=True")
        print(f"  en_snapshots=step{args.en_snapshot_step}->step{args.en_snapshot_step_b}")
        print(f"  en_threshold_pair={args.en_threshold_pair}")
        print(f"  en_region_steps={en_region_steps}")
        print(f"  en_finish_steps={finish_steps}")
        print(f"  effective_timesteps={effective_timesteps}")
        print(f"  en_region_cache_start_step={args.en_region_cache_start_step}")
    else:
        print("  en_region_sampling=False")


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-image inference",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  # 1) 原始 Lumina 采样：显式关闭区域独立采样
  CUDA_VISIBLE_DEVICES=0 conda run -n lumina_dimoo python inference/inference_t2i.py \\
    --checkpoint /mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights \\
    --vae_ckpt /mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights \\
    --prompt "a glass greenhouse filled with tropical plants at sunrise" \\
    --height 1024 --width 1024 --timesteps 32 \\
    --no-en_region_sampling \\
    --output_dir debug_results/original_lumina

  # 2) 只保存 EN 图：采样仍是原始 Lumina
  CUDA_VISIBLE_DEVICES=0 conda run -n lumina_dimoo python inference/inference_t2i.py \\
    --checkpoint /mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights \\
    --vae_ckpt /mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights \\
    --prompt "a glass greenhouse filled with tropical plants at sunrise" \\
    --height 1024 --width 1024 --timesteps 32 \\
    --en_heatmap --no-en_region_sampling \\
    --output_dir debug_results/en_heatmap_only

  # 3) 默认区域独立采样：不额外保存 EN 图
  CUDA_VISIBLE_DEVICES=0 conda run -n lumina_dimoo python inference/inference_t2i.py \\
    --checkpoint /mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights \\
    --vae_ckpt /mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights \\
    --prompt "a glass greenhouse filled with tropical plants at sunrise" \\
    --height 1024 --width 1024 --timesteps 32 \\
    --en_region_steps "4,12,20" \\
    --output_dir debug_results/en_region_sampling

  # 4) 默认区域独立采样，同时保存 EN 图
  CUDA_VISIBLE_DEVICES=0 conda run -n lumina_dimoo python inference/inference_t2i.py \\
    --checkpoint /mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights \\
    --vae_ckpt /mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights \\
    --prompt "a glass greenhouse filled with tropical plants at sunrise" \\
    --height 1024 --width 1024 --timesteps 32 \\
    --en_heatmap --en_region_steps "4,12,20" \\
    --output_dir debug_results/en_region_sampling_with_heatmap
""",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--painting_mode", type=str, default=None, help="Inpainting for image-inpainting task & outpainting for imahe-extrapolation task")
    parser.add_argument("--painting_image", type=str, default=None, help="Inpainting & outpainting image path")
    parser.add_argument("--mask_h_ratio", type=float, default=1, help="Height ratio for mask region of In/Out paint task")
    parser.add_argument("--mask_w_ratio", type=float, default=0.2, help="Width ratio for mask region of In/Out paint task")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--timesteps", type=int, default=32, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_text_to_image", help="Output directory")
    parser.add_argument("--use-cache", action='store_true', help="Enable caching for faster inference")
    parser.add_argument("--cache_ratio", type=float, default=0.9, help="Ratio of reused tokens, in (0,1); the higher the faster")
    parser.add_argument("--warmup_ratio", type=float, default=0.3, help="Warmup ratio for caching, in [0,1); the lower the faster")
    parser.add_argument("--refresh_interval", type=int, default=5, help="Refresh all cache every `refresh_interval` steps, in (1, timesteps-int(warmup_ratio*timesteps)-1]; the higher the faster")
    parser.add_argument("--en_heatmap", action="store_true", help="Save EN step8->step9 segmentation visualization without changing sampling")
    parser.add_argument("--en_snapshot_step", type=int, default=8, help="Zero-based earlier EN snapshot step")
    parser.add_argument("--en_snapshot_step_b", type=int, default=9, help="Zero-based later EN snapshot step")
    parser.add_argument("--en_threshold_pair", type=str, default="0.18:0.48", help='EN tri-state thresholds, e.g. "0.18:0.48"')
    parser.add_argument("--en_alpha", type=float, default=0.45, help="Red overlay alpha for EN heatmap")
    parser.add_argument(
        "--en_region_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable EN region-independent sampling schedule (default: enabled; use --no-en_region_sampling for original Lumina)",
    )
    parser.add_argument("--en_region_steps", type=str, default="4,12,20", help="Remaining zero-based sampling steps for EN labels 0,1,2")
    parser.add_argument(
        "--en_region_cache_start_step",
        type=int,
        default=10,
        help="First zero-based step that may use cache when EN region sampling is enabled",
    )
    
    args = parser.parse_args()
    
    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
    BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]           # End of Image

    # Set Random seed
    if args.seed != 0:
        setup_seed(args.seed)
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    
    # Initial image parameters
    if args.painting_mode:
        img = Image.open(args.painting_image)
        width, height = img.size
    else:
        height = args.height
        width = args.width

    output_root = args.output_dir
    args.output_dir = os.path.join(output_root, build_run_dir_name(args, height, width))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output root: {args.output_dir}")
    print_generation_config(args, height, width)

    # Load VQ-VAE
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(height, width)
    
    print(f"Generate image size: {height}x{width}")
    print(f"Calculated VQ sequence length: {seq_len}")
    print(f"Tokens per line (newline_every): {newline_every}")
    
    # Get prompt templates
    templates = create_prompt_templates()

    # Get prompt
    prompt_text = args.prompt

    # Generate prompts using utility function
    input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)

    # build initial sequence
    con_prompt_token = tokenizer(input_prompt)["input_ids"]
    uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]
    
    # build image mask predition
    if args.painting_mode:
        img_mask_token, img_vis = encode_img_with_paint(img, vqvae=vqvae, mask_h_ratio=args.mask_h_ratio, mask_w_ratio=args.mask_w_ratio, mask_mode=args.painting_mode)
    else:
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number = NEW_LINE)
    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

    prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
    uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

    # image satrt index
    code_start = len(con_prompt_token) + 2 
    
    # Generate VQ tokens
    start_time = time.time()
    en_snapshot_step = args.en_snapshot_step
    en_snapshot_step_b = args.en_snapshot_step_b
    need_en_snapshots = args.en_heatmap or args.en_region_sampling
    if need_en_snapshots:
        if en_snapshot_step_b <= en_snapshot_step:
            raise ValueError("--en_snapshot_step_b must be later than --en_snapshot_step")
        threshold_low, threshold_high = parse_en_threshold_pair(args.en_threshold_pair)
        en_snapshot_steps = [en_snapshot_step, en_snapshot_step_b]
        en_region_context = {}
        if args.en_region_sampling:
            en_region_steps = parse_en_region_steps(args.en_region_steps)
        else:
            en_region_steps = None
    else:
        en_snapshot_steps = []
        en_region_steps = None
        en_region_context = {}

    def compute_en_region_labels(snapshots):
        pair_start_time = time.time()
        image_a = decode_vq_to_image(
            snapshots[en_snapshot_step],
            os.path.join(args.output_dir, "_en_step_a.png"),
            vae_ckpt=args.vae_ckpt,
            image_height=height,
            image_width=width,
            vqvae=vqvae,
        )
        image_b = decode_vq_to_image(
            snapshots[en_snapshot_step_b],
            os.path.join(args.output_dir, "_en_step_b.png"),
            vae_ckpt=args.vae_ckpt,
            image_height=height,
            image_width=width,
            vqvae=vqvae,
        )

        en_compute_start = time.time()
        overlay_labels, _ = en_tristate_segmentation(
            image_a,
            image_b,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            grid_size=(token_grid_height, token_grid_width),
        )
        en_compute_time = time.time() - en_compute_start

        en_region_context.update(
            image_a=image_a,
            image_b=image_b,
            overlay_labels=overlay_labels,
            en_compute_time=en_compute_time,
            pair_elapsed_time=time.time() - pair_start_time,
        )
        return overlay_labels

    if args.en_region_sampling:
        finish_steps = [en_snapshot_step_b + step_count for step_count in en_region_steps]
        effective_timesteps = max(finish_steps) + 1
        print(
            "EN region sampling: "
            f"zero_based_snapshots=step{en_snapshot_step}->step{en_snapshot_step_b} "
            f"region_steps={en_region_steps} finish_steps={finish_steps} "
            f"effective_timesteps={effective_timesteps} "
            f"cache_start_step={args.en_region_cache_start_step}"
        )
    else:
        print("Sampling mode: original Lumina cosine schedule")

    generate_start_time = time.time()
    generation_result = generate_image(
        model,
        prompt_ids,
        seq_len=seq_len,
        newline_every=newline_every,
        timesteps=args.timesteps,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        uncon_ids=uncon_ids,
        code_start=code_start,
        use_cache=args.use_cache,
        cache_ratio=args.cache_ratio,
        refresh_interval=args.refresh_interval,
        warmup_ratio=args.warmup_ratio,
        snapshot_steps=en_snapshot_steps if need_en_snapshots else None,
        en_region_steps=en_region_steps if args.en_region_sampling else None,
        en_region_snapshot_steps=en_snapshot_steps if args.en_region_sampling else None,
        en_region_label_callback=compute_en_region_labels if args.en_region_sampling else None,
        en_region_cache_start_step=args.en_region_cache_start_step if args.en_region_sampling else None,
    )
    if need_en_snapshots:
        vq_tokens, snapshots = generation_result
    else:
        vq_tokens = generation_result
        snapshots = {}
    
    # Generate filename
    words = prompt_text.split()
    filename_words = words[:10] if len(words) > 10 else words
    filename = "_".join(filename_words)
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    filename = f"{filename}_{height}x{width}_t{args.timesteps}_cfg{args.cfg_scale}_seed{args.seed}.png"
    save_path = os.path.join(args.output_dir, filename)
    
    # Decode VQ codes to PNG and save
    out_img = decode_vq_to_image(
        vq_tokens, save_path, 
        vae_ckpt=args.vae_ckpt, 
        image_height=height, 
        image_width=width,
        vqvae=vqvae
    )
    if args.painting_mode:
        w1, h1 = img_vis.size
        w2, h2 = out_img.size
        canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), "white")
        canvas.paste(img_vis, (0, 0))
        canvas.paste(out_img, (w1, 0))
        concat_path = save_path.replace(".png", "_concat.png")
        canvas.save(concat_path)
    else:
        out_img.save(save_path)
    generate_elapsed_time = time.time() - generate_start_time
    print(f"[✓] Saved {save_path}")

    if args.en_heatmap:
        if en_snapshot_step not in snapshots:
            raise RuntimeError(f"EN snapshot step {en_snapshot_step} was not captured")
        if en_snapshot_step_b not in snapshots:
            raise RuntimeError(f"EN snapshot step {en_snapshot_step_b} was not captured")

        if en_region_context:
            image_a = en_region_context["image_a"]
            en_labels = en_region_context["overlay_labels"]
            en_compute_time = en_region_context["en_compute_time"]
            pair_elapsed_time = en_region_context["pair_elapsed_time"]
        else:
            pair_start_time = time.time()
            image_a = decode_vq_to_image(
                snapshots[en_snapshot_step],
                save_path,
                vae_ckpt=args.vae_ckpt,
                image_height=height,
                image_width=width,
                vqvae=vqvae,
            )
            image_b = decode_vq_to_image(
                snapshots[en_snapshot_step_b],
                save_path,
                vae_ckpt=args.vae_ckpt,
                image_height=height,
                image_width=width,
                vqvae=vqvae,
            )

            en_compute_start = time.time()
            en_labels, _ = en_tristate_segmentation(
                image_a,
                image_b,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                grid_size=(token_grid_height, token_grid_width),
            )
            en_compute_time = time.time() - en_compute_start
            pair_elapsed_time = time.time() - pair_start_time
        region_counts = {
            "converged": int((en_labels == 0).sum()),
            "near_converged": int((en_labels == 1).sum()),
            "non_converged": int((en_labels == 2).sum()),
        }
        compare_label = f"step{en_snapshot_step}_to_step{en_snapshot_step_b}"
        heatmap_path = save_path.replace(".png", f"_en_tristate_{compare_label}.png")
        save_en_tristate_overlay(image_a, en_labels, heatmap_path, alpha=args.en_alpha)
        ratio = pair_elapsed_time / generate_elapsed_time if generate_elapsed_time > 0 else 0.0
        print(f"[✓] Saved EN visualization {heatmap_path}")
        print(
            f"ENTriStateTime {compare_label}: total={pair_elapsed_time:.4f}s "
            f"compute={en_compute_time:.4f}s generation={generate_elapsed_time:.4f}s "
            f"ratio={ratio:.4f} thresholds={threshold_low:.4f}:{threshold_high:.4f} "
            f"label_shape={en_labels.shape[0]}x{en_labels.shape[1]} "
            f"converged={region_counts['converged']} "
            f"near_converged={region_counts['near_converged']} "
            f"non_converged={region_counts['non_converged']}/{en_labels.size}"
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Time: {elapsed_time:.2f}s")
       


if __name__ == '__main__':
    main()
