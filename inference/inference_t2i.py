# -*- coding: utf-8 -*-
"""
Text-to-image inference script
"""
import os
import json
import argparse
import time
import torch
from transformers import AutoConfig, AutoTokenizer
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line, encode_img_with_paint
from utils.en_segmentation import en_binary_segmentation, save_en_overlay, save_en_score_heatmap
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates



def main():
    parser = argparse.ArgumentParser(description="Text-to-image inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--painting_mode", type=str, default=None, help="Inpainting for image-inpainting task & outpainting for imahe-extrapolation task")
    parser.add_argument("--painting_image", type=str, default=None, help="Inpainting & outpainting image path")
    parser.add_argument("--mask_h_ratio", type=float, default=1, help="Height ratio for mask region of In/Out paint task")
    parser.add_argument("--mask_w_ratio", type=float, default=0.2, help="Width ratio for mask region of In/Out paint task")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_text_to_image", help="Output directory")
    parser.add_argument("--use-cache", action='store_true', help="Enable caching for faster inference")
    parser.add_argument("--cache_ratio", type=float, default=0.9, help="Ratio of reused tokens, in (0,1); the higher the faster")
    parser.add_argument("--warmup_ratio", type=float, default=0.3, help="Warmup ratio for caching, in [0,1); the lower the faster")
    parser.add_argument("--refresh_interval", type=int, default=5, help="Refresh all cache every `refresh_interval` steps, in (1, timesteps-int(warmup_ratio*timesteps)-1]; the higher the faster")
    parser.add_argument("--en_heatmap", action="store_true", help="Save EN binary segmentation heatmap overlay")
    parser.add_argument("--en_snapshot_step", type=int, default=None, help="1-based generation step used as early EN snapshot")
    parser.add_argument("--en_snapshot_step_b", type=int, default=None, help="Optional later EN snapshot step; compare step -> step_b instead of step -> final")
    parser.add_argument("--en_alpha", type=float, default=0.45, help="Red overlay alpha for EN heatmap")
    
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
    
    # Create Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    if args.en_heatmap and en_snapshot_step is None:
        en_snapshot_step = max(1, args.timesteps // 2)
    en_snapshot_steps = []
    if args.en_heatmap:
        en_snapshot_steps.append(en_snapshot_step)
        if args.en_snapshot_step_b is not None:
            en_snapshot_steps.append(args.en_snapshot_step_b)
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
        snapshot_steps=en_snapshot_steps if args.en_heatmap else None,
    )
    if args.en_heatmap:
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
    print(f"[✓] Saved {save_path}")

    if args.en_heatmap:
        if en_snapshot_step not in snapshots:
            raise RuntimeError(f"EN snapshot step {en_snapshot_step} was not captured")
        if args.en_snapshot_step_b is not None and args.en_snapshot_step_b not in snapshots:
            raise RuntimeError(f"EN snapshot step {args.en_snapshot_step_b} was not captured")
        if args.en_snapshot_step_b is not None and args.en_snapshot_step_b <= en_snapshot_step:
            raise ValueError("--en_snapshot_step_b must be later than --en_snapshot_step")

        snapshot_path = save_path.replace(".png", f"_en_snapshot_step{en_snapshot_step}.png")

        snapshot_img = decode_vq_to_image(
            snapshots[en_snapshot_step],
            snapshot_path,
            vae_ckpt=args.vae_ckpt,
            image_height=height,
            image_width=width,
            vqvae=vqvae,
        )
        snapshot_img.save(snapshot_path)

        if args.en_snapshot_step_b is None:
            compare_img = out_img
            compare_label = f"step{en_snapshot_step}"
            print(f"[✓] Saved EN snapshot {snapshot_path}")
        else:
            snapshot_b_path = save_path.replace(".png", f"_en_snapshot_step{args.en_snapshot_step_b}.png")
            compare_img = decode_vq_to_image(
                snapshots[args.en_snapshot_step_b],
                snapshot_b_path,
                vae_ckpt=args.vae_ckpt,
                image_height=height,
                image_width=width,
                vqvae=vqvae,
            )
            compare_img.save(snapshot_b_path)
            compare_label = f"step{en_snapshot_step}_to_step{args.en_snapshot_step_b}"
            print(f"[✓] Saved EN snapshot {snapshot_path}")
            print(f"[✓] Saved EN snapshot {snapshot_b_path}")

        heatmap_path = save_path.replace(".png", f"_en_heatmap_{compare_label}.png")
        score_heatmap_path = save_path.replace(".png", f"_en_score_heatmap_{compare_label}.png")
        mask_path = save_path.replace(".png", f"_en_mask_{compare_label}.npy")
        score_path = save_path.replace(".png", f"_en_scores_{compare_label}.npy")
        en_mask, en_scores = en_binary_segmentation(snapshot_img, compare_img)
        import numpy as np

        np.save(mask_path, en_mask)
        np.save(score_path, en_scores.astype(np.float32))
        save_en_overlay(compare_img, en_mask, heatmap_path, alpha=args.en_alpha)
        save_en_score_heatmap(compare_img, en_scores, score_heatmap_path)
        print(f"[✓] Saved EN mask {mask_path}")
        print(f"[✓] Saved EN scores {score_path}")
        print(f"[✓] Saved EN heatmap {heatmap_path}")
        print(f"[✓] Saved EN score heatmap {score_heatmap_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Time: {elapsed_time:.2f}s")
       


if __name__ == '__main__':
    main()
