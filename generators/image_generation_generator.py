# -*- coding: utf-8 -*-
"""
Image generation generator
"""
import torch
import math
from typing import Callable, Optional
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk
from model import LLaDAForMultiModalGeneration


@torch.no_grad()
def generate_image(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    use_cache=False,
    cache_ratio=0.9,
    refresh_interval=5,
    warmup_ratio=0.3,
    snapshot_steps: Optional[list] = None,
    return_stats: bool = False,
    en_region_steps: Optional[list] = None,
    en_region_snapshot_steps: Optional[list] = None,
    en_region_label_callback: Optional[Callable[[dict], object]] = None,
    en_region_cache_start_step: Optional[int] = None,
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens
    
    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: CFG scale
        uncon_ids: Unconditional input
        code_start: Image token satrt index
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator
        en_region_steps: Remaining sampling steps for EN labels 0/1/2
        en_region_snapshot_steps: Zero-based snapshot steps used to compute EN labels
        en_region_label_callback: Callback that receives snapshots and returns EN labels
        en_region_cache_start_step: First zero-based step that may use cache in EN region sampling
    
    Returns:
        Final VQ codes (1, seq_len)
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt
    
    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    runtime_use_cache = use_cache
    if isinstance(model, LLaDAForMultiModalGeneration):
        model.caching(runtime_use_cache)
    else:  # DDP
        model.module.caching(runtime_use_cache)

    if en_region_steps is not None:
        if len(en_region_steps) != 3:
            raise ValueError(f"en_region_steps must contain 3 integers, got {en_region_steps}")
        en_region_steps = [int(step_count) for step_count in en_region_steps]
        if any(step_count < 0 for step_count in en_region_steps):
            raise ValueError(f"en_region_steps must be non-negative, got {en_region_steps}")
        if en_region_snapshot_steps is None or len(en_region_snapshot_steps) != 2:
            raise ValueError("en_region_snapshot_steps must contain the two zero-based EN snapshot steps")
        en_region_snapshot_steps = [int(step_no) for step_no in en_region_snapshot_steps]
        if en_region_snapshot_steps[1] <= en_region_snapshot_steps[0]:
            raise ValueError(f"EN snapshot steps must be increasing, got {en_region_snapshot_steps}")
        timesteps = en_region_snapshot_steps[1] + max(en_region_steps) + 1
        if en_region_label_callback is None:
            raise ValueError("en_region_label_callback is required when en_region_steps is set")

    if en_region_steps is not None and en_region_cache_start_step is not None:
        en_region_cache_start_step = int(en_region_cache_start_step)
        if en_region_cache_start_step < 1:
            raise ValueError("en_region_cache_start_step must be >= 1 so step0 can initialize cache masks")
        if en_region_cache_start_step > timesteps:
            raise ValueError(
                f"en_region_cache_start_step={en_region_cache_start_step} exceeds effective timesteps={timesteps}"
            )
        warmup_step = en_region_cache_start_step - 1
    else:
        warmup_step = int(timesteps * warmup_ratio)
    refresh_steps = torch.zeros(timesteps, dtype=torch.bool)
    snapshot_steps = set(snapshot_steps or [])
    snapshots = {}
    en_region_labels = None
    en_region_base_step = None
    en_region_initial_counts = None

    # Initialize compute masks for token cache
    cond_to_compute_mask = None
    uncond_to_compute_mask = None

    def extract_vq_ids(tokens: torch.LongTensor) -> torch.LongTensor:
        vq_ids = tokens[0, code_start:-2]
        return vq_ids[vq_ids != newline_id].view(1, seq_len).clone()

    position_to_vq = None
    if en_region_steps is not None:
        image_positions = torch.arange(code_start, x.size(1) - 2, device=device)
        vq_positions = image_positions[x[0, code_start:-2] != newline_id]
        if vq_positions.numel() != seq_len:
            raise ValueError(f"Expected {seq_len} VQ token positions, got {vq_positions.numel()}")
        position_to_vq = torch.full((x.size(1),), -1, dtype=torch.long, device=device)
        position_to_vq[vq_positions] = torch.arange(seq_len, dtype=torch.long, device=device)

    def select_region_masks(
        flat_idx: torch.LongTensor,
        conf: torch.Tensor,
        *,
        step: int,
    ) -> torch.BoolTensor:
        current_region_labels = en_region_labels[position_to_vq[flat_idx]]
        local_step = step - en_region_base_step
        mask_sel_flat = torch.zeros(flat_idx.numel(), dtype=torch.bool, device=device)

        for region_id, region_step_count in enumerate(en_region_steps):
            region_positions = (current_region_labels == region_id).nonzero(as_tuple=False).view(-1)
            region_unknown_count = region_positions.numel()
            if region_unknown_count == 0:
                continue

            if region_step_count <= 0 or local_step >= region_step_count:
                keep_count = 0
            else:
                frac = math.cos(0.5 * math.pi * (local_step / region_step_count))
                keep_count = int(math.floor(int(en_region_initial_counts[region_id].item()) * frac))
                keep_count = max(1, keep_count)
                keep_count = min(keep_count, region_unknown_count)

            if keep_count <= 0:
                continue
            if keep_count >= region_unknown_count:
                mask_sel_flat[region_positions] = True
                continue

            region_conf = conf[:, region_positions]
            keep_tensor = torch.tensor([keep_count], dtype=torch.long, device=device)
            region_mask = mask_by_random_topk(
                keep_tensor,
                region_conf,
                temperature=temperature,
                generator=generator,
            ).view(-1)
            mask_sel_flat[region_positions[region_mask]] = True

        return mask_sel_flat.view(1, -1)

    for step in range(timesteps):
        if not runtime_use_cache or step <= warmup_step or (step-warmup_step) % refresh_interval == 0:
            refresh_steps[step] = True
    compute_ratio = 1 - cache_ratio

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    cond_forward_time_ms = 0.0
    uncond_forward_time_ms = 0.0
    cond_forward_steps = 0
    uncond_forward_steps = 0
    gpu_total_time_seconds = None
    early_exit_step = None
    use_cuda_timing = torch.cuda.is_available() and str(device).startswith("cuda")
    if use_cuda_timing:
        gpu_total_start = torch.cuda.Event(enable_timing=True)
        gpu_total_end = torch.cuda.Event(enable_timing=True)
        gpu_total_start.record()

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            if early_exit_step is None:
                early_exit_step = step
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        if runtime_use_cache and step and refresh_steps[step]:
            if isinstance(model, LLaDAForMultiModalGeneration):
                model.empty_cache()
            else:  # DDP
                model.module.empty_cache()

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)

            if use_cuda_timing:
                cond_start_event = torch.cuda.Event(enable_timing=True)
                cond_end_event = torch.cuda.Event(enable_timing=True)
                cond_start_event.record()
            cond_logits = model(x, infer=True,
                    cat='cond', use_cache=runtime_use_cache, 
                    to_compute_mask=cond_to_compute_mask if not refresh_steps[step] else None,
                ).logits[..., vocab_offset : vocab_offset + codebook_size]
            if use_cuda_timing:
                cond_end_event.record()
            cond_mask_logits = cond_logits[vq_mask].view(B, -1, codebook_size)
            if use_cuda_timing:
                uncond_start_event = torch.cuda.Event(enable_timing=True)
                uncond_end_event = torch.cuda.Event(enable_timing=True)
                uncond_start_event.record()
            uncond_logits = model(uncond, infer=True,
                    cat='uncond', use_cache=runtime_use_cache, 
                    to_compute_mask=uncond_to_compute_mask if not refresh_steps[step] else None,
                ).logits[..., vocab_offset : vocab_offset + codebook_size]
            if use_cuda_timing:
                uncond_end_event.record()
            uncond_mask_logits = uncond_logits[uncond_vq_mask].view(B, -1, codebook_size)
            if use_cuda_timing:
                torch.cuda.synchronize()
                cond_forward_time_ms += cond_start_event.elapsed_time(cond_end_event)
                uncond_forward_time_ms += uncond_start_event.elapsed_time(uncond_end_event)
            cond_forward_steps += 1
            uncond_forward_steps += 1
            logits = (1 + cfg_scale) * cond_mask_logits - cfg_scale * uncond_mask_logits
        else:
            if use_cuda_timing:
                cond_start_event = torch.cuda.Event(enable_timing=True)
                cond_end_event = torch.cuda.Event(enable_timing=True)
                cond_start_event.record()
            logits = model(
                x, infer=True
            ).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            if use_cuda_timing:
                cond_end_event.record()
                torch.cuda.synchronize()
                cond_forward_time_ms += cond_start_event.elapsed_time(cond_end_event)
            cond_forward_steps += 1

        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        if step in snapshot_steps:
            snapshots[step] = extract_vq_ids(x)

        if en_region_labels is not None and step > en_region_base_step:
            mask_sel = select_region_masks(flat_idx, conf, step=step)
        else:
            mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        if (
            en_region_steps is not None
            and en_region_labels is None
            and step == en_region_snapshot_steps[1]
        ):
            missing_steps = [step_no for step_no in en_region_snapshot_steps if step_no not in snapshots]
            if missing_steps:
                raise RuntimeError(f"EN snapshots were not captured for steps {missing_steps}")

            labels = en_region_label_callback(snapshots)
            en_region_labels = torch.as_tensor(labels, dtype=torch.long, device=device).view(-1)
            if en_region_labels.numel() != seq_len:
                raise ValueError(
                    f"EN labels must contain {seq_len} tokens, got {en_region_labels.numel()}"
                )
            if en_region_labels.min().item() < 0 or en_region_labels.max().item() > 2:
                raise ValueError("EN labels must use values 0, 1, and 2")

            current_flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
            current_vq_idx = position_to_vq[current_flat_idx]
            current_labels = en_region_labels[current_vq_idx]
            en_region_initial_counts = torch.bincount(current_labels, minlength=3)[:3].long()
            en_region_base_step = step
            print(
                f"[EN] base_step={en_region_base_step} counts={en_region_initial_counts.tolist()}",
                flush=True,
            )

        if runtime_use_cache and step < timesteps - 1 and not refresh_steps[step+1]:
            cond_conf = cond_logits.max(dim=-1)[0]
            cond_conf_threshold = torch.quantile(cond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            cond_to_compute_mask = cond_conf <= cond_conf_threshold

            uncond_conf = uncond_logits.max(dim=-1)[0]
            uncond_conf_threshold = torch.quantile(uncond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            uncond_to_compute_mask = uncond_conf <= uncond_conf_threshold
            

    # Remove newline tokens
    vq_ids = extract_vq_ids(x)

    if use_cuda_timing:
        gpu_total_end.record()
        torch.cuda.synchronize()
        gpu_total_time_seconds = gpu_total_start.elapsed_time(gpu_total_end) / 1000.0

    run_stats = {
        "executed_steps": int(early_exit_step) if early_exit_step is not None else int(timesteps),
        "total_timesteps": int(timesteps),
        "cond_forward_time_seconds": cond_forward_time_ms / 1000.0,
        "uncond_forward_time_seconds": uncond_forward_time_ms / 1000.0,
        "gpu_time_seconds": gpu_total_time_seconds,
        "cond_forward_steps": int(cond_forward_steps),
        "uncond_forward_steps": int(uncond_forward_steps),
        "avg_cond_forward_time_per_step_seconds": (cond_forward_time_ms / 1000.0 / max(1, cond_forward_steps)),
        "avg_uncond_forward_time_per_step_seconds": (uncond_forward_time_ms / 1000.0 / max(1, uncond_forward_steps)),
        "early_exit_step": early_exit_step,
    }

    if snapshot_steps and return_stats:
        return vq_ids, snapshots, run_stats
    if snapshot_steps:
        return vq_ids, snapshots
    if return_stats:
        return vq_ids, run_stats
    return vq_ids
