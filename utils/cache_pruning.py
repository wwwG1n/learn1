# -*- coding: utf-8 -*-
"""KV-cache pruning helpers for Lumina-DiMOO adaptive token sampling.

After region 0 (converged) and region 1 (about-to-converge) tokens have
finished sampling, this module prunes their KV-cache entries using a
Chebyshev distance-decay policy: tokens farther from region 2 (target,
still being sampled) are pruned first.

The pruning reduces the sequence dimension of the KV cache so that
subsequent forward passes for region 2 run on a shorter sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CachePrunePlan:
    """Describes which positions to keep after cache pruning."""
    keep_indices: torch.LongTensor      # full-sequence positions to keep
    position_ids: torch.LongTensor      # original absolute positions (for RoPE)
    pruned_count: int                   # number of tokens pruned
    total_before: int                   # sequence length before pruning


def _dilate_grid(mask: torch.BoolTensor, radius: int) -> torch.BoolTensor:
    """Dilate a 2D boolean grid by `radius` using 4-connected neighbours."""
    if radius <= 0:
        return mask.clone()
    out = mask.clone()
    cur = mask.clone()
    for _ in range(radius):
        nxt = cur.clone()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        out |= nxt
        cur = nxt
    return out


def _anchor_mask(
    h: int,
    w: int,
    *,
    stride: int,
    ratio: float,
    device: torch.device,
) -> torch.BoolTensor:
    """Deterministic anchor grid that survives pruning."""
    if stride <= 0 and ratio <= 0:
        return torch.zeros((h, w), dtype=torch.bool, device=device)

    rows = torch.arange(h, device=device).view(h, 1)
    cols = torch.arange(w, device=device).view(1, w)
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    if stride > 0:
        mask |= ((rows % stride) == 0) & ((cols % stride) == 0)
    if ratio > 0:
        threshold = int(max(0.0, min(1.0, ratio)) * 1000)
        hashed = (rows * 131 + cols * 197 + 17) % 1000
        mask |= hashed < threshold
    return mask


def _chebyshev_distance_to_target(target_grid: torch.BoolTensor) -> torch.LongTensor:
    """Per-cell Chebyshev distance to the nearest True cell via iterative 8-neighbour dilation."""
    h, w = target_grid.shape
    device = target_grid.device
    max_iter = h + w
    dist = torch.full((h, w), max_iter, dtype=torch.long, device=device)
    dist[target_grid] = 0
    if not target_grid.any():
        return dist

    cur = target_grid.clone()
    for d in range(1, max_iter):
        nxt = cur.clone()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt[1:, 1:] |= cur[:-1, :-1]
        nxt[1:, :-1] |= cur[:-1, 1:]
        nxt[:-1, 1:] |= cur[1:, :-1]
        nxt[:-1, :-1] |= cur[1:, 1:]
        new_cells = nxt & (~cur)
        if not new_cells.any():
            break
        dist[new_cells] = d
        cur = nxt
    return dist


def build_cache_prune_plan(
    x: torch.LongTensor,
    *,
    en_region_labels: torch.LongTensor,
    position_to_vq: torch.LongTensor,
    mask_token_id: int,
    code_start: int,
    newline_id: int,
    grid_h: int,
    grid_w: int,
    prune_ratio: float = 0.5,
    context_radius: int = 2,
    anchor_stride: int = 0,
    anchor_ratio: float = 0.0,
    min_keep: int = 0,
) -> CachePrunePlan:
    """Build a cache pruning plan based on distance-decay from region 2.

    Parameters
    ----------
    x : (1, T) current token sequence
    en_region_labels : (seq_len,) labels 0/1/2 for each VQ token
    position_to_vq : (T,) mapping from full-sequence position to VQ index (-1 for non-VQ)
    mask_token_id : mask token id
    code_start : index where image tokens begin in the full sequence
    newline_id : newline token id
    grid_h, grid_w : token grid dimensions
    prune_ratio : fraction of candidate tokens to prune
    context_radius : boundary protection radius around target
    anchor_stride : deterministic anchor grid stride
    anchor_ratio : random anchor ratio
    min_keep : minimum number of candidate tokens to keep
    """
    device = x.device
    total_len = x.shape[1]

    # Identify VQ token positions in the full sequence (excluding newlines)
    body = x[0, code_start:-2]
    valid_mask = (body != newline_id)
    valid_positions = valid_mask.nonzero(as_tuple=False).view(-1) + code_start
    assert valid_positions.numel() == grid_h * grid_w, (
        f"VQ token count {valid_positions.numel()} != grid {grid_h}x{grid_w}"
    )

    # Build grid-level masks from EN region labels
    # region 2 = target (still being sampled)
    # region 0+1 = candidates for pruning (already finished)
    target_grid = (en_region_labels == 2).reshape(grid_h, grid_w)
    candidate_region_grid = (
        (en_region_labels == 0) | (en_region_labels == 1)
    ).reshape(grid_h, grid_w)

    # Only prune committed (non-mask) tokens in region 0+1
    image_ids = x[0, valid_positions]
    committed_grid = (image_ids != mask_token_id).reshape(grid_h, grid_w)
    prune_candidate_grid = candidate_region_grid & committed_grid

    # Boundary protection: tokens close to target region are kept
    boundary = _dilate_grid(target_grid, context_radius) & prune_candidate_grid

    # Anchor protection
    anchors = _anchor_mask(
        grid_h, grid_w, stride=anchor_stride, ratio=anchor_ratio, device=device
    ) & prune_candidate_grid

    # Actual candidates = prune_candidate minus boundary minus anchors
    candidate = prune_candidate_grid & (~boundary) & (~anchors)
    n_candidate = int(candidate.sum().item())
    n_region_01 = int(prune_candidate_grid.sum().item())

    # Compute prune count
    prune_target = int(round(n_region_01 * float(max(0.0, min(1.0, prune_ratio)))))
    available_for_prune = max(0, n_candidate - max(0, int(min_keep)))
    prune_actual = max(0, min(prune_target, available_for_prune))

    # Distance-decay: prune tokens farthest from region 2 first
    pruned_grid = torch.zeros_like(candidate)
    if prune_actual > 0 and n_candidate > 0:
        dist = _chebyshev_distance_to_target(target_grid)
        big = dist.max().item() + 10
        score = torch.where(candidate, dist, torch.full_like(dist, -big))
        flat_score = score.reshape(-1)
        topk = torch.topk(flat_score, k=prune_actual)
        prune_flat = torch.zeros_like(flat_score, dtype=torch.bool)
        prune_flat[topk.indices] = True
        pruned_grid = prune_flat.view(grid_h, grid_w) & candidate

    # Build full-sequence keep mask
    keep_image_grid = ~pruned_grid
    keep_full = torch.ones(total_len, dtype=torch.bool, device=device)
    image_keep_flat = keep_image_grid.reshape(-1)
    keep_full[valid_positions] = image_keep_flat

    keep_indices = keep_full.nonzero(as_tuple=False).view(-1).to(torch.long)
    position_ids = keep_indices.clone()

    return CachePrunePlan(
        keep_indices=keep_indices,
        position_ids=position_ids,
        pruned_count=prune_actual,
        total_before=total_len,
    )


def prune_model_cache(model, plan: CachePrunePlan, cat: str = 'cond') -> None:
    """Physically shrink KV cache tensors in all layers according to the plan.

    After this call, cache tensors have shape (B, new_seq_len, D) where
    new_seq_len = plan.keep_indices.numel().
    """
    keep = plan.keep_indices

    # Prune per-block KV cache
    blocks = model.model.transformer.blocks
    for block in blocks:
        if cat in block.cache['k']:
            # cache shape: (B, T, D) -> (B, new_T, D)
            k_cache = block.cache['k'][cat]
            v_cache = block.cache['v'][cat]
            block.cache['k'][cat] = k_cache[:, keep, :]
            block.cache['v'][cat] = v_cache[:, keep, :]

    # Prune logit cache
    if cat in model.model.logit_cache:
        logit_cache = model.model.logit_cache[cat]
        model.model.logit_cache[cat] = logit_cache[:, keep, :]


def prune_both_caches(model, plan: CachePrunePlan) -> None:
    """Prune both 'cond' and 'uncond' caches using the same plan."""
    prune_model_cache(model, plan, cat='cond')
    prune_model_cache(model, plan, cat='uncond')
