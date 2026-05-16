# -*- coding: utf-8 -*-
"""
EN segmentation utilities.

The implementation follows ``47293_EN二值分割算法_数学描述.md`` and returns a
token-grid mask or label map based on token-level non-convergence scores.
"""
from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d, gaussian_filter, laplace, median_filter, uniform_filter


EPS = 1e-6
GRID_SIZE = (64, 64)
COLOR_PERCENTILES = (2, 98)
SSIM_WIN = 11
SSIM_SIGMA = 1.5
SSIM_RANGE = 1.0
SSIM_PERCENTILES = (2, 98)
FREQ_PERCENTILES = (2, 98)
PIX_PERCENTILES = (2, 98)
TOKEN_PRE_PERCENTILES = (1, 99)
POOL_PERCENTILE = 80
THRESHOLD = 0.15
MAX_ITER = 8
AREA_MAX = 96
BLOCK_SIZE = 2


def _as_rgb01(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _robust_normalize(x: np.ndarray, low: float, high: float) -> np.ndarray:
    q_low, q_high = np.percentile(x, [low, high])
    return np.clip((x - q_low) / (q_high - q_low + EPS), 0.0, 1.0).astype(np.float32)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    # sRGB D65 conversion, implemented locally to avoid adding a dependency.
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0)
    lab = np.empty_like(f, dtype=np.float32)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _gaussian_filter_window(x: np.ndarray, window_size: int, sigma: float) -> np.ndarray:
    radius = window_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    filtered = convolve1d(x, kernel, axis=0, mode="reflect")
    return convolve1d(filtered, kernel, axis=1, mode="reflect")


def _ssim_distance(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    c1 = (0.01 * SSIM_RANGE) ** 2
    c2 = (0.03 * SSIM_RANGE) ** 2

    mu_a = _gaussian_filter_window(gray_a, SSIM_WIN, SSIM_SIGMA)
    mu_b = _gaussian_filter_window(gray_b, SSIM_WIN, SSIM_SIGMA)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = _gaussian_filter_window(gray_a * gray_a, SSIM_WIN, SSIM_SIGMA) - mu_a2
    sigma_b2 = _gaussian_filter_window(gray_b * gray_b, SSIM_WIN, SSIM_SIGMA) - mu_b2
    sigma_ab = _gaussian_filter_window(gray_a * gray_b, SSIM_WIN, SSIM_SIGMA) - mu_ab

    ssim = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2) + EPS
    )
    return 1.0 - np.clip(ssim, -1.0, 1.0)


def _local_high_frequency_energy(gray_img: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(gray_img, sigma=1.0, mode="reflect")
    high = laplace(smoothed, mode="reflect")
    return uniform_filter(high * high, size=15, mode="nearest").astype(np.float32)


def _pixel_non_convergence(image_a: Image.Image, image_b: Image.Image) -> np.ndarray:
    rgb_a = _as_rgb01(image_a)
    rgb_b = _as_rgb01(image_b.resize(image_a.size, Image.BICUBIC))

    lab_a = gaussian_filter(_rgb_to_lab(rgb_a), sigma=(3.0, 3.0, 0.0), mode="reflect")
    lab_b = gaussian_filter(_rgb_to_lab(rgb_b), sigma=(3.0, 3.0, 0.0), mode="reflect")
    color_diff = _robust_normalize(np.linalg.norm(lab_a - lab_b, axis=-1), *COLOR_PERCENTILES)

    gray_a = _gray(rgb_a)
    gray_b = _gray(rgb_b)
    structure_diff = _robust_normalize(_ssim_distance(gray_a, gray_b), *SSIM_PERCENTILES)

    energy_a = _local_high_frequency_energy(gray_a)
    energy_b = _local_high_frequency_energy(gray_b)
    missing_detail = np.maximum(energy_b - energy_a, 0.0) / (energy_b + EPS)
    detail_diff = _robust_normalize(
        missing_detail * _robust_normalize(energy_b, *FREQ_PERCENTILES),
        *FREQ_PERCENTILES,
    )

    return _robust_normalize(0.20 * color_diff + 0.35 * structure_diff + 0.45 * detail_diff, *PIX_PERCENTILES)


def _token_scores(u_pix: np.ndarray, grid_size: Tuple[int, int], pool_percentile: float) -> np.ndarray:
    smooth = median_filter(_robust_normalize(u_pix, *TOKEN_PRE_PERCENTILES), size=3, mode="nearest")
    grid_h, grid_w = grid_size
    h, w = smooth.shape
    cell_h = h // grid_h
    cell_w = w // grid_w
    if cell_h <= 0 or cell_w <= 0:
        raise ValueError(f"Input image is too small for fixed EN grid {grid_h}x{grid_w}: got {h}x{w}")

    scores = np.zeros((grid_h, grid_w), dtype=np.float32)
    for i in range(grid_h):
        for j in range(grid_w):
            cell = smooth[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            scores[i, j] = np.percentile(cell, pool_percentile)
    return scores


def _repair_small_e_islands(labels: np.ndarray, area_max: int, max_iter: int) -> np.ndarray:
    labels = labels.astype(np.uint8, copy=True)
    h, w = labels.shape
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for _ in range(max_iter):
        visited = np.zeros_like(labels, dtype=bool)
        changed = False
        for y in range(h):
            for x in range(w):
                if labels[y, x] != 0 or visited[y, x]:
                    continue

                comp = []
                boundary_values = []
                queue = deque([(y, x)])
                visited[y, x] = True
                while queue:
                    cy, cx = queue.popleft()
                    comp.append((cy, cx))
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if labels[ny, nx] == 0:
                            if not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                        else:
                            boundary_values.append(labels[ny, nx])

                if 0 < len(comp) <= area_max and boundary_values and all(v == 1 for v in boundary_values):
                    for cy, cx in comp:
                        labels[cy, cx] = 1
                    changed = True
        if not changed:
            break
    return labels


def _block_finalize(labels: np.ndarray, block_size: int) -> np.ndarray:
    finalized = labels.astype(np.uint8, copy=True)
    h, w = finalized.shape
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = finalized[y : y + block_size, x : x + block_size]
            block[...] = int(block.max())
    return finalized


def en_binary_segmentation(
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    threshold: float = THRESHOLD,
    grid_size: Tuple[int, int] = GRID_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mask, token_scores)`` for EN binary segmentation."""
    u_pix = _pixel_non_convergence(image_a, image_b)
    scores = _token_scores(u_pix, grid_size, POOL_PERCENTILE)
    labels = (scores >= threshold).astype(np.uint8)
    labels = _repair_small_e_islands(labels, AREA_MAX, MAX_ITER)
    labels = _block_finalize(labels, BLOCK_SIZE)
    return labels, scores


def en_tristate_segmentation(
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    threshold_low: float,
    threshold_high: float,
    grid_size: Tuple[int, int] = GRID_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, token_scores)`` with 0=converged, 1=near-converged, 2=non-converged."""
    if threshold_high <= threshold_low:
        raise ValueError(f"threshold_high must be larger than threshold_low, got {threshold_low}, {threshold_high}")
    u_pix = _pixel_non_convergence(image_a, image_b)
    scores = _token_scores(u_pix, grid_size, POOL_PERCENTILE)
    labels = np.zeros_like(scores, dtype=np.uint8)
    labels[(scores >= threshold_low) & (scores <= threshold_high)] = 1
    labels[scores > threshold_high] = 2
    return labels, scores


def save_en_overlay(
    base_image: Image.Image,
    mask: np.ndarray,
    output_path: str,
    *,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Overlay non-converged tokens on ``base_image`` with translucent red."""
    base = base_image.convert("RGBA")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(base.size, Image.NEAREST)
    red = Image.new("RGBA", base.size, (*color, int(round(255 * alpha))))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.paste(red, (0, 0), mask_img)
    out = Image.alpha_composite(base, overlay)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return out


def save_en_tristate_overlay(
    base_image: Image.Image,
    labels: np.ndarray,
    output_path: str,
    *,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay 0=converged, 1=near-converged, 2=non-converged labels."""
    base = base_image.convert("RGBA")
    label_img = Image.fromarray(labels.astype(np.uint8), mode="L").resize(base.size, Image.NEAREST)
    label_arr = np.asarray(label_img, dtype=np.uint8)

    overlay = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    colors = {
        0: (210, 255, 210),  # converged: very light green
        1: (255, 230, 120),  # near-converged: light amber
        2: (255, 0, 0),      # non-converged: red
    }
    alphas = {
        0: 0.10,
        1: 0.28,
        2: alpha,
    }
    for label, color in colors.items():
        mask = label_arr == label
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay[mask, 3] = int(round(255 * alphas[label]))

    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return out


def save_en_score_heatmap(
    base_image: Image.Image,
    scores: np.ndarray,
    output_path: str,
    *,
    alpha: float = 0.65,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Overlay continuous token scores in [0, 1] as red heat intensity."""
    base = base_image.convert("RGBA")
    heat = np.clip(scores.astype(np.float32), 0.0, 1.0)
    heat_img = Image.fromarray((heat * 255).round().astype(np.uint8), mode="L").resize(base.size, Image.NEAREST)
    heat_alpha = np.asarray(heat_img, dtype=np.float32) / 255.0
    heat_alpha = (heat_alpha * alpha * 255).round().astype(np.uint8)
    heat_rgba = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    heat_rgba[..., 0] = color[0]
    heat_rgba[..., 1] = color[1]
    heat_rgba[..., 2] = color[2]
    heat_rgba[..., 3] = heat_alpha
    overlay = Image.fromarray(heat_rgba, mode="RGBA")
    out = Image.alpha_composite(base, overlay)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return out
