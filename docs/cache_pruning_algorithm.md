# Cache Pruning 算法：完整数学描述

> 本文档完整描述 Lumina-DiMOO AdaTokenPruning 中 **Cache Pruning** 及其依赖的 **EN 自适应 Token 采样** 算法的实现细节与所有超参数。

---

## 目录

1. [系统总览](#1-系统总览)
2. [MaskGit 并行解码基础](#2-maskgit-并行解码基础)
3. [EN 三态 Region 分割](#3-en-三态-region-分割)
4. [EN 分区自适应 Remask](#4-en-分区自适应-remask)
5. [Cache Pruning（核心算法）](#5-cache-pruning核心算法)
6. [Max Logit Cache（选择性重算）](#6-max-logit-cache选择性重算)
7. [完整推理流程](#7-完整推理流程)
8. [全部超参数汇总](#8-全部超参数汇总)
9. [文件索引](#9-文件索引)

---

## 1. 系统总览

本系统由三个相互协作的推理优化机制组成（**训练阶段无任何 pruning 逻辑**）：

```
┌──────────────────────────────────────────────────────────────────┐
│                    MaskGit 并行解码主循环                         │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ EN 三态 Region   │  │ EN 分区自适应     │  │ Max Logit      │  │
│  │ 分割             │→│ Remask           │  │ Cache          │  │
│  │ (step 8,9 触发)  │  │ (分区独立步数)    │  │ (选择性重算)   │  │
│  └────────┬────────┘  └────────┬─────────┘  └────────────────┘  │
│           │                    │                                 │
│           ▼                    ▼                                 │
│  ┌─────────────────────────────────────────┐                    │
│  │ Cache Pruning (Chebyshev 距离衰减)       │                    │
│  │ Region 0+1 完成后一次性触发              │                    │
│  └─────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计**：本实现 **不使用 attention score** 做 pruning。Cache pruning 的重要性度量是 token 到 Region 2 的 **Chebyshev 距离**。

---

## 2. MaskGit 并行解码基础

### 2.1 符号定义

| 符号 | 含义 |
|------|------|
| \(T\) | 总步数 (timesteps) |
| \(\mathbf{x} \in \mathbb{Z}^{1 \times L}\) | 完整 token 序列（text + image tokens） |
| \(L\) | 序列长度 |
| \(N = H_g \times W_g\) | VQ token 网格大小（如 \(32 \times 32 = 1024\)） |
| \(\texttt{[MASK]}\) | mask token（id = 126336） |
| \(C\) | codebook 大小（默认 8192） |

### 2.2 每步核心操作

在第 \(t\) 步（\(t = 0, 1, \ldots, T-1\)）：

**Step 1: Forward pass (CFG)**

$$
\ell_{\text{cond}} = f_\theta(\mathbf{x})[\texttt{VQ offset} : \texttt{VQ offset} + C]
$$

$$
\ell_{\text{uncond}} = f_\theta(\mathbf{x}_{\text{uncond}})[\texttt{VQ offset} : \texttt{VQ offset} + C]
$$

$$
\ell = (1 + s) \cdot \ell_{\text{cond}} - s \cdot \ell_{\text{uncond}}
$$

其中 \(s\) 是 CFG scale（默认 4.0），仅在 mask 位置提取 logits。

**Step 2: Gumbel-Max 采样**

$$
\hat{z}_i = \arg\max_c \left( \frac{\ell_{i,c}}{\tau} + g_{i,c} \right), \quad g \sim \text{Gumbel}(0, 1)
$$

其中 \(\tau\) 是温度参数（默认 1.0）。

**Step 3: 置信度计算**

$$
p = \text{softmax}(\ell), \quad \text{conf}_i = p_{i, \hat{z}_i}
$$

**Step 4: Remask（决定下一步仍需 mask 的 token）**

使用 cosine schedule 计算本步需保持 mask 的数量：

$$
n_{\text{keep}} = \left\lfloor N_{\text{unknown}} \cdot \cos\!\left(\frac{\pi}{2} \cdot \frac{t+1}{T}\right) \right\rfloor
$$

在所有 unknown token 中，置信度最低的 \(n_{\text{keep}}\) 个被重新设为 \(\texttt{[MASK]}\)（通过 `mask_by_random_topk` 实现，含 Gumbel 噪声打破平局）。

### 2.3 `mask_by_random_topk` 实现细节

给定采样置信度 \(\text{conf}\) 和要保留 mask 的数量 \(n\)：

$$
\text{score}_i = \log(\text{conf}_i) + \tau \cdot g_i, \quad g_i \sim \text{Gumbel}(0,1)
$$

按 score 升序排列，score 最低的 \(n\) 个 token 被标记为继续 mask（即最不确定的 token 留到下一步）。

---

## 3. EN 三态 Region 分割

### 3.1 目标

在两个连续快照步（默认 step 8 和 step 9）各拍一次 VQ 快照，解码为图像后计算每个 token 的**非收敛分数**，将 token 网格划分为三个区域。

### 3.2 像素级非收敛度计算

给定快照图像 \(I_a\)（step 8）和 \(I_b\)（step 9），均转为 RGB \(\in [0, 1]^{H \times W \times 3}\)：

#### (a) 颜色差异 \(d_{\text{color}}\)

将 RGB 转为 CIE Lab 色彩空间（经 \(\sigma=3\) 的 Gaussian 模糊），计算逐像素 Lab 向量差：

$$
d_{\text{color}}(p) = \text{RobustNorm}\!\left(\|\text{Lab}(I_a(p)) - \text{Lab}(I_b(p))\|_2\right)
$$

其中 `RobustNorm` 使用第 2 和第 98 百分位做 min-max 裁剪归一化。

#### (b) 结构差异 \(d_{\text{ssim}}\)

计算灰度图的逐像素 SSIM 距离（窗口 11, \(\sigma=1.5\)）：

$$
\text{SSIM}(p) = \frac{(2\mu_a\mu_b + c_1)(2\sigma_{ab} + c_2)}{(\mu_a^2 + \mu_b^2 + c_1)(\sigma_a^2 + \sigma_b^2 + c_2)}
$$

$$
d_{\text{ssim}}(p) = \text{RobustNorm}(1 - \text{SSIM}(p))
$$

其中 \(c_1 = (0.01)^2\), \(c_2 = (0.03)^2\)（SSIM\_RANGE = 1.0）。

#### (c) 高频细节差异 \(d_{\text{detail}}\)

对灰度图计算局部高频能量（Laplacian → 15×15 均值池化），取细节缺失量：

$$
E(p) = \text{MeanPool}_{15}\!\left(\text{Laplacian}(\text{GaussSmooth}_1(I))^2\right)
$$

$$
\text{missing}(p) = \frac{\max(E_b(p) - E_a(p),\; 0)}{E_b(p) + \epsilon}
$$

$$
d_{\text{detail}}(p) = \text{RobustNorm}\!\left(\text{missing}(p) \cdot \text{RobustNorm}(E_b(p))\right)
$$

#### (d) 综合非收敛度

$$
u_{\text{pix}}(p) = \text{RobustNorm}\!\left(0.20 \cdot d_{\text{color}}(p) + 0.35 \cdot d_{\text{ssim}}(p) + 0.45 \cdot d_{\text{detail}}(p)\right)
$$

### 3.3 Token 级分数

将像素级非收敛图 \(u_{\text{pix}}\) 划分为 \(H_g \times W_g\) 的 cell 网格（经 \(\sigma=3\) 中值滤波 + RobustNorm 预处理）。每个 cell 的分数取其内部像素的第 80 百分位：

$$
s_{i,j} = \text{Percentile}_{80}\!\left(\{u_{\text{pix}}(p) : p \in \text{cell}(i,j)\}\right)
$$

### 3.4 三态标签

$$
\text{label}_{i,j} = \begin{cases}
0 & \text{if } s_{i,j} < \theta_{\text{low}} \quad \text{（已收敛，Region E）} \\
1 & \text{if } \theta_{\text{low}} \leq s_{i,j} \leq \theta_{\text{high}} \quad \text{（近收敛，Region N）} \\
2 & \text{if } s_{i,j} > \theta_{\text{high}} \quad \text{（未收敛，Region Target）}
\end{cases}
$$

默认阈值：\(\theta_{\text{low}} = 0.18\)，\(\theta_{\text{high}} = 0.48\)。

---

## 4. EN 分区自适应 Remask

### 4.1 核心思想

EN label 计算完成后（`en_region_base_step`，默认 step 9），不再使用全局统一的 cosine schedule，而是每个 region 独立控制剩余步数：

| Region | 语义 | 默认步数 | 完成 step |
|--------|------|---------|-----------|
| 0 | 已收敛（E） | 4 | 13 |
| 1 | 近收敛（N） | 12 | 21 |
| 2 | 未收敛（Target） | 20 | 29 |

### 4.2 分区 Remask 公式

设 `en_region_base_step` = \(t_0\)，Region \(r\) 的预算步数为 \(S_r\)，当前 local step \(\delta = t - t_0\)：

$$
\text{frac}_r = \cos\!\left(\frac{\pi}{2} \cdot \frac{\delta}{S_r}\right)
$$

$$
n_{\text{keep}}^{(r)} = \left\lfloor N_r^{(0)} \cdot \text{frac}_r \right\rfloor
$$

其中 \(N_r^{(0)}\) 是 Region \(r\) 在 \(t_0\) 时刻的 unknown token 初始计数。

- 若 \(n_{\text{keep}}^{(r)} < 1\)，则 clamp 到 1
- 若 \(\delta \geq S_r\)，则 \(n_{\text{keep}}^{(r)} = 0\)（该 region 采样结束，所有 token 定型）

在每个 region 内部，仍使用 `mask_by_random_topk` 基于置信度选择要保留 mask 的 token。

---

## 5. Cache Pruning（核心算法）

### 5.1 触发条件

Cache pruning 是**一次性操作**，当且仅当以下条件全部满足时触发：

1. `cache_prune_ratio > 0`
2. EN region labels 已计算完成
3. Region 0 和 Region 1 都已完成采样，即 \(\delta \geq S_0\) 且 \(\delta \geq S_1\)

在默认配置下，触发时刻为 global step **21**（\(= 9 + 12\)）。

### 5.2 目标

从 KV cache 和输入序列中物理删除已收敛的 Region 0+1 token，使后续 Region 2 的 forward pass 在更短的序列上进行。

### 5.3 算法步骤

#### Step 1: 构建网格级 mask

设 VQ token 网格为 \(G \in \mathbb{Z}^{H_g \times W_g}\)，EN label 网格为 \(\Lambda \in \{0, 1, 2\}^{H_g \times W_g}\)：

$$
\text{target}_{i,j} = \mathbb{1}[\Lambda_{i,j} = 2]
$$

$$
\text{candidate\_region}_{i,j} = \mathbb{1}[\Lambda_{i,j} \in \{0, 1\}]
$$

$$
\text{committed}_{i,j} = \mathbb{1}[G_{i,j} \neq \texttt{[MASK]}]
$$

$$
\text{prune\_candidate}_{i,j} = \text{candidate\_region}_{i,j} \wedge \text{committed}_{i,j}
$$

#### Step 2: 边界保护（Boundary Protection）

使用 4 连通域膨胀 target 区域 \(r\) 次（默认 \(r = 2\)），保护 target 边界附近的 token：

$$
\text{boundary} = \text{Dilate}(\text{target},\; r) \wedge \text{prune\_candidate}
$$

**膨胀算法**：对 2D 布尔网格迭代 \(r\) 次，每次将每个 True 扩展到其上下左右 4 邻居。

#### Step 3: Anchor 保护

为防止信息丢失，保留稀疏的 anchor token：

**确定性 anchor**（步长 \(\sigma\)）：

$$
\text{anchor\_det}_{i,j} = \mathbb{1}[i \bmod \sigma = 0] \wedge \mathbb{1}[j \bmod \sigma = 0]
$$

**伪随机 anchor**（比例 \(\rho\)）：

$$
\text{anchor\_rand}_{i,j} = \mathbb{1}\!\left[\text{hash}(i, j) < \lfloor 1000\rho \rfloor\right]
$$

其中 \(\text{hash}(i, j) = (131i + 197j + 17) \bmod 1000\)。

$$
\text{anchors} = (\text{anchor\_det} \vee \text{anchor\_rand}) \wedge \text{prune\_candidate}
$$

默认 \(\sigma = 0\)（禁用），\(\rho = 0.0\)（禁用）。

#### Step 4: 实际候选集

$$
\text{candidate} = \text{prune\_candidate} \setminus \text{boundary} \setminus \text{anchors}
$$

#### Step 5: 计算 prune 数量

$$
n_{\text{prune\_target}} = \text{round}(|\text{prune\_candidate}| \times \alpha)
$$

$$
n_{\text{available}} = \max(0, |\text{candidate}| - n_{\text{min\_keep}})
$$

$$
n_{\text{prune}} = \max(0, \min(n_{\text{prune\_target}}, n_{\text{available}}))
$$

其中 \(\alpha\) 是 `cache_prune_ratio`。

#### Step 6: Chebyshev 距离计算

对 target 区域计算每个格子的 Chebyshev 距离：

$$
d_{\text{cheb}}(i,j) = \min_{(i', j') \in \text{target}} \max(|i - i'|, |j - j'|)
$$

实现方式为 **8 连通域 BFS**：从所有 target 格子出发，每次迭代向 8 个方向（上、下、左、右、4 个对角）扩展，第 \(d\) 轮新覆盖的格子距离为 \(d\)。

#### Step 7: 选择要 prune 的 token

在 candidate 中，按 Chebyshev 距离降序排列，选取距离最大的 \(n_{\text{prune}}\) 个 token：

$$
\text{prune\_set} = \text{TopK}_{n_{\text{prune}}}\!\left(\{d_{\text{cheb}}(i,j) : (i,j) \in \text{candidate}\}\right)
$$

**直觉**：离 Region 2（仍在采样）最远的 token 对后续生成影响最小，优先删除。

#### Step 8: 物理裁剪

对每层 Transformer block 的 KV cache 做 index-select：

$$
\mathbf{K}_l^{\text{new}} = \mathbf{K}_l[:, \text{keep\_indices}, :], \quad
\mathbf{V}_l^{\text{new}} = \mathbf{V}_l[:, \text{keep\_indices}, :]
$$

同时裁剪 logit cache：

$$
\text{logit\_cache}^{\text{new}} = \text{logit\_cache}[:, \text{keep\_indices}, :]
$$

对 cond 和 uncond 分别执行裁剪（uncond 的 keep indices 需做 offset 映射）。

### 5.4 Pruning 后的状态变化

1. 输入序列 \(\mathbf{x}\) 压缩为 \(\mathbf{x}_{\text{compact}} = \mathbf{x}[:, \text{keep\_indices}]\)
2. **禁用 incremental cache**（`model.caching(False)`），后续步全量 forward
3. 保存 pruning 前的完整 VQ grid 快照（`pruned_vq_ids_snapshot`）
4. 重建 `position_to_vq` 映射、`code_start` 等辅助变量
5. 保留原始绝对位置 ID（`active_position_ids`）用于 RoPE

### 5.5 推理结束后序列重建

推理结束时，将 compact 序列中的采样结果合并回完整 VQ grid：

```
for i in compact_sequence:
    vq_idx = position_to_vq[i]
    if vq_idx >= 0:
        full_vq_ids[vq_idx] = x_compact[i]
```

---

## 6. Max Logit Cache（选择性重算）

这是与 Cache Pruning 正交的独立加速机制，在 EN 分区和 Pruning 之外运行。

### 6.1 原理

每步 forward 后，根据每个位置 logits 的最大值（max logit）评估该位置的"确定程度"。下一步仅重算最不确定的位置，其余复用缓存。

### 6.2 公式

设 `cache_ratio` = \(\gamma\)（默认 0.9），`compute_ratio` = \(1 - \gamma = 0.1\)：

$$
\text{ml}_i = \max_c \ell_{i,c}^{(\text{cond})}
$$

$$
\theta = \text{Quantile}(\{\text{ml}_i\},\; 1 - \gamma)
$$

$$
\text{to\_compute\_mask}_i = \mathbb{1}[\text{ml}_i \leq \theta]
$$

约 10% 最低 max-logit 位置被重算，其余 90% 直接复用上一步缓存。

### 6.3 KV Cache 更新

对标记为需要重算的位置：

$$
\mathbf{K}_l[\text{mask}] \leftarrow \mathbf{K}_l^{\text{new}}, \quad
\mathbf{V}_l[\text{mask}] \leftarrow \mathbf{V}_l^{\text{new}}
$$

$$
\text{logit\_cache}[\text{mask}] \leftarrow \text{logits}^{\text{new}}
$$

对未标记位置，KV cache 和 logit cache 不变，返回完整的 logit\_cache 作为本步输出。

### 6.4 刷新机制

- **Warmup**：在前 `warmup_step` 步（EN 模式下 = `en_region_cache_start_step - 1` = 9）不使用 cache
- **周期刷新**：每 `refresh_interval` 步（默认 5）清空 cache，全量重算
- 刷新步无 `to_compute_mask`，所有位置完整 forward

---

## 7. 完整推理流程

```
输入: model, prompt, uncon_ids, 各超参数
─────────────────────────────────────────────────
1. 初始化 x = [text_tokens | BOA | BOI | MASK×N | EOI | EOA]
2. 计算 effective_timesteps:
     若 EN 模式: T = en_snapshot_step_b + max(en_region_steps) + 1
                   默认 = 9 + 20 + 1 = 30
     否则: T = timesteps (默认 64)

3. 开启 KV cache
4. 预计算 refresh_steps 位图

for t = 0, 1, ..., T-1:
    若所有 token 已 commit: 提前退出

    ── Cache Pruning Gate ──────────────────────
    if 未 prune 且 ratio > 0 且 EN labels 就绪:
        δ = t - t₀
        if δ ≥ S₀ AND δ ≥ S₁:    # Region 0,1 都完成
            plan = build_cache_prune_plan(...)
            物理裁剪 cond/uncond KV cache
            压缩 x 序列
            禁用 incremental cache
            标记 cache_pruned = True

    ── Forward (CFG) ───────────────────────────
    cond_logits = model(x, to_compute_mask, position_ids)
    uncond_logits = model(uncond, ...)
    logits = (1+s)·cond - s·uncond    (仅 mask 位置)

    ── 采样 & 写入 ─────────────────────────────
    sampled = GumbelMax(logits, τ)
    conf = softmax(logits)[sampled]
    x[mask_positions] = sampled

    ── EN 快照 & Label 计算 ─────────────────────
    if t == en_snapshot_step_b (默认 9):
        解码 step 8, 9 的 VQ 快照为图像
        labels = en_tristate_segmentation(img_a, img_b)
        记录 en_region_initial_counts
        t₀ = t

    ── Remask ──────────────────────────────────
    if EN labels 就绪 且 t > t₀:
        对每个 region r ∈ {0,1,2}:
            n_keep = ⌊N_r^(0) · cos(π/2 · δ/S_r)⌋
            用 mask_by_random_topk 选择 region 内的 n_keep 个 token 继续 mask
    else:
        全局 cosine schedule remask

    ── 更新 Cache Mask ─────────────────────────
    if cache 启用 且 非刷新步:
        to_compute_mask = (max_logit ≤ quantile(max_logit, 1-γ))

结束循环
─────────────────────────────────────────────────
若 cache_pruned: merge compact 结果回完整 VQ grid
decode VQ → 图像
```

---

## 8. 全部超参数汇总

### 8.1 Cache Pruning 参数

| 参数 | CLI 标志 | 默认值 | 类型 | 描述 |
|------|----------|--------|------|------|
| prune\_ratio | `--cache_prune_ratio` | **0.0**（禁用） | float | Region 0+1 已 commit token 的 prune 比例 \(\alpha\) |
| context\_radius | `--cache_prune_context_radius` | **2** | int | target 边界保护半径 \(r\)（4 连通域膨胀） |
| anchor\_stride | `--cache_prune_anchor_stride` | **0**（禁用） | int | 确定性 anchor 网格步长 \(\sigma\) |
| anchor\_ratio | `--cache_prune_anchor_ratio` | **0.0**（禁用） | float | 伪随机 anchor 保留比例 \(\rho\) |
| min\_keep | `--cache_prune_min_keep` | **0** | int | candidate 中最少保留 token 数 |
| token\_grid\_height | — | **32** | int | VQ token 网格高度 \(H_g\) |
| token\_grid\_width | — | **32** | int | VQ token 网格宽度 \(W_g\) |

### 8.2 EN 分区采样参数

| 参数 | CLI 标志 | 默认值 | 类型 | 描述 |
|------|----------|--------|------|------|
| en\_region\_sampling | `--en_region_sampling` | **True** | bool | 启用 EN 分区独立采样 |
| en\_snapshot\_step | `--en_snapshot_step` | **8** | int | 第一个 EN 快照步（0-based） |
| en\_snapshot\_step\_b | `--en_snapshot_step_b` | **9** | int | 第二个 EN 快照步；此步计算 labels |
| en\_threshold\_pair | `--en_threshold_pair` | **"0.18:0.48"** | str | 三态分割阈值 \((\theta_\text{low}, \theta_\text{high})\) |
| en\_region\_steps | `--en_region_steps` | **"4,12,20"** | str | Region 0/1/2 的剩余采样步数 \((S_0, S_1, S_2)\) |
| en\_region\_cache\_start\_step | `--en_region_cache_start_step` | **10** | int | EN 模式下首个可用 cache 的步 |
| en\_alpha | `--en_alpha` | **0.45** | float | EN heatmap 可视化透明度 |

**派生量（默认配置）**：
- \(T_{\text{eff}} = 9 + \max(4, 12, 20) + 1 = 30\)
- warmup\_step = 9
- Region 0 完成：step 13 (\(= 9 + 4\))
- Region 1 完成 / Cache pruning 触发：step 21 (\(= 9 + 12\))
- Region 2 完成：step 29 (\(= 9 + 20\))

### 8.3 Max Logit Cache 参数

| 参数 | CLI 标志 | 默认值 | 类型 | 描述 |
|------|----------|--------|------|------|
| use\_cache | `--use_cache` / `--no_cache` | **True** | bool | 启用 KV cache |
| cache\_ratio | `--cache_ratio` | **0.9** | float | 复用比例 \(\gamma\)；只重算最低 \((1-\gamma)\) max-logit 位置 |
| warmup\_ratio | `--warmup_ratio` | **0.3** | float | 原始模式 warmup 比例（EN 模式被覆盖） |
| refresh\_interval | `--refresh_interval` | **5** | int | 每 N 步全量刷新 cache |

### 8.4 通用生成参数

| 参数 | CLI 标志 | 默认值 | 描述 |
|------|----------|--------|------|
| timesteps | `--timesteps` | **64**（EN 模式下被覆盖为 30） | 名义总步数 |
| cfg\_scale | `--cfg_scale` | **4.0** | Classifier-free guidance 强度 |
| temperature | `--temperature` | **1.0** | Gumbel 采样温度 \(\tau\) |
| height / width | `--height` / `--width` | **512** / **512** | 图像尺寸（影响 VQ grid 大小） |

### 8.5 EN 分割内部常量（无 CLI 暴露）

| 常量 | 值 | 来源文件 | 描述 |
|------|-----|---------|------|
| COLOR\_PERCENTILES | (2, 98) | en\_segmentation.py | 颜色差异 RobustNorm 百分位 |
| SSIM\_WIN | 11 | en\_segmentation.py | SSIM 窗口大小 |
| SSIM\_SIGMA | 1.5 | en\_segmentation.py | SSIM 高斯核标准差 |
| SSIM\_RANGE | 1.0 | en\_segmentation.py | SSIM 值域范围 |
| SSIM\_PERCENTILES | (2, 98) | en\_segmentation.py | 结构差异 RobustNorm 百分位 |
| FREQ\_PERCENTILES | (2, 98) | en\_segmentation.py | 高频细节 RobustNorm 百分位 |
| PIX\_PERCENTILES | (2, 98) | en\_segmentation.py | 综合非收敛度 RobustNorm 百分位 |
| TOKEN\_PRE\_PERCENTILES | (1, 99) | en\_segmentation.py | Token 分数预处理百分位 |
| POOL\_PERCENTILE | 80 | en\_segmentation.py | Cell 内取第 80 百分位 |
| 颜色/结构/细节权重 | 0.20 / 0.35 / 0.45 | en\_segmentation.py | 三通道融合权重 |
| 高斯模糊 \(\sigma\)（Lab） | 3.0 | en\_segmentation.py | Lab 色彩空间预模糊 |
| 高斯模糊 \(\sigma\)（Laplacian） | 1.0 | en\_segmentation.py | 高频能量预平滑 |
| 均值池化窗口 | 15 | en\_segmentation.py | 高频能量局部池化 |
| 中值滤波窗口 | 3 | en\_segmentation.py | Token 分数中值滤波 |

---

## 9. 文件索引

| 文件路径 | 功能 |
|----------|------|
| `utils/cache_pruning.py` | Cache pruning 核心：`build_cache_prune_plan`、`prune_model_cache` |
| `utils/en_segmentation.py` | EN 三态分割：像素非收敛度 → token 分数 → 三态 label |
| `utils/generation_utils.py` | `mask_by_random_topk`、`cosine_schedule`、`gumbel_max_sample` |
| `generators/image_generation_generator.py` | 主采样循环：EN 分区 + cache + pruning 触发与序列重建 |
| `model/modeling_llada.py` | Transformer 模型、KV cache、logit cache、RoPE |
| `evaluation/gen_eval/geneval_lumina_dimoo.py` | 唯一完整暴露所有 `cache_prune_*` CLI 参数的入口 |
