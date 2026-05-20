# EN 三分类分割算法当前实现说明

本文档描述当前仓库中实际启用的 EN 分割算法实现。相关代码位于：

- `utils/en_segmentation.py`
- `inference/inference_t2i.py`

当前推断脚本只生成 `step10 -> step11` 的 EN 可视化，不再生成 `step24 -> step25`。

## 1. 输入与目标

给定同一次生成过程中的两张 RGB 快照：

- `I_a`：较早快照，默认 `step10`
- `I_b`：较晚快照，默认 `step11`

算法计算 token 网格上的未收敛度，并输出 `64x64` 的三分类标签图：

- `0`：收敛区
- `1`：中间区 / 近收敛区
- `2`：未收敛区

当前可视化底图使用较早快照 `I_a`，即 `step10 -> step11` 的图覆盖在 `step10` 快照上。

## 2. 像素级未收敛度

输入图像会先转为 RGB，并归一化到 `[0, 1]`。若两张快照尺寸不同，`I_b` 会 resize 到 `I_a` 尺寸。

鲁棒归一化定义为：

```text
R_{p_l,p_h}(X) = clip((X - Q_{p_l}(X)) / (Q_{p_h}(X) - Q_{p_l}(X) + eps), 0, 1)
```

其中：

```text
eps = 1e-6
```

像素级未收敛度由三项组成。

### 2.1 颜色差

RGB 转 Lab 后，对 Lab 空间图像做高斯平滑：

```text
sigma_Lab = 3.0
percentiles = (2, 98)
```

颜色差为两张快照平滑 Lab 图的 L2 距离，并做鲁棒归一化。

### 2.2 结构差

结构差使用灰度图上的 SSIM 距离：

```text
D_s = R_{2,98}(1 - SSIM(I_a, I_b))
```

SSIM 参数：

```text
window = 11
sigma = 1.5
range = 1.0
percentiles = (2, 98)
```

实现中使用显式 `11x11` 高斯核。

### 2.3 细节缺失

先计算局部高频能量：

```text
E(I) = mean_filter_15x15((Laplacian(Gaussian(gray(I), sigma=1.0)))^2)
```

细节缺失项：

```text
D_f = R_{2,98}(max(E(I_b)-E(I_a), 0) / (E(I_b)+eps) * R_{2,98}(E(I_b)))
```

参数：

```text
sigma_gray = 1.0
local_energy_window = 15
percentiles = (2, 98)
```

### 2.4 三项融合

最终像素级未收敛度：

```text
U_pix = R_{2,98}(0.20 * D_c + 0.35 * D_s + 0.45 * D_f)
```

融合权重：

```text
w_color = 0.20
w_structure = 0.35
w_frequency = 0.45
```

## 3. 像素到 token 网格

当前 EN 网格固定为：

```text
GRID_SIZE = 64 x 64
```

对 `U_pix` 先做二次归一化和中值滤波：

```text
U_smooth = median_filter_3x3(R_{1,99}(U_pix))
```

然后将图像划分为 `64x64` 个不重叠 cell。单个 cell 尺寸为：

```text
floor(H0 / 64) x floor(W0 / 64)
```

若 `H0` 或 `W0` 不能被 64 整除，只使用每个 cell 对应的整除区域。

每个 token cell 的未收敛度为该 cell 内像素未收敛度的 `80%` 分位数：

```text
u_ij = Q80(U_smooth over cell_ij)
```

因此输出 score 图为：

```text
scores in [0, 1]^{64 x 64}
```

## 4. 三分类判定

当前推断默认只处理：

```text
step10 -> step11
```

默认阈值：

```text
T1 = 0.15
T2 = 0.45
```

分类规则：

```text
score < T1          -> 0, 收敛区
T1 <= score <= T2   -> 1, 中间区 / 近收敛区
score > T2          -> 2, 未收敛区
```

注意：`utils/en_segmentation.py` 中仍保留了旧的二值分割函数 `en_binary_segmentation()`，但当前 `inference_t2i.py` 的 EN 可视化入口使用的是三分类函数 `en_tristate_segmentation()`。

## 5. 可视化规则

当前 EN 可视化函数为：

```text
save_en_tristate_overlay()
```

底图：

```text
step10 -> step11 使用 step10 快照作为底图
```

颜色与透明度：

```text
收敛区     label=0  very light green  RGB=(210, 255, 210), alpha=0.10
中间区     label=1  light amber       RGB=(255, 230, 120), alpha=0.28
未收敛区   label=2  red               RGB=(255, 0, 0),     alpha=en_alpha
```

其中 `en_alpha` 是命令行参数，默认：

```text
en_alpha = 0.45
```

## 6. 推断脚本参数

当前相关命令行参数：

```text
--en_heatmap
--en_snapshot_step      default=10
--en_snapshot_step_b    default=11
--en_threshold_pair     default="0.15:0.45"
--en_alpha              default=0.45
```

开启 `--en_heatmap` 后，生成器会额外捕获 `step10` 和 `step11` 的 VQ token 快照，用于 EN 分割。

## 7. 输出文件

开启 EN 后，每个 prompt 默认输出两个文件：

```text
<name>.png
<name>_en_tristate_step10_to_step11.png
```

当前不会输出：

- `step24_to_step25` 可视化
- 快照 PNG
- `.npy` mask
- 连续 score heatmap

## 8. 运行示例

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n lumina_dimoo python inference/inference_t2i.py \
  --checkpoint "/mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights" \
  --vae_ckpt "/mnt/data1/yanfeihong/projs/Lumina-DiMOO/weights" \
  --prompt "a glass greenhouse filled with tropical plants at sunrise" \
  --height 1024 --width 1024 --timesteps 32 \
  --use-cache \
  --en_heatmap \
  --output_dir "debug_results/en_single_pair_1024_test"
```

典型日志会包含：

```text
ENTriStateTime step10_to_step11:
  total=<EN可视化总耗时>
  compute=<纯EN分割计算耗时>
  generation=<生成原图耗时>
  ratio=<EN总耗时 / 生成耗时>
  thresholds=0.1500:0.4500
  label_shape=64x64
  converged=<收敛区数量>
  near_converged=<中间区数量>
  non_converged=<未收敛区数量>/4096
```

## 9. 当前默认超参数汇总

```text
GRID_SIZE = (64, 64)
eps = 1e-6

COLOR_PERCENTILES = (2, 98)
SSIM_WIN = 11
SSIM_SIGMA = 1.5
SSIM_RANGE = 1.0
SSIM_PERCENTILES = (2, 98)
FREQ_PERCENTILES = (2, 98)
PIX_PERCENTILES = (2, 98)
TOKEN_PRE_PERCENTILES = (1, 99)

POOL_PERCENTILE = 80

融合权重:
  color = 0.20
  structure = 0.35
  frequency = 0.45

三分类阈值:
  T1 = 0.15
  T2 = 0.45

可视化:
  converged alpha = 0.10
  near-converged alpha = 0.28
  non-converged alpha = 0.45
```
