# Lumina 每个 step 采样 token 数量如何确定

这份说明基于仓库当前实现（主要在 `generators/image_generation_generator.py`）。

## 先说结论

Lumina 在第 `step` 步里，**会对当前所有仍为 mask 的位置都进行一次采样**；  
随后再用一个调度函数决定“下一步还要保留多少 mask”，其余位置就被确定下来，不再参与后续采样。

所以可以分成两个数量：

- **本步参与采样的 token 数**：`unknown_cnt`（当前 mask 数）
- **本步采样后继续保留为 mask 的 token 数**：`keep_n`
- **本步真正被确定下来的 token 数**：`unknown_cnt - keep_n`

---

## 具体怎么算

设：

- `T = timesteps`
- `L = vq_len`（初始需要生成的图像 token 总数）
- `t` 从 `0` 开始计数

在 `t < T-1` 时：

1. 计算下一步保留比例

   $$
   \mathrm{frac}_t = \cos\left(\frac{\pi}{2}\cdot\frac{t+1}{T}\right)
   $$

   纯文本写法：`frac_t = cos((pi/2) * ((t+1)/T))`

2. 计算下一步保留 mask 数

   $$
   \mathrm{keep\_n}_t = \max\left(1,\left\lfloor L\cdot \mathrm{frac}_t\right\rfloor\right)
   $$

   纯文本写法：`keep_n_t = max(1, floor(L * frac_t))`

在最后一步 `t = T-1`：

$$
\mathrm{keep\_n}_{T-1} = 0
$$

纯文本写法：`keep_n_(T-1) = 0`

也就是最后一步后不再保留 mask，全部 token 收敛为最终结果。

> 注意：`keep_n` 是按初始总长度 `L` 计算的，而不是按当前 `unknown_cnt` 计算。  
> 通过“重新 mask 低置信度 token”的方式，`unknown_cnt` 会被拉到这个目标数量附近。

---

## 每步流程（直观版）

每个 step 做 4 件事：

1. 对当前所有 mask 位置做前向，得到 logits。
2. 在这些位置上采样候选 token（Gumbel-Max 或温度控制采样）。
3. 计算每个候选 token 的置信度。
4. 只保留 `keep_n` 个“低置信度位置”继续 mask；高置信度位置固定下来。

因此：

- 前期 `keep_n` 大，固定下来的 token 少（探索更多）
- 后期 `keep_n` 小，固定下来的 token 多（快速收敛）
- 最后一步 `keep_n=0`，全部定稿

---

## 一个小例子

假设：

- `L = 256`
- `T = 8`

则大致有：

- step 1 后保留约 `floor(256*cos(pi/16)) ≈ 251`
- step 4 后保留约 `floor(256*cos(4*pi/16)) ≈ 181`
- step 7 后保留约 `floor(256*cos(7*pi/16)) ≈ 49`
- step 8 后保留 `0`

含义：越往后，每一步“最终确定”的 token 数越来越多。

---

## 和“采样数量”最容易混淆的点

- 如果你问“模型本步会采样多少个位置？”：答案是 **当前所有 mask 位置**（即 `unknown_cnt`）。
- 如果你问“这一步会新增多少个已确定 token？”：答案是约 **`unknown_cnt - keep_n`**。
- `temperature`、`cfg_scale` 会影响采样分布和置信度排序，但**不直接改变 `keep_n` 的公式**。

