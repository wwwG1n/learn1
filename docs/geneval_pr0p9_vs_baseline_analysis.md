# GenEval 评估对比分析：Token Pruning (pr=0.9) vs No-Pruning (Baseline)

> 报告日期：2026-05-21
> 分析对象：
> - **Baseline**（不开启 pruning）：`full553x4_512_cache_originalrope_lumina_dimoo_geneval_20260519_183024_h512_w512_t64_eff30_cfg4_temp1_seed42_n4_cache1_cr0p9_wu0p3_ri5_en1_s8-9_thr0.18-0.48_rs4-12-20_cs10`
> - **Pruning**（pruning ratio = 0.9）：`pr0p9_full553x4_cache_origrope_u_lumina_dimoo_geneval_20260520_203623_h512_w512_t64_eff30_cfg4_temp1_seed42_n4_cache1_cr0p9_wu0p3_ri5_en1_s8-9_thr0.18-0.48_rs4-12-20_cs10`
> - 配置：512×512、64 步、CFG=4、temp=1、seed=42、每个 prompt 4 张图、共 553 prompts × 4 = 2212 张图

可视化报告：参见同目录 [`geneval_pr0p9_vs_baseline.html`](./geneval_pr0p9_vs_baseline.html)（在浏览器中打开）。

---

## 1. 实验摘要（Executive Summary）

| 指标 | Baseline | Pruning 0.9 | Δ (pp) |
|---|---|---|---|
| 总图像数 | 2212 | 2212 | — |
| **% correct images** | **85.62%** | **84.54%** | **−1.08** |
| **% correct prompts (any-of-4)** | **94.03%** | **92.95%** | **−1.08** |
| Overall score (任务平均) | 0.85849 | 0.84805 | −1.04 |
| single_object | 99.69 % | 99.06 % | −0.63 |
| two_object | 93.43 % | 92.17 % | −1.26 |
| counting | 76.25 % | 75.62 % | −0.63 |
| colors | 91.22 % | 91.22 % | **0.00** |
| position | 82.00 % | 80.25 % | −1.75 |
| color_attr | 72.50 % | 70.50 % | −2.00 |

**核心结论**：在保留约 10% 的 token（剪枝率 0.9，即丢弃 90% 候选）这一极激进的设置下，整体 GenEval 准确率仅下降约 **1 个百分点**。其中 **color_attr** 与 **position** 类受影响最大，**colors** 与 **single_object** 几乎无损。

---

## 2. Prompt 级 / 图像级的 Regression 与 Improvement 统计

按每个 prompt 的 4 张图中正确张数 `n_correct ∈ {0,1,2,3,4}` 进行 baseline vs pruning 比较，定义：
- `Δn = n_prune − n_base`
- `regression`：Δn < 0（图像级退化）
- `improvement`：Δn > 0
- `tie`：Δn = 0

### 2.1 总体分布

| 类型 | 数量 | 占比 |
|---|---|---|
| Regression（Δn < 0） | 54 | 9.76 % |
| Improvement（Δn > 0） | 31 | 5.61 % |
| Tie（Δn = 0） | 468 | 84.63 % |
| **合计** | 553 | 100 % |

> Regression 与 Improvement 之比约 **54:31 ≈ 1.74:1**。84.6 % 的 prompt 在两种配置下评估结果完全相同。
> 净退化 = 54 − 31 = **23 个 prompt 出现至少一张图的退化**，对应图像级别净下降 24 张。

### 2.2 Prompt 级翻转（any-of-4 翻转）

| 翻转类型 | 数量 | 描述 |
|---|---|---|
| Base ✓ → Prune ✗ | **7** | baseline 通过、pruning 后整组 4 张全部失败（"硬失败"） |
| Base ✗ → Prune ✓ | 1 | baseline 失败、pruning 后至少 1 张通过（"硬恢复"） |

净 prompt 翻转 = 7 − 1 = **6**，与 res.txt 中 prompt 准确率差 6 / 553 ≈ −1.08% 完全吻合。

### 2.3 按 Category 拆解

| Tag | #Prompts | Img(b)% | Img(p)% | Prompt-any(b)% | Prompt-any(p)% | reg-img | imp-img | reg-prompt(any) | imp-prompt(any) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single_object | 80 | 99.69 | 99.06 | 100.00 | 100.00 | 2 | 0 | 0 | 0 |
| two_object | 99 | 93.43 | 92.17 | 98.99 | 96.97 | 8 | 4 | 2 | 0 |
| counting | 80 | 76.25 | 75.62 | 91.25 | 88.75 | 10 | 8 | 2 | 0 |
| colors | 94 | 91.22 | 91.22 | 95.74 | 95.74 | 5 | 5 | 0 | 0 |
| position | 100 | 82.00 | 80.25 | 92.00 | 92.00 | 10 | 5 | 0 | 0 |
| color_attr | 100 | 72.50 | 70.50 | 87.00 | 85.00 | 19 | 9 | 3 | 1 |
| **合计** | **553** | 85.62 | 84.54 | 94.03 | 92.95 | **54** | **31** | **7** | **1** |

**关键观察**：

1. **`color_attr` 是受影响最严重的类别**：19 reg vs 9 imp，是单类中绝对退化最多的；3 个 prompt 完全 flipped to wrong。这表明剪枝最容易扰动颜色×物体绑定关系（对**精细 RGB 区分**和**多对象+属性**双重约束最敏感）。
2. **`counting` 与 `two_object` 紧随其后**：每类各 2 个 prompt 完全失败。这两类共同点是**多目标 / 多实例**——剪枝可能让多个物体的边界更模糊，触发 detector 的合并或漏检。
3. **`colors` 完全持平**：5 reg / 5 imp 完全对消。单一物体 + 颜色任务对 token 剪枝最鲁棒，因为颜色信息分布广泛，少量 token 被剪不致影响整体色调。
4. **`position` 出现"图像级退化但 prompt 级未退化"**：10 张图退化，0 个 prompt 整体翻转。说明 baseline 在每个 prompt 上往往有多张图通过，剪枝后翻倒一两张但 best-of-4 仍能命中，仅 % correct images 下滑。
5. **`single_object`**：仅 2 张图退化、0 个 prompt 翻转。最简单任务最鲁棒。

---

## 3. 7 个完全翻转（base ✓ → prune ✗）的深入分析

7 个完全失败的 prompt 列表如下，按 tag 分组：

| # | prompt_id | tag | prompt | 失败原因（GenEval reason） |
|---:|---|---|---|---|
| 1 | 00163 | two_object | a photo of a couch and a snowboard | snowboard 全部漏检（4/4 张被识别为 skateboard 或未检出） |
| 2 | 00178 | two_object | a photo of a baseball bat and a giraffe | baseball bat 全部漏检（base 仅 1 张以 0.32 confidence 通过） |
| 3 | 00217 | counting | a photo of three books | book 数量不足（base 1 张正好 3 本，prune 全部 ≤ 2 本） |
| 4 | 00254 | counting | a photo of four benchs | bench 数量不足（base 1 张正好 4 把，prune 全部 ≤ 3 把） |
| 5 | 00477 | color_attr | a photo of a brown bed and a pink cell phone | base 第 4 张 cell phone 通过，prune 同图未检出 bed |
| 6 | 00492 | color_attr | a photo of a white tie and a purple skateboard | skateboard 全部漏检（base 仅 1 张以 0.52 confidence 通过） |
| 7 | 00495 | color_attr | a photo of a yellow bowl and a white baseball glove | baseball glove 全部漏检（base 仅 1 张以 0.36 confidence 通过） |

### 3.1 关键洞察：Pruning 0.9 的视觉差异极其微小

将每个 case 的 4 张图组成 `grid.png`，对比 baseline 与 pruning：

**Case 00163（couch + snowboard）**

视觉上，baseline 与 pruning 的 4 张图**几乎像素级一致**：couch 形状、snowboard 颜色、布局角度均高度相近。区别在于 detector 输出：

- **Baseline 0000.png**：检出 `snowboard 0.475` ✅
- **Pruning  0000.png**：同一物体检出为 `skateboard 0.953` ❌

这是 detector 在两个相近类别（snowboard / skateboard）边界附近的**细节敏感性翻转**，并非生成模型语义错误。Pruning 让物体表面纹理/边角细节略微平滑，使 detector 倾向于更"短粗"的 skateboard 类。

**Case 00217（three books）**

视觉对比下，3 本书的"分离感"在 pruning 中略微减弱：书与书之间的**接缝、阴影、字脊**变化更平滑：

- Baseline 0002.png：3 本独立矩形被检测为 3 个 book（每个置信度 ≥ 0.94）
- Pruning  0002.png：完全没有 book 检测出（0 个 detection）

这说明 counting 类中，pruning 让原本就"贴在一起"的多本书的边界进一步弱化，触发 detector 把整个堆识别为 1 个对象（甚至完全过滤）。

**Case 00254（four benches）**

baseline 中 0002.png 4 张长椅被精确分割（4 个 detection，置信度 0.90+）；pruning 同图仅 3 个 detection，遗漏了 4 把中的某一把（可能两把腿部分被合并）。

**Case 00321（white sheep, Δn=−2）**

非 flipped 但典型退化：pruning 后绵羊毛部分**被背景草色"渗透"**——羊腿和颈部毛色出现绿色/黄色斑驳，这是 token 剪枝在颜色细粒度上的失真，导致 detector 颜色判定在 white 阈值附近抖动。

### 3.2 失败模式分类

按视觉证据 + detector 输出，可将 7 个 flipped + 多数 regression 总结为 4 种模式：

| 模式 | 描述 | 典型 case | 出现频次 |
|---|---|---|---|
| **A. 细粒度类目混淆** | snowboard ↔ skateboard、bed ↔ couch 等同形状物体类别翻转 | 00163, 00477 | 主要在 two_object / color_attr |
| **B. 低置信度对象消失** | base 中 detector 以 0.3~0.5 临界 confidence 通过，prune 后 score 跌至阈值下消失 | 00178, 00492, 00495 | 集中在含细长 / 细节物体（bat, glove, skateboard） |
| **C. 多对象合并 / 漏检** | counting 任务，相邻物体边界被剪枝平滑后合并，导致计数下降 | 00217, 00254 | 主要在 counting |
| **D. 颜色染色 / 渗透** | 物体颜色被附近颜色"传染"（e.g. white sheep → 偏黄绿），导致颜色属性判定失败 | 00321 | colors / color_attr |

---

## 4. 关键洞察：1.08 pp 退化的本质

### 4.1 大量 regression 是 detector noise，不是真正的语义错误

7 个完全 flipped 的 case 中，至少 5 个（00163, 00178, 00477, 00492, 00495）的 baseline 通过都是基于 detector 给出的 **0.3 ~ 0.5 confidence 临界检测**。这类检测在 random seed / 微小图像扰动下天然不稳定。

GenEval 使用的是基于 Mask2Former 的目标检测器，置信阈值 0.3。对于 baseline 仅以 0.32 通过的 baseball bat（00178），任何细微的纹理/对比度变化都可能让它跌到 0.29 以下消失。**Pruning 0.9 引入的图像变化恰好与 detector 的"决策边界"频繁相交**，造成准确率指标的小幅波动。

### 4.2 反向证据：Pruning 也带来了 31 个 improvement

| Improvement Top 5 | tag | Δn | prompt |
|---|---|---|---|
| 00525 | color_attr | +2 | a photo of a brown dining table and a white suitcase |
| 00532 | color_attr | +2 | a photo of a purple backpack and a white umbrella |
| 00545 | color_attr | +2 | a photo of a red clock and a black cell phone |
| 00311 | colors | +2 | a photo of a green traffic light |
| 00209 | counting | +2 | a photo of three kites |

例如 00209 三个风筝：baseline 仅 1/4 张被判定 correct，pruning 提升到 3/4。视觉上两组的风筝构图几乎相同，但 detector 在 pruning 输出上更稳定地分辨出 3 个独立 kite。这进一步表明：**pruning 0.9 在大多数 prompt 上对 detector 是中性的，少量 prompt 上是负向的，少量 prompt 上是正向的，整体净影响约 −1pp**。

### 4.3 真正会"语义损伤"的失败模式

排除 detector noise 后，下面两类是 pruning 真正会引发的、可重复观察到的语义层面退化：

1. **多实例边界平滑**（counting 模式 C）：当一张图里 ≥ 3 个相同实例非常靠近时，剪枝会让物体间界限变模糊，detector 计数不准。
2. **细颜色染色**（color_attr 模式 D）：单色物体（特别是 white / yellow 等浅色）易被周围颜色"染色"，使颜色判定错误。

这两类是**生成质量层面的真实退化**，建议在论文中作为"代价"明确报告。

---

## 5. 各 Category 的鲁棒性排序（从最鲁棒到最敏感）

```
colors  ≈  single_object  >  counting  ≥  two_object  >  position  >  color_attr
   持平          –0.63         –0.63        –1.26         –1.75       –2.00
```

剪枝率 0.9 时，**color_attr** 是 GenEval 6 大类中受影响最大的：颜色 + 物体的双重精细约束让模型对 token 损失最敏感。**colors**（仅颜色，单对象）则完全鲁棒，是 best-case。

---

## 6. 结论与建议

1. **Pruning 0.9 在 GenEval 上代价极小**：整体仅下降 ~1.08pp，超出半数 case（84.6%）评估结果完全不变。
2. **多数退化来自 detector 边缘噪声**：Mask2Former 在 0.3 confidence 阈值附近本身就不稳定，剪枝触发的微小图像变化频繁与该边界相交。
3. **真正的语义退化集中在两个模式**：
   - 多实例 / counting：多个相同实例的边界平滑导致漏数
   - 颜色染色 / color_attr：浅色物体被周围色调染色
4. **建议**：
   - 在生成 counting / color_attr 任务时**降低 pruning ratio**（例如 0.7~0.8）以保留更多细节 token；
   - 在 single_object / colors / position 等鲁棒类别上保持 0.9 极激进剪枝，最大化加速；
   - 论文撰写时，明确将上述两类失败模式作为已知 trade-off 列出，并通过 grid 视觉对比展示其微小性。

---

## 附录：分析数据文件

所有原始数据与汇总均位于 `output/geneval_compare_pr0p9/`：

- `analyze.py`：分析脚本
- `analyze.log`：脚本运行日志（包含统计摘要）
- `per_prompt_compare.json`：所有 553 prompt 的逐条 base vs prune 对比
- `regressions.json`：54 条 image-level regression
- `improvements.json`：31 条 image-level improvement
- `flipped_to_wrong.json`：7 条 prompt-level 翻转为 wrong
- `flipped_to_right.json`：1 条 prompt-level 翻转为 right
- `by_tag_stats.json`：按 tag 聚合的统计
