# 等宽向量（任务 A）与 AGFL 层对照（任务 B）报告

生成时间：2026-11-17
执行目录：`D:\实时测谎系统\Code_fixed`
数据源：`prepared_v3_features`（4701 试次 / 34 被试 / 126 导 / 256 Hz / 窗 −0.2~1.5 s）
依据：`AI指令_第四轮_等宽向量与AGFL对照_20261117.md`

---

## 1. 结论速览

| 项目 | 结果 |
|---|---|
| 任务 A：20 个向量是否每个 16 维 | **是**，拼接宽度 256（eeg）/ 320（eeg_physiology）/ 64（physiology） |
| 任务 A：纯 eeg 与 v3.1 是否逐位一致 | **是**，pooled 0.5324 / 被试均值 0.5302，逐试次概率差 = 0 |
| 任务 B：五组配置（within 33 人） | none 0.5082 / k0 0.5070 / intra 0.5102 / inter 0.5113 / combined 0.5083 |
| 任务 B：`intra` vs `none` 增益 | **+0.20 个百分点**，95% CI ±1.54，p = 0.79 |
| 任务 B：A/B 两组方向 | A +0.24 / B +0.15 个百分点，CI 均跨 0 → **未观察到可复现提升** |
| 任务 B：k0 vs none | within −0.12 点（通过 §3.6 的 0.5 点判据）；LOSO −1.03 点（超出，见 §5.3） |
| 标签打乱 | intra 0.4978 / none 0.4990 → 回到随机，**无泄漏** |
| 判定 | **AGFL 跑通了，但不有效**：在无稳定信号的输入上，图滤波没有提供可迁移的判别信息 |

---

## 2. 任务 A：等宽向量

### 2.1 改动

| 文件 | 改动 |
|---|---|
| `deception/evaluation.py` | 新增 `_EqualWidthProjector(width)`：`f_g > width` 走折内 PCA，`f_g < width` 零填充，输出宽度恒为 `width`；`_group_scaler` 新增 `equal_width=False`（默认 False，旧命令行为不变）；新增 `_report_vector_widths` 打印宽度表 |
| `deception/cli.py` | 新增 `--equal-width-vectors` |
| 未改动 | `features.py`、`feature_columns`、`_vector_groups` 的 16 组定义、PCA 0.95、模型清单、阈值网格、折划分、seed |

### 2.2 宽度表（feature_set = eeg / eeg_physiology）

| 组 | 原始宽度 | 输出宽度 | 处理 |
|---|---|---|---|
| eeg_sw1_theta | 126 | 16 | PCA 16 |
| eeg_sw1_alpha | 126 | 16 | PCA 16 |
| eeg_sw1_beta | 126 | 16 | PCA 16 |
| eeg_sw1_gamma | 126 | 16 | PCA 16 |
| eeg_sw2_theta / alpha / beta / gamma | 126 | 16 | PCA 16 |
| eeg_sw3_theta / alpha / beta / gamma | 126 | 16 | PCA 16 |
| eeg_sw4_theta / alpha / beta / gamma | 126 | 16 | PCA 16 |
| ecg | 6 | 16 | **pad 10** |
| rsp | 6 | 16 | **pad 10** |
| eda | 6 | 16 | **pad 10** |
| emg | 12 | 16 | **pad 4** |
| **拼接** | — | **320** | 16×16 + 4×16 |

纯 `eeg` 时拼接宽度 256；纯 `physiology` 时 64。

### 2.3 一致性证据

| 指标 | v3.1 | 任务 A | 差异 |
|---|---|---|---|
| eeg / logreg pooled balanced_acc | 0.5324 | **0.5324** | 0 |
| eeg / logreg 被试均值（33 人） | 0.5302 ± 0.0562 | **0.5302 ± 0.0562** | 0 |
| eeg / lda pooled | 0.5238 | 0.5238 | 0 |
| 逐试次概率差（eeg/logreg） | — | — | **max = 0.000000** |

`eeg` 全部逐位一致。

**生理相关特征集有 14 个试次的差异**（`sub-S210208` 全部试次），原因已定位：旧 `_GroupProjector` 在 `n_samples−1 < f_g` 时会对**较窄的组**也做 PCA——S210208 每折仅 11 个训练样本，于是 12 维的 EMG 组被压到 10 维。新 `_EqualWidthProjector` 按 §2.2 规定「`f_g < 16` 一律零填充、不 PCA」，因此在这一点上行为不同。

这是任务 A 修正了旧实现的退化路径，不是回归：

| 特征集 | v3.1 33 人均值 | 任务 A 33 人均值 | v3.1 34 人均值 | 任务 A 34 人均值 | 差异试次 |
|---|---|---|---|---|---|
| physiology / lda | 0.5123 | **0.5123** | 0.5103 | 0.5146 | 14（全部为 S210208） |
| eeg_physiology / lda | 0.5234 | **0.5234** | 0.5211 | 0.5195 | 14（全部为 S210208） |

33 人主表完全一致；只有 `sub-S210208` 的 14 个试次因该退化路径被修正而变化。

### 2.4 双协议结果（33 人主表）

**被试内 5 折（主表）**

| 特征集 | 模型 | pooled bacc | AUC | 被试均值 ± SD | 95% CI |
|---|---|---|---|---|---|
| eeg | logreg | **0.5324** | 0.5328 | **0.5302 ± 0.0562** | ±0.0199 |
| eeg_physiology | lda | 0.5302 | 0.5396 | 0.5234 ± 0.0491 | ±0.0174 |
| eeg | lda | 0.5238 | 0.5380 | 0.5179 ± 0.0571 | ±0.0202 |
| eeg_physiology | logreg | 0.5236 | 0.5335 | 0.5219 ± 0.0458 | ±0.0162 |
| physiology | lda | 0.5200 | 0.5303 | 0.5123 ± 0.0474 | ±0.0168 |
| physiology | logreg | 0.5101 | 0.5211 | 0.5114 ± 0.0505 | ±0.0179 |
| — | dummy | — | — | 0.4998 ± 0.0010 | ±0.0003 |

**跨被试 LOSO（对照）**

| 特征集 | 模型 | pooled bacc | AUC | 被试均值 ± SD | 95% CI |
|---|---|---|---|---|---|
| eeg | logreg | 0.5068 | 0.5208 | 0.5059 ± 0.0365 | ±0.0127 |
| eeg_physiology | lda | 0.5064 | 0.5184 | 0.5051 ± 0.0327 | ±0.0116 |
| eeg | lda | 0.5025 | 0.5201 | 0.5046 ± 0.0321 | ±0.0114 |
| eeg_physiology | logreg | 0.5006 | 0.5190 | 0.4978 ± 0.0335 | ±0.0119 |
| physiology | lda | 0.4908 | 0.4969 | 0.4928 ± 0.0217 | ±0.0077 |
| physiology | logreg | 0.4902 | 0.4990 | 0.4913 ± 0.0221 | ±0.0078 |
| — | dummy | 0.5000 | 0.5000 | 0.5000 ± 0.0000 | ±0.0000 |

34 人附录见 `results\v4b_task1_summary.csv`。

---

## 3. 任务 B：AGFL 层

### 3.1 实现与公式对应

| 公式 | 实现 | 文件 |
|---|---|---|
| `H = LN(F P + 1 bᵀ)` | `AGFLLayer` 输入处 `layer_norm(h + bias)`，`bias` 可学 | `deception/agfl.py` |
| `S = (H Wq)(H Wk)ᵀ / √a` | `Wq/Wk: Linear(d, a, bias=False)`，除以 `sqrt(a)` | 同上 |
| `A_ij ∝ exp(S_ij)`，只保留 top-κ | `masked_topk_softmax`：先 `masked_fill` 再 `topk` 再 softmax | 同上 |
| 每行至少一个合法邻居，否则报错 | `legal.min() == 0` → `ValueError` | 同上 |
| `V⁰ = H Wv`；`U^l = A V^{l−1}` | `Wv: Linear(d, d, bias=False)`，循环 `hops` 次 | 同上 |
| 逐节点范数控制 | `v = u * ‖v0‖ / (‖u‖ + 1e-6)` | 同上 |
| `α = softmax(c)`，`T = Σ α_l V^l` | 可学 `c ∈ R^{K+1}` | 同上 |
| `G(H) = LN(H + T Wo)` | `layer_norm(h + Wo(T))` | 同上 |

设计选择（§3.2 允许、需在报告注明）：
- **自环排除**（`build_edge_mask` 对角线置 False）
- **同一套层参数在两模态图间共享**（intra 时 EEG 16 节点与生理 4 节点共用 `Wq/Wk/Wv/Wo/c/b`，约 1.1k 参数）
- **`combined`** = intra（两张模态内图）→ inter（20 节点单图）
- **`k0`** = 层存在但 `hops=0`，不构造邻接；`none` = 完全无层

### 3.2 六条单测

`tests/test_agfl.py`，**7 项全部通过**：

1. 邻接每行和为 1、非负、至多 κ 个非零、对角线为 0
2. 掩码在 top-κ 之前生效（被屏蔽的高分边权重为 0）
3. 块对角掩码无跨模态边
4. 某节点合法邻居为 0 时抛错
5. `K=0` 时换一个掩码输出逐元素不变
6. 输出形状/dtype 正确、无 NaN、`‖V^l_i‖ ≤ ‖V⁰_i‖`
7. 同 seed 两次前向完全一致

### 3.3 训练配置（与 §3.4 一致）

编码器折内 `StandardScaler + PCA(16)`／零填充；PCA 0.95 折内拟合于编码器输出，AGFL 输出用同一基；`Linear(m→2)`；`CrossEntropyLoss(weight=n/(2·count_c))`；Adam lr 1e-3、wd 1e-2；60 epochs 全批量、无 early stopping；seeds 2026/2027/2028；CPU、4 线程；折划分复用 `_outer_splits`；指标复用 `deception/metrics.py`。

### 3.4 主表（within，33 人，3 seed 平均）

| 配置 | pooled bacc | AUC | 被试均值 ± SD | 95% CI | Δ vs none | Δ vs 官方基线 |
|---|---|---|---|---|---|---|
| 官方基线（sklearn v3.1 eeg/logreg） | 0.5324 | 0.5328 | 0.5302 ± 0.0562 | ±0.0199 | — | — |
| `none` | 0.5090 | 0.5106 | 0.5082 ± 0.0381 | ±0.0135 | — | −2.20 点 |
| `k0` | 0.5079 | 0.5152 | 0.5070 ± 0.0424 | ±0.0150 | −0.12 点 | −2.32 点 |
| `intra` | **0.5122** | **0.5155** | **0.5102 ± 0.0434** | ±0.0154 | **+0.20 点** | −2.00 点 |
| `inter` | 0.5127 | 0.5161 | 0.5113 ± 0.0422 | ±0.0150 | +0.31 点 | −1.89 点 |
| `combined` | 0.5097 | 0.5149 | 0.5083 ± 0.0419 | ±0.0149 | +0.01 点 | −2.19 点 |

配对检验（逐被试，相对 `none`）：

| 配置 | 平均增益 | 95% CI | p |
|---|---|---|---|
| k0 | −0.0012 | ±0.0151 | 0.870 |
| intra | +0.0020 | ±0.0154 | 0.794 |
| inter | +0.0031 | ±0.0150 | 0.679 |
| combined | +0.0001 | ±0.0145 | 0.988 |

### 3.5 LOSO 对照表（33 人，3 seed 平均）

| 配置 | pooled bacc | AUC | 被试均值 ± SD | 95% CI | Δ vs none |
|---|---|---|---|---|---|
| `none` | 0.5146 | 0.5177 | 0.5137 ± 0.0271 | ±0.0096 | — |
| `k0` | 0.5014 | 0.5063 | 0.5034 ± 0.0278 | ±0.0099 | −1.03 点 |
| `intra` | 0.5069 | 0.5088 | 0.5082 ± 0.0276 | ±0.0098 | −0.55 点 |
| `inter` | 0.5058 | 0.5080 | 0.5069 ± 0.0280 | ±0.0099 | −0.68 点 |
| `combined` | 0.5065 | 0.5103 | 0.5066 ± 0.0275 | ±0.0097 | −0.71 点 |

配对检验：k0 −0.0103（CI ±0.0107，p=0.059）；intra −0.0055（p=0.310）；inter −0.0068（p=0.209）；combined −0.0071（p=0.225）。

### 3.6 必做对照

**① K=0 消融**

| 协议 | none | k0 | 差值 | 是否 < 0.5 点 |
|---|---|---|---|---|
| within | 0.5082 | 0.5070 | −0.12 点 | 是 |
| LOSO | 0.5137 | 0.5034 | **−1.03 点** | **否** |

LOSO 超出 0.5 点判据，已按 §3.6 先查代码，结论见 §5.3。

**② 标签打乱（intra，seed 2026，within）**

| 配置 | 打乱后 pooled bacc | 被试均值 |
|---|---|---|
| none | 0.4993 | 0.4990 ± 0.0430 |
| intra | 0.4982 | 0.4978 ± 0.0424 |

回落到 0.48–0.52，**无泄漏**。

**③ A/B 被试闸门（intra − none，within）**

| 组 | n | 平均增益 | SD | 95% CI | p |
|---|---|---|---|---|---|
| A（编号偶数） | 17 | +0.0024 | 0.0407 | ±0.0209 | 0.807 |
| B（编号奇数） | 16 | +0.0015 | 0.0476 | ±0.0254 | 0.900 |

两组同向但 CI 均跨 0，且幅度 < 0.3 点 → **判为未观察到可复现提升**。

**④ 灵敏度 d=8（within，seed 2026）**

| 配置 | pooled bacc | 被试均值 ± SD |
|---|---|---|
| none | 0.5172 | 0.5164 ± 0.0515 |
| intra | 0.5075 | 0.5059 ± 0.0475 |

`intra` 在 d=8 下仍低于 `none`，结论不依赖宽度选择。

---

## 4. 判定（按 §6 预写死的表）

四组配置与 `none` 的差值全部 **< 0.02**（最大 0.0031），A/B 两组 CI 均跨 0，打乱标签回到随机。对应 §6 第一行：

> **图滤波未提供可迁移的判别信息。**

这与 `负结果报告.md` §3 的分半信度 ≈ 0 互相印证：如果输入里没有稳定的锁时类差，任何图结构都没有可交换的信息。

同时必须写清两条边界：

1. **AGFL 跑通了**（单测 7/7、五组配置、双协议、3 seed 全部完成，coverage = 1.0），但**不代表 AGFL 有效**。
2. torch 同源基线 `none`（0.5082）比 sklearn 官方基线（0.5302）低 2.2 点，说明两者不是同一个模型（前者是线性 softmax 头，后者是 shrinkage LDA + 超参调优）。**Δ vs none 与 Δ vs 官方基线不能混用**。

---

## 5. 限制与说明

### 5.1 与第三轮的冲突

第三轮 §4.3 曾要求「生理保持原生宽度、不要动」。本轮按第四轮 §2.1 改为零填充到 16，生理组的等宽改造**只影响 `sub-S210208` 的 14 个试次**（旧投影在其 11 样本折上把 12 维 EMG 压到 10 维）。其余 33 人结果与 v3.1 逐位一致。

### 5.2 编码器与 PCA

- 编码器（StandardScaler + PCA(16)/零填充）在每个外折训练集上 `fit`，测试折只 `transform`
- 全局 PCA 0.95 拟合于训练折编码器输出；AGFL 输出用同一基投影
- 没有任何全量预导出（红线 1）

### 5.3 LOSO 的 k0 − none = −1.03 点

按要求先查了代码与数值，结论是**不是实现错误**：

1. 三个 seed 的被试级 SD 为 2.7 点，1.03 点约为 0.4 SD；配对检验 p=0.059，CI 刚好擦过 0。
2. `k0` 与 `none` 的差别按定义就是「多一层 LN + 残差 + Wo + 可学 bias」。LN 会把 16 维 PCA token 重新归一化，这是结构性的表示变化，不是梯度 bug。
3. 单测 5（`K=0` 时换掩码输出不变）与单测 7（确定性）均通过；训练损失单调下降，无 NaN。
4. within 协议下差值为 −0.12 点，符合预期；LOSO 训练样本更多、模型更容易拟合训练被试的特异结构，因此加一层后的退化更明显。

结论：这一差值属于「层内 LN/残差带来的容量与优化差异」，不是需要回滚的实现缺陷；但它也说明 **`k0` 不是 `none` 的无害替代**，比较 AGFL 增益时应以 `none` 为准、`k0` 只作辅助。

### 5.4 其它

- `combined` < `intra`（0.5083 < 0.5102），与「跨模态边稀释模态内结构」一致，也和生理组稀释 EEG（0.5234 < 0.5302）同向，属可解释。
- 34 人附录（含 `sub-S210208`）见各 `*_summary.csv`。
- 未做早停、未用测试折调参、未用测试折选 seed/d/κ/K。

---

## 6. 复现命令

```powershell
$py = "D:\实时测谎系统\Code_fixed\.venv\Scripts\python.exe"
$in = "D:\实时测谎系统\Code_fixed\prepared_v3_features"
$common = "--pca-components 0.95 --inner-splits 5 --thresholds 0.40 0.45 0.50 0.55 0.60 --seed 2026 --n-jobs -1"

# 任务 A
& $py -m deception benchmark --input $in --output "D:\实时测谎系统\Code_fixed\results\v4b_task1_within" `
  --feature-sets eeg physiology eeg_physiology --models lda logreg --grouping within `
  --per-modality-scaling --equal-width-vectors --vector-groups --vector-width 16 $common

& $py -m deception benchmark --input $in --output "D:\实时测谎系统\Code_fixed\results\v4b_task1_loso" `
  --feature-sets eeg physiology eeg_physiology --models lda logreg --grouping loso `
  --per-modality-scaling --equal-width-vectors --vector-groups --vector-width 16 $common

# 任务 B
& $py -m deception agfl --input $in --output "D:\实时测谎系统\Code_fixed\results\v4b_agfl_within" `
  --grouping within --configs none k0 intra inter combined --seeds 2026 2027 2028 `
  --d 16 --a 16 --hops 2 --kappa 5 --epochs 60 --lr 0.001 --weight-decay 0.01

& $py -m deception agfl --input $in --output "D:\实时测谎系统\Code_fixed\results\v4b_agfl_loso" `
  --grouping loso --configs none k0 intra inter combined --seeds 2026 2027 2028 `
  --d 16 --a 16 --hops 2 --kappa 5 --epochs 60 --lr 0.001 --weight-decay 0.01

& $py -m deception agfl --input $in --output "D:\实时测谎系统\Code_fixed\results\v4b_agfl_d8" `
  --grouping within --configs none intra --seeds 2026 --d 8 --a 8 --hops 2 --kappa 5

& $py -m deception agfl --input $in --output "D:\实时测谎系统\Code_fixed\results\v4b_agfl_shuffled" `
  --grouping within --configs none intra --seeds 2026 --shuffle-labels
```

---

## 7. 自检表（对应 §9）

| # | 检查项 | 回答 |
|---|---|---|
| 1 | 20 个向量是否每个 16 维？宽度表是否列出 padding？ | 是，§2.2，ecg/rsp/eda 各 pad 10、emg pad 4 |
| 2 | 纯 eeg 与 v3.1 是否逐位一致？ | 是，0.5324 / 0.5302，逐试次概率差 0 |
| 3 | within 与 LOSO 是否都跑了？ | 是 |
| 4 | 是否同时提供 sklearn 官方基线与 torch `none`？差值多少？ | 是，0.5302 vs 0.5082，差 2.20 点 |
| 5 | `k0` 与 `none` 差值？是否 < 0.5 点？ | within −0.12（是）；LOSO −1.03（否，已查，见 §5.3） |
| 6 | 标签打乱后是否回落到 0.48–0.52？ | 是，0.4978 / 0.4990 |
| 7 | A/B 两组方向是否一致？CI 是否给出？ | 同向（+0.24 / +0.15 点），CI 均跨 0 |
| 8 | 折划分是否复用 `_outer_splits`？ | 是，未重写 |
| 9 | 标准化 / PCA / 图构造是否只在训练折 fit？ | 是，编码器与 PCA 基均折内拟合 |
| 10 | 6 条单测是否全绿？`run_checks` 失败项是否仍只有 2 个 memmap？ | 7/7 通过；107 项测试、2 个 memmap 失败 |
| 11 | 是否避免「融合有效」「SOTA」类表述？ | 是 |
| 12 | 若任一配置 > 0.70 是否先查泄漏？ | 无任何配置 > 0.70 |
