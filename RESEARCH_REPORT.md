# 在质量感知训练下保持长尾 PID 透出率的研究报告

**作者**: zhouzidie  
**日期**: 2026-08-12  
**状态**: 进行中（实验数据待补充）

---

## 1. 研究背景与动机

### 1.1 推荐系统中的长尾问题

短视频推荐系统中，内容（PID）的交互频次呈典型的幂律分布：少数头部 PID 占据绝大多数曝光，而大量长尾 PID（低交互频次视频）几乎得不到系统性推荐机会。以本研究数据集 VK-LSVD 为例，5-core 过滤后仍有 41,656 个 PID，其中大量 PID 在训练集中出现次数极少。

长尾 PID 曝光不足的根本原因**不是内容质量低，而是系统性的马太效应**：

```
新视频无历史数据
    → 推荐系统不敢推
    → 没有曝光就没有数据
    → 下一轮训练仍然稀疏
    → 马太效应不断强化
```

### 1.2 RSFT 引入的新问题

近年来，强化学习式微调（RSFT，Reinforcement Learning from Supervised Fine-Tuning）被引入推荐系统训练流程，核心思路是：用用户真实反馈（完播率、点赞、分享等）构造 reward 信号，只保留高 reward 的训练样本，过滤掉低质量互动数据。

RSFT 的动机是合理的：低质量互动（用户快速划走、误触等）会引入噪声，污染模型对用户偏好的学习。

然而，RSFT 存在一个**系统性偏见**：

- 长尾 PID 的样本少 → 每个 reward bucket 内的统计样本不足 → reward 估计噪声大
- 即使是高质量的长尾视频，其 reward 估计方差也比头部视频大得多
- RSFT 过滤时，高方差的长尾 PID 更容易被误判为低质量而丢弃

结果：**RSFT 在过滤低质量数据的同时，不成比例地丢失了长尾 PID 的训练覆盖**。

### 1.3 研究问题

> 在生成式推荐系统（以 TIGER 为代表）中，RSFT 训练数据过滤是否导致长尾 PID 的召回覆盖率下降？如果是，如何在保持数据质量的前提下，恢复或提升长尾 PID 的透出率？

---

## 2. 相关工作

### 2.1 生成式推荐系统

TIGER（Text-ID Generative Recommendation）将推荐问题转化为序列生成任务：
- 通过 RQ-VAE 将每个 item 编码为层级语义 ID（SID），由 4 个 token 组成
- 用 T5 模型的编码器接受历史序列，解码器通过 beam search 自回归生成目标 SID
- 天然支持泛化到未见 item（cold start），理论上有利于长尾推荐

然而，beam search 生成过程存在固有偏向：高频 PID 对应的 SID token 在训练时出现次数多，模型对其概率估计更准确，导致 beam search 优先生成头部 PID。

### 2.2 长尾推荐的已有方法

| 方法类别 | 代表工作 | 核心思路 | 局限性 |
|---------|---------|---------|--------|
| 去偏训练 | IPS, DICE | 用逆倾向得分重加权样本 | 需要曝光概率估计，实际难以获取 |
| 长尾重采样 | Tail-Net | 对低频 item 上采样 | 可能引入噪声，损害整体质量 |
| 解耦表示学习 | ESAM | 分离流行度和内容表示 | 模型结构复杂，难以与生成式模型结合 |
| 后处理重排 | MMR, DPP | 推理时增加多样性 | 对召回阶段无帮助 |

本研究的特殊性在于：问题不来自固有的曝光偏差，而来自**RSFT 数据过滤对长尾 PID 的系统性损伤**，已有方法均未直接针对这一场景。

### 2.3 RSFT 在推荐系统中的应用

RLHF/RSFT 在 LLM 对齐领域取得显著成果，被借鉴到推荐系统。现有工作（如 CTRL、RLRF）主要关注整体推荐质量提升，对长尾 PID 的副作用研究较少。本研究填补这一空白。

---

## 3. 诊断实验

### 3.1 实验设计

在 TIGER 框架下，设计三组对照实验：

| 实验组 | 过滤方式 | 数据量变化 | PID 覆盖率预期 |
|--------|---------|-----------|--------------|
| **Baseline** | 无过滤 | 100% | 100% |
| **A: Random Drop** | 随机丢弃 20% 样本 | 80% | ≈100% |
| **B: PID Drop** | 随机丢弃 20% 的 PID | <80% | 80% |
| **C: RSFT Drop** | 丢弃 reward < 0.2 的样本 | ≈80% | 待测量 |

**核心指标**：
- `recalled_pid_coverage`：beam search 结果中覆盖的唯一 PID 数 / 总 PID 数
- `Recall@K`（全量测试集）：整体召回率
- `Good_Recall@K`（高 reward 测试集，reward ≥ 0.2）：对高质量互动的召回
- `Bad_Recall@K`（低 reward 测试集，reward < 0.2）：对低质量互动的召回

**reward 定义**：
```
bucket = floor(log(duration) / log(1.2))
reward = timespent 在 bucket 内的分位数百分比
```
按时长分桶计算分位数，避免不同时长视频间的直接比较偏差（21 个 bucket，覆盖 5-180s）。

### 3.2 数据集

- **来源**: VK-LSVD（VK 平台短视频日志，1% 采样）
- **规模**: 6,392 用户，41,656 item，519,469 训练样本（5-core 过滤后，leave-one-out 划分）
- **测试集**: 6,392 条（全量）/ 4,825 条（test_good, reward ≥ 0.2）/ 1,567 条（test_bad, reward < 0.2）

### 3.3 已有结果

**Baseline**（epoch 69 early stopping，89 epoch 总训练）：

| 指标 | 值 |
|------|-----|
| `Recall@10` | 0.0202 |
| `NDCG@10` | 0.0105 |
| `Good_Recall@10` | 0.0214 |
| `Bad_Recall@10` | 0.0153 |
| `recalled_pid_coverage` | 8.89% (3,706/41,656) |

**观察**：
1. Baseline 的 `recalled_pid_coverage` 仅 8.89%，说明 TIGER 在全量数据下已存在严重的长尾覆盖不足
2. `Good_Recall > Bad_Recall`，验证了高 reward 互动更容易被召回（符合预期）
3. 实验 A/B/C 的结果待完成

### 3.4 预期假设

- **H1**: 实验 A（随机 drop）与 Baseline 指标接近（数据量减少但 PID 覆盖不变）
- **H2**: 实验 B（PID drop）`recalled_pid_coverage` 显著下降（直接减少了 20% PID）
- **H3**: 实验 C（RSFT drop）`recalled_pid_coverage` 下降幅度 > 实验 A，且长尾损失主要来自低频 PID 的丢弃

若 H3 成立，则证明 RSFT 对长尾召回存在系统性伤害，为后续改进提供实证依据。

---

## 4. 问题分析：为什么 RSFT 会伤害长尾

### 4.1 reward 估计的稀疏性偏差

设某 PID $i$ 在训练集中出现 $n_i$ 次，其 bucket 内总样本数为 $N_b$。

reward 估计误差（标准误）约为：
$$\text{SE}(r_i) \approx \frac{1}{\sqrt{n_i}}$$

对于头部 PID（$n_i = 1000$），SE ≈ 0.032；对于长尾 PID（$n_i = 5$），SE ≈ 0.447。

RSFT 用 reward < 0.2 作为过滤阈值时，长尾 PID 因 SE 过大，大量真实 reward ≥ 0.2 的样本被误判为低质量。

### 4.2 长尾 PID 在 SID 空间的稀疏性

RQ-VAE 训练时，低频 PID 的重建误差更高，对应的 SID token 在 codebook 中共享更多（碰撞），导致：
- 长尾 PID 的 SID 与其他 PID 区分度低
- beam search 生成长尾 SID 需要多步精确匹配，概率被路径乘法放大损耗
- RSFT 进一步减少长尾 PID 的训练频次后，这一问题加剧

### 4.3 马太效应的复利

$$\text{Coverage}_{t+1} = f(\text{Coverage}_t, \text{RSFT}(\text{Coverage}_t))$$

RSFT 的反馈循环：每一轮过滤都使长尾 PID 在下一轮的 reward 估计更不可靠，形成持续恶化的正反馈。

---

## 5. 改进方案

### 方案一：reward 加权损失（推荐）

**核心思路**：不丢弃样本，而是用 reward 作为样本权重调整梯度贡献。

```python
# 当前 RSFT：丢弃低 reward 样本
filtered = [x for x in dataset if x.reward >= threshold]

# 改进：reward 加权损失
loss = cross_entropy(logits, target_ids)  # shape: (batch,)
weights = torch.tensor([x.reward for x in batch])  # 归一化到 [0, 1]
weighted_loss = (loss * weights).mean()
```

**优势**：
- PID 覆盖率完全不受影响（不丢弃任何样本）
- 高 reward 样本对梯度贡献更大，保持质量导向
- 低频 PID 即使 reward 低，仍有机会被学习

**潜在问题**：reward 估计噪声直接传导到训练权重，对长尾 PID 的稀疏 reward 需要做平滑处理。

**改进版**：贝叶斯平滑 reward
```python
# 对低频 PID 做 reward 平均回归（向全局均值平滑）
GLOBAL_MEAN = 0.5
SMOOTH_COUNT = 20  # 等效先验样本数

smoothed_reward = (n_samples * reward + SMOOTH_COUNT * GLOBAL_MEAN) / (n_samples + SMOOTH_COUNT)
```

### 方案二：长尾保护采样

**核心思路**：对低频 PID 强制保留最低样本量，不参与 RSFT 过滤。

```python
# RSFT 过滤时跳过低频 PID
COLD_THRESHOLD = 20  # 交互次数 < 20 视为冷启动 PID

def rsft_filter(sample):
    pid = sample.target
    if pid_freq[pid] < COLD_THRESHOLD:
        return True  # 无论 reward 如何，强制保留
    return sample.reward >= REWARD_THRESHOLD
```

**优势**：简单直接，不改变模型结构  
**缺点**：需要调整 `COLD_THRESHOLD`，且对 reward 质量仍未利用

### 方案三：内容感知 SID 编码

**核心思路**：RQ-VAE 除使用交互历史 embedding 外，加入视频内容特征（帧特征、标题），减少对交互频次的依赖。

对于长尾 PID：
- 交互 embedding 稀疏 → 重建质量差
- 内容 embedding 可以从视觉/文本模态获取，不受频次影响

内容感知 SID 使得即使没有历史交互，长尾 PID 仍能获得有区分度的 SID，beam search 可以正确识别。

**现实可行性**：VK 数据集包含视频封面和标题，可以用 CLIP/BERT 提取内容 embedding 与交互 embedding 融合。

### 方案四：推理阶段长尾增强

在 beam search 推理时，对对长尾 PID 的第一个 SID token 增加 log-probability bonus：

```python
# 标准 beam search scores
scores = model(input_ids)  # (batch, vocab_size)

# 长尾增强：对低频 PID 对应的 token 加 bonus
for token_id, pids in token_to_pids.items():
    if any(pid_freq[pid] < TAIL_THRESHOLD for pid in pids):
        scores[:, token_id] += TAIL_BONUS  # e.g., 0.5

next_tokens = scores.topk(beam_size)
```

**优势**：不改变训练流程，可作为推理时的轻量级 patch  
**缺点**：bonus 超参难以调整，可能影响头部 PID 的精度

---

## 6. 实验规划

### 6.1 当前实验进度

| 实验 | 状态 | 说明 |
|------|------|------|
| Baseline | ✅ 完成 | `Recall@10=0.0202`, `pid_cov=8.89%` |
| A: Random Drop 20% | 🔄 进行中（epoch 12/200） | |
| B: PID Drop 20% | ⏳ 待启动 | |
| C: RSFT Drop 20% | ⏳ 待启动 | |

### 6.2 后续实验规划

**阶段二**（诊断结论成立后）：验证改进方案效果

| 实验 | 对比对象 | 核心指标 |
|------|---------|---------|
| Reward-weighted Loss | Baseline vs RSFT Drop | `pid_cov`, `Good_Recall@10` |
| Cold-PID Protection | RSFT Drop | `pid_cov` for freq<20 PIDs |
| Content-aware SID | Baseline | long-tail `Recall@10` |

**评估维度**：
1. 整体召回质量不应显著下降（`Recall@10` 与 Baseline 差距 < 5%）
2. 长尾 PID 透出率提升（`recalled_pid_coverage` 上升）
3. 高质量互动召回保持（`Good_Recall@10` 不低于 RSFT Drop）

---

## 7. 预期结论与意义

### 7.1 预期结论

1. **诊断结论**：RSFT 以 X% 的长尾 PID 覆盖损失换取 Y% 的整体质量提升，其中长尾 PID 损失不成比例于数据量减少（对比实验 A 与 C 的差异）

2. **机制解释**：损失主要来自低频 PID 的 reward 估计偏差，而非真实质量差异

3. **改进方案**：reward 加权损失在保持质量的前提下，将长尾覆盖率恢复至接近 Baseline 水平

### 7.2 学术意义

- 首次系统量化 RSFT 对生成式推荐系统长尾覆盖率的影响
- 提出 reward 加权损失作为 RSFT 的无损替代方案
- 揭示"数据质量过滤"与"内容多样性"之间的 trade-off 机制

### 7.3 工业意义

- 指导短视频平台 RSFT 流程的长尾保护策略设计
- 提供可量化的 `recalled_pid_coverage` 指标用于监控系统多样性
- 减少优质新创作者的"冷启动壁垒"，有利于内容生态健康

---

## 8. 附录

### A. reward 分桶方案

```
bucket = floor(log(duration) / log(1.2))
```

| Bucket | 时长范围 | 视频类型 |
|--------|---------|--------|
| 0-5 | 5-9s | 超短视频 |
| 6-11 | 9-18s | 短视频 |
| 12-17 | 18-35s | 中短视频 |
| 18-23 | 35-68s | 中视频 |
| 24-29 | 68-130s | 中长视频 |

bucket 内分位数作为 reward，消除时长对完播率的天然影响。

### B. 实验配置

```python
MODEL_CONFIG = dict(
    batch_size=256,
    infer_size=96,
    lr=1e-4,
    max_len=50,
    topk_list=[5, 10, 20],
    beam_size=30,
    seed=2025,
    num_epochs=200,
    patience=20,        # early stopping
)
```

硬件：2×Tesla T4 (15GB each)，DataParallel 训练

### C. 关键代码路径

- 数据预处理：`data/process_vk.py`
- SID 生成：`rqvae/main.py` → `rqvae/generate_code.py`
- 实验 A：`model/experiment_drop.py`
- 实验 B：`model/experiment_pid_drop.py`
- 实验 C：`model/experiment_reward_drop.py`
- Dataset：`model/dataset.py`（`GenRecDataset`）
