# TIGER 实验文档：RSFT 质量过滤对 PID 覆盖率与召回率的影响

## 一、研究背景与问题

### 研究问题

RSFT（Reward-based Sequential Fine-Tuning）通过对训练样本按 reward 进行质量过滤，保留高质量样本。本实验旨在验证以下因果链：

```
质量过滤（RSFT）
    → 部分低质量 item 的训练样本被删除
    → 训练集 PID 覆盖率下降
    → 模型对低覆盖 PID 的生成能力下降
    → 测试集 Recall@K 降低、召回 PID 覆盖率降低
```

### 核心假设

> 训练集 PID 覆盖率降低会导致 TIGER 的 Recall@K 和召回 PID 覆盖率下降。

### 实验策略

使用三组平行实验，分离不同的影响因素：

| 实验组 | Drop 方式 | 控制的变量 | 目的 |
|--------|---------|---------|------|
| A. 随机样本 Drop | 随机丢弃 20% 训练样本 | 样本量 | 排除"样本量减少"的干扰 |
| B. 随机 PID Drop | 随机排除 20% PID 的所有样本 | PID 覆盖率（中性） | 直接验证 PID 缺失的纯粹影响 |
| C. Reward 质量过滤 | 只保留 top 80% 完播率样本 | 样本质量 | 模拟真实 RSFT 效果 |

每组均与 **baseline（无 drop）** 对比，baseline 结果持久化复用，不重复训练。

---

## 二、数据集

### 2.1 Beauty（Amazon Beauty Reviews）

| 项目 | 数值 |
|------|------|
| 数据集 | Amazon Beauty Reviews |
| item 数 | 12,101 |
| 训练样本数 | ~198,000 |
| SID 生成方法 | sentence-t5-base embedding (768 维) → RQ-VAE |
| reward 字段 | ❌ 无（只能运行实验 A、B） |

**适用实验**：A（随机样本 Drop）、B（随机 PID Drop）

### 2.2 VK-LSVD（VK 短视频推荐数据集）

| 项目 | 数值 |
|------|------|
| 数据集 | VK-LSVD week_00（取 1% 用户采样 + 5-core 过滤） |
| item 数 | 35,472 |
| 训练样本数 | ~519,000 |
| SID 生成方法 | 预训练行为 embedding (64 维) → RQ-VAE |
| reward 字段 | ✅ `target_reward`（完播率分位数：timespent / duration） |

**适用实验**：A、B、C（完整三组对比）

---

## 三、数据预处理

### 3.1 Beauty 数据（已完成）

Beauty 数据已预处理完毕，文件位于 `data/Beauty/`。

```
data/Beauty/
├── train.parquet          # 训练集
├── valid.parquet          # 验证集
├── test.parquet           # 测试集
├── Beauty_t5_rqvae.npy   # SID 编码（item_idx → [c0,c1,c2,c3]）
└── item_mapping.npy       # item 映射
```

### 3.2 VK-LSVD 数据预处理

**Step 1：生成 train/valid/test parquet**

```bash
cd data
python process_vk.py
```

`process_vk.py` 执行流程：
1. 读取 `VK-LSVD/week_00.parquet`，随机采样 1% 用户（seed=42）
2. 与 `items_metadata.parquet` 合并，计算完播率 `completion_rate = timespent / duration`
3. 计算全局分位数 reward：`target_reward = completion_rate.rank(pct=True)`
4. 迭代 5-core 过滤（user 和 item 各出现 ≥5 次）
5. 按用户行序排列（行索引即时间序）
6. Leave-one-out 划分：test=最后一条，valid=倒数第二条，train=滑动窗口其余
7. 保存到 `data/VK-LSVD/processed/`，含 `target_reward` 列

输出文件：

```
data/VK-LSVD/processed/
├── train.parquet          # 519,469 训练样本，含 target_reward 列
├── valid.parquet          # 6,392 验证样本
├── test.parquet           # 6,392 测试样本
├── item_emb.npy           # 35,472 × 64 item embedding
├── item_mapping.npy       # item 映射
└── user_mapping.npy       # user 映射
```

**Step 2：训练 RQ-VAE，生成 SID**

```bash
cd rqvae
python main.py \
  --data_path ../data/VK-LSVD/processed/item_emb.npy \
  --ckpt_dir ./ckpt/VK-LSVD \
  --layers 64 64 32 \
  --device cuda
```

**Step 3：使用训练好的 RQ-VAE 生成 SID 编码**

```bash
cd rqvae
python generate_code.py --dataset VK-LSVD
# 输出：../data/VK-LSVD/processed/VK_rqvae.npy
```

---

## 四、指标体系

每组实验记录以下指标：

| 指标 | 说明 |
|------|------|
| `n_train_samples` | 过滤后训练样本数 |
| `train_pid_coverage` | 训练集 target 唯一 PID 数 / 全量 item 数 |
| `train_coverage_over_test` | 训练集 PID ∩ 测试集 PID / 测试集唯一 PID 数 |
| `recalled_pid_coverage` | beam search 召回的唯一 PID 数 / 全量 item 数 |
| `Recall@5/10/20` | 标准召回率 |
| `NDCG@5/10/20` | 标准归一化折损累计增益 |

---

## 五、实验 A：随机样本 Drop

### 实验设计

随机丢弃 20% 训练样本（`(history, target)` 对），其余不变。

- **目的**：控制"样本量减少"这一变量，为实验 C 提供对照
- **预期**：Recall 轻微下降；PID 覆盖率基本不变（随机 drop 不会系统性删除某些 PID）

### 运行命令

```bash
# Beauty 数据集
cd model
python experiment_drop.py \
  --dataset Beauty \
  --epochs 50 \
  --device cuda \
  --drop_ratios 0.2

# VK-LSVD 数据集（需先完成 RQ-VAE 训练）
cd model
python experiment_drop.py \
  --dataset VK-LSVD \
  --epochs 50 \
  --device cuda \
  --drop_ratios 0.2
```

### 结果文件

- `results/Beauty_drop_experiment.csv`
- `results/VK_drop_experiment.csv`
- Baseline 缓存：`results/Beauty_baseline.json` / `results/VK_baseline.json`

---

## 六、实验 B：随机 PID Drop

### 实验设计

随机选择 20% 的 target PID，删除其所有训练样本（保留 80% PID）。

- **目的**：精确控制 PID 覆盖率，验证 PID 缺失本身对模型的影响
- **预期**：train_pid_coverage 精确下降 20%；Recall 单调下降

### 运行命令

```bash
# Beauty 数据集
cd model
python experiment_pid_drop.py \
  --dataset Beauty \
  --epochs 50 \
  --device cuda \
  --pid_keep_ratios 0.8

# VK-LSVD 数据集
cd model
python experiment_pid_drop.py \
  --dataset VK-LSVD \
  --epochs 50 \
  --device cuda \
  --pid_keep_ratios 0.8
```

### 结果文件

- `results/Beauty_pid_drop_experiment.csv`
- `results/VK_pid_drop_experiment.csv`

---

## 七、实验 C：Reward 质量过滤（模拟 RSFT）

### 实验设计

只保留 `target_reward >= 0.2` 的训练样本（完播率分位数 ≥ 0.2，即 top 80% 质量样本）。

- **目的**：模拟 RSFT 的样本质量过滤效果
- **reward 定义**：`target_reward = (timespent / duration).rank(pct=True)`，越高表示该次交互完播率越高
- **预期**：低 reward item（很少被完整观看的视频）会被过滤，导致这些 item 的 PID 从训练集消失，进而影响召回

> **注意**：此实验仅支持 VK-LSVD 数据集（Beauty 无 reward 字段）。

### 运行命令

```bash
# VK-LSVD 数据集（唯一可用数据集）
cd model
python experiment_reward_drop.py \
  --dataset VK-LSVD \
  --epochs 50 \
  --device cuda \
  --reward_thresholds 0.2
```

### 结果文件

- `results/VK_reward_drop_experiment.csv`

---

## 八、Baseline 复用机制

三个实验脚本共享同一份 baseline（无 drop 的全量训练结果）：

```
第一个运行的实验脚本：
    → 检查 results/{dataset}_baseline.json 是否存在
    → 不存在 → 训练 baseline 并保存
    
后续实验脚本：
    → 检查 results/{dataset}_baseline.json 是否存在
    → 存在 → 直接加载，跳过训练
```

**效果**：三个实验总训练次数 = baseline × 1 + 各实验 × 1（无重复浪费）。

---

## 九、多 GPU 支持

代码自动检测可用 GPU 数量：

- **单 GPU**：正常训练
- **多 GPU**：自动启用 `nn.DataParallel`，训练 batch 均分到各 GPU
- **推理（beam search）**：自动使用 `model.module.generate()`，避免 DataParallel 兼容问题

无需额外配置，`--device cuda` 即可利用所有可用 GPU。

---

## 十、后台运行（服务端）

```bash
# 在服务端后台并行运行三个实验
cd ~/tiger_pid_test/model

# 实验 A：随机样本 Drop
nohup python experiment_drop.py \
  --dataset VK-LSVD --epochs 50 --device cuda \
  > ../results/vk_drop.log 2>&1 &

# 实验 B：随机 PID Drop
nohup python experiment_pid_drop.py \
  --dataset VK-LSVD --epochs 50 --device cuda \
  > ../results/vk_pid_drop.log 2>&1 &

# 实验 C：Reward 质量过滤
nohup python experiment_reward_drop.py \
  --dataset VK-LSVD --epochs 50 --device cuda \
  > ../results/vk_reward_drop.log 2>&1 &

# 查看进度
tail -f ../results/vk_reward_drop.log
```

---

## 十一、结果分析框架

实验完成后，对比以下关系：

### 11.1 主要对比

| 对比 | 关注指标 | 结论方向 |
|------|---------|---------|
| 实验 C vs Baseline | Recall@K、PID 覆盖率 | RSFT 过滤是否损伤召回 |
| 实验 C vs 实验 A | Recall@K 降幅 | 是否超出"样本量减少"的解释 |
| 实验 C vs 实验 B | PID 覆盖率降幅、Recall@K 降幅 | 质量过滤 vs 随机 PID 删除的差异 |

### 11.2 判断假设是否成立

| 场景 | 结论 |
|------|------|
| C 的 Recall 下降 ≈ A 的 Recall 下降 | 影响主要来自样本量减少，非质量过滤 |
| C 的 Recall 下降 > A，且 ≈ B | 影响主要来自 PID 缺失，RSFT 导致 PID 覆盖率下降 |
| C 的 Recall 下降 > A，且 > B | 质量过滤引入额外分布偏移，不仅仅是 PID 缺失问题 |
| C 的 Recall 反而提升 | RSFT 质量过滤有助于推荐，噪声样本被有效去除 |

---

## 十二、文件目录结构

```
TIGER-main/
├── data/
│   ├── Beauty/
│   │   ├── train.parquet
│   │   ├── valid.parquet
│   │   ├── test.parquet
│   │   └── Beauty_t5_rqvae.npy
│   ├── VK-LSVD/
│   │   ├── week_00.parquet          # 原始数据
│   │   ├── items_metadata.parquet   # item 元数据（含 duration）
│   │   ├── item_embeddings.npz      # item embedding (1962万×64)
│   │   └── processed/               # 预处理输出
│   │       ├── train.parquet        # 含 target_reward 列
│   │       ├── valid.parquet
│   │       ├── test.parquet
│   │       ├── item_emb.npy         # 35472×64
│   │       └── VK_rqvae.npy         # SID（RQ-VAE 生成后）
│   └── process_vk.py                # VK-LSVD 预处理脚本
├── rqvae/
│   ├── main.py                      # RQ-VAE 训练
│   └── generate_code.py             # SID 生成（支持 --dataset VK-LSVD）
├── model/
│   ├── main.py                      # TIGER 模型定义、train/evaluate
│   ├── dataset.py                   # GenRecDataset（含 reward_threshold）
│   ├── experiment_drop.py           # 实验 A
│   ├── experiment_pid_drop.py       # 实验 B
│   └── experiment_reward_drop.py    # 实验 C
└── results/
    ├── Beauty_baseline.json         # Beauty baseline（共享）
    ├── VK_baseline.json             # VK baseline（共享）
    ├── Beauty_drop_experiment.csv
    ├── VK_drop_experiment.csv
    ├── VK_pid_drop_experiment.csv
    └── VK_reward_drop_experiment.csv
```
