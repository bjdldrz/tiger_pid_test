# PID 覆盖率对 TIGER 召回率影响的实验文档

## 1. 实验背景

TIGER 是一个基于 T5 编解码器的生成式推荐模型，使用 Semantic ID（SID）表示 item。SID 由 RQ-VAE 从 item 语义 embedding 生成，每个 item 对应一个 4 维离散编码。

**核心问题**：训练样本中的 PID（item ID）覆盖率降低，是否会导致模型的 Recall@K 和召回 PID 覆盖率下降？

本文档包含两个实验：

| 实验 | 脚本 | 控制方式 | 结果文件 |
|------|------|---------|---------|
| 随机 Drop 样本 | `experiment_drop.py` | 随机丢弃训练样本对 | `results/drop_experiment.csv` |
| PID 维度 Drop | `experiment_pid_drop.py` | 直接排除部分 PID 的所有样本 | `results/pid_drop_experiment.csv` |

---

## 2. 数据集

- **数据集**：Amazon Beauty（5-core）
- **数据路径**：`data/Beauty/`
- **交互划分**：Leave-one-out，测试集取最后一次交互，验证集取倒数第二次，其余为训练集
- **item 总数**：12101

### 数据文件说明

| 文件 | 说明 |
|------|------|
| `data/Beauty/train.parquet` | 训练集（每行：user_id, history list, target item） |
| `data/Beauty/valid.parquet` | 验证集（leave-one-out 倒数第二条） |
| `data/Beauty/test.parquet` | 测试集（leave-one-out 最后一条） |
| `data/Beauty/item_emb.parquet` | item 语义 embedding（sentence-t5-base，768 维） |
| `data/Beauty/item_mapping.npy` | item 原始 ID → 整数索引映射 |
| `data/Beauty/user_mapping.npy` | user 原始 ID → 整数索引映射 |
| `data/Beauty/Beauty_t5_rqvae.npy` | RQ-VAE 生成的 SID 编码（12101 × 4） |

---

## 3. 数据处理流程

### 3.1 原始数据预处理

运行 `data/process.ipynb` 完成以下步骤：

1. 读取 `reviews_Beauty_5.json.gz`（用户交互记录）
2. 按时间排序，leave-one-out 划分训练 / 验证 / 测试集
3. 用 `sentence-transformers/sentence-t5-base` 生成每个 item 的 768 维语义 embedding
4. 保存 `train/valid/test.parquet` 和 `item_emb.parquet`

```bash
cd data
jupyter nbconvert --to notebook --execute process.ipynb
```

### 3.2 生成 SID 编码

运行 `rqvae/generate_code.py`，使用预训练 RQ-VAE checkpoint 将 item embedding 量化为 4 维 SID：

```bash
cd rqvae
python generate_code.py
```

**输出**：`data/Beauty/Beauty_t5_rqvae.npy`，shape `(12101, 4)`，每行为一个 item 的 SID 编码。

SID 偏移计算规则：
```
offsets = [code[i] + i * 256 + 1 for i in range(4)]
```
确保 4 个维度的 token 不重叠，vocab_size = 256 × 4 + 1 = 1025。

---

## 4. 训练样本构造

`model/dataset.py` 中 `process_data` 函数的训练集构造方式：

**滑动窗口展开**：对每条用户序列 `[i1, i2, ..., in]`，生成 n-1 条训练样本：

```
([i1], i2)
([i1, i2], i3)
...
([i1, ..., i_{n-1}], i_n)
```

每条样本的 `history` 左填充到 `max_len=20`，`target` 转换为 SID token 序列。

---

## 5. 实验设计

### 5.1 随机 Drop 样本实验（experiment_drop.py）

**目的**：验证样本数量减少本身是否影响性能（控制变量）。

**方法**：在滑动窗口展开后，按 `drop_ratio` 随机丢弃训练样本对，PID 覆盖率间接下降。

```python
# dataset.py 中的实现
if self.drop_ratio > 0.0:
    rng = random.Random(self.seed)
    n_keep = int(len(processed_data) * (1.0 - self.drop_ratio))
    processed_data = rng.sample(processed_data, n_keep)
```

**实验组**：`drop_ratio` = [0.0, 0.2, 0.4, 0.6, 0.8]

### 5.2 PID 维度 Drop 实验（experiment_pid_drop.py）

**目的**：直接控制训练集 PID 覆盖率，验证 PID 缺失是否是性能下降的因果变量。

**方法**：随机选取 `pid_keep_ratio` 比例的 PID 保留，将所有 target 为被排除 PID 的训练样本全部移除，PID 覆盖率**精确等于** `pid_keep_ratio`。

```python
# dataset.py 中的实现
if self.pid_keep_ratio < 1.0:
    rng_pid = random.Random(self.seed)
    all_pids = sorted(set(item['target'] for item in processed_data))
    n_keep_pids = int(len(all_pids) * self.pid_keep_ratio)
    kept_pids = set(rng_pid.sample(all_pids, n_keep_pids))
    processed_data = [item for item in processed_data if item['target'] in kept_pids]
```

**实验组**：`pid_keep_ratio` = [1.0, 0.8, 0.6, 0.4, 0.2]

### 5.3 两种 Drop 方式的对比

| 维度 | 随机 Drop 样本 | PID 维度 Drop |
|------|--------------|--------------|
| Drop 单位 | `(history, target)` 样本对 | target PID |
| PID 覆盖率控制 | 间接、随机涌现 | **精确等于 pid_keep_ratio** |
| 样本数量变化 | 与 drop_ratio 成比例下降 | 依赖 PID 频率分布，低频 PID 样本少，下降幅度可能不同 |
| 科学意义 | 验证"样本少"是否影响性能 | **直接验证"PID 缺失"是否影响性能** |

---

## 6. 模型与训练配置

| 参数 | 值 |
|------|-----|
| 模型 | T5（编解码器结构） |
| num_layers / num_decoder_layers | 4 |
| d_model | 128 |
| d_ff | 1024 |
| num_heads | 6 |
| d_kv | 64 |
| dropout_rate | 0.1 |
| vocab_size | 1025 |
| batch_size | 256 |
| lr | 1e-4 |
| max_len | 20 |
| beam_size | 20 |
| epochs | 50 |
| 随机种子 | 2025 |

验证集选择标准：每个 epoch 结束后在验证集评估 NDCG@20，保存最佳 checkpoint，最终在测试集上报告指标。

---

## 7. 评估指标

| 指标 | 说明 |
|------|------|
| `Recall@K` (K=5,10,20) | beam search Top-K 中命中 ground truth 的比例 |
| `NDCG@K` (K=5,10,20) | 带位置折扣的归一化排序增益 |
| `train_pid_coverage` | 训练集 target 唯一 PID 数 / 全量 12101 item |
| `train_coverage_over_test` | 训练集与测试集共有的 target PID 数 / 测试集唯一 target PID 数 |
| `recalled_pid_coverage` | beam search 生成的有效 PID 数 / 全量 12101 item |

---

## 8. 运行指令

### 环境准备

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pyarrow fastparquet tqdm transformers sentence-transformers scikit-learn
```

### 生成 SID（如尚未生成）

```bash
cd rqvae
python generate_code.py
# 输出：../data/Beauty/Beauty_t5_rqvae.npy
```

### 运行随机 Drop 实验

```bash
cd model
nohup python experiment_drop.py --epochs 50 --device cuda \
  > ../results/experiment_drop.log 2>&1 &
```

### 运行 PID 维度 Drop 实验

```bash
cd model
nohup python experiment_pid_drop.py --epochs 50 --device cuda \
  > ../results/experiment_pid_drop.log 2>&1 &
```

### 监控运行状态

```bash
# 查看进程
ps aux | grep experiment

# 实时查看日志
tail -f results/experiment_drop.log
tail -f results/experiment_pid_drop.log

# 查看 GPU 占用
watch -n 2 nvidia-smi

# 查看当前结果
cat results/drop_experiment.csv
cat results/pid_drop_experiment.csv
```

---

## 9. 结果分析方向

实验完成后，通过以下维度验证假设：

1. **PID 覆盖率 vs Recall@K**：随 `pid_keep_ratio` 下降，Recall@10 是否单调下降？
2. **PID 覆盖率 vs 召回 PID 覆盖率**：模型能否"生成"未见过的 PID 对应的 SID？
3. **随机 drop vs PID drop 对比**：相同覆盖率水平下，两种 drop 方式的 Recall@K 差异揭示了什么？
4. **train_coverage_over_test vs Recall@K**：训练集对测试集的覆盖率是否比全量覆盖率更能预测性能？

---

## 10. 文件结构

```
TIGER-main/
├── data/
│   └── Beauty/
│       ├── train.parquet
│       ├── valid.parquet
│       ├── test.parquet
│       ├── item_emb.parquet
│       └── Beauty_t5_rqvae.npy
├── model/
│   ├── dataset.py          # 数据集类，支持 drop_ratio 和 pid_keep_ratio
│   ├── dataloader.py       # DataLoader 封装
│   ├── main.py             # TIGER 模型、训练和评估函数
│   ├── experiment_drop.py  # 随机 Drop 实验脚本
│   └── experiment_pid_drop.py  # PID 维度 Drop 实验脚本
├── rqvae/
│   ├── generate_code.py    # 用 RQ-VAE 生成 SID
│   ├── main.py             # RQ-VAE 训练入口
│   └── models/             # RQ-VAE 模型结构
├── results/
│   ├── drop_experiment.csv
│   └── pid_drop_experiment.csv
└── EXPERIMENT.md           # 本文档
```
