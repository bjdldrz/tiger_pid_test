"""
process_vk.py
=============
VK-LSVD 数据预处理脚本

流程：
  1. 读取 week_00.parquet，取前 1% 数据（约 1 天）
  2. 与 items_metadata.parquet 合并，计算完播率 reward = timespent / duration
  3. 迭代 5-core 过滤（user 和 item 各出现 >= 5 次）
  4. 按用户行索引排序（行序即时间序）
  5. Leave-one-out 划分：test=最后一条，valid=倒数第二条，train=其余
  6. 保存 train/valid/test.parquet（含 reward 列）
  7. 保存 item_mapping.npy、user_mapping.npy
  8. 从 item_embeddings.npz 中提取子集 embedding，保存为 item_emb.parquet

输出目录：data/VK-LSVD/processed/

使用方法：
  cd data
  python process_vk.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DATA_DIR = Path('./VK-LSVD')
OUT_DIR  = DATA_DIR / 'processed'
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATIO = 0.01   # 随机采样 1% 的用户
CORE         = 5      # 5-core 过滤
MAX_LEN      = 20     # 序列最大长度（与 TIGER 一致）

# ---------------------------------------------------------------------------
# Step 1: 读取并采样
# ---------------------------------------------------------------------------
print('[1/8] Loading interaction log...')
df = pd.read_parquet(DATA_DIR / 'week_00.parquet')
all_users = df.user_id.unique()
rng = np.random.default_rng(seed=42)
sampled_users = rng.choice(all_users, size=int(len(all_users) * SAMPLE_RATIO), replace=False)
df = df[df.user_id.isin(sampled_users)].reset_index(drop=True)
print(f'  Sampled {len(sampled_users):,} users ({SAMPLE_RATIO*100:.0f}%), '
      f'{len(df):,} interactions, {df.item_id.nunique():,} items')

# ---------------------------------------------------------------------------
# Step 2: 合并 metadata，计算完播率 reward
# ---------------------------------------------------------------------------
print('[2/8] Merging metadata and computing reward...')
meta = pd.read_parquet(DATA_DIR / 'items_metadata.parquet')[['item_id', 'duration']]
df = df.merge(meta, on='item_id', how='left')

# ---------- reward 计算：按 duration 分桶后桶内分位数 ----------
# 分桶：bucket = floor(log(duration) / log(1.2))
# 同一时长量级的视频放在同一桶内，消除视频时长差异的影响
BUCKET_BASE = 1.2
import math
df['duration_bucket'] = (
    np.log(df['duration'].clip(lower=1)) / math.log(BUCKET_BASE)
).apply(math.floor)

# 桶内分位数：timespent 在同桶所有交互中的百分位，作为 reward (0~1)
df['reward'] = df.groupby('duration_bucket')['timespent'].rank(pct=True)

# 打印分桶信息
n_buckets = df['duration_bucket'].nunique()
bucket_counts = df['duration_bucket'].value_counts().sort_index()
print(f'  duration buckets: {n_buckets} buckets (base={BUCKET_BASE})')
print(f'  bucket size range: min={bucket_counts.min()}, max={bucket_counts.max()}, '
      f'median={bucket_counts.median():.0f}')
print(f'  reward (bucket-wise pct): mean={df.reward.mean():.3f}, '
      f'median={df.reward.median():.3f}, '
      f'75th={df.reward.quantile(0.75):.3f}')

# ---------------------------------------------------------------------------
# Step 3: 5-core 迭代过滤
# ---------------------------------------------------------------------------
print(f'[3/8] Applying {CORE}-core filtering...')
for i in range(20):
    before = len(df)
    item_cnt = df.item_id.value_counts()
    df = df[df.item_id.isin(item_cnt[item_cnt >= CORE].index)]
    user_cnt = df.user_id.value_counts()
    df = df[df.user_id.isin(user_cnt[user_cnt >= CORE].index)]
    after = len(df)
    print(f'  Iter {i+1}: {after:,} interactions, '
          f'{df.user_id.nunique():,} users, {df.item_id.nunique():,} items')
    if before == after:
        break

# ---------------------------------------------------------------------------
# Step 4: 按用户行索引排序（行索引即全局时间序）
# ---------------------------------------------------------------------------
print('[4/8] Sorting by interaction order...')
df = df.sort_values(['user_id', df.index.name or 'index'] if 'index' in df.columns
                    else ['user_id']).reset_index(drop=False)
# 用原始 index 列（行号）作为时间序代理
if 'index' in df.columns:
    df = df.sort_values(['user_id', 'index']).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Step 5: 重建 user/item 映射（连续整数索引，从 1 开始）
# ---------------------------------------------------------------------------
print('[5/8] Building user/item mappings...')
unique_users = sorted(df.user_id.unique())
unique_items = sorted(df.item_id.unique())

user2idx = {u: i+1 for i, u in enumerate(unique_users)}
item2idx = {it: i+1 for i, it in enumerate(unique_items)}

df['user_idx'] = df['user_id'].map(user2idx)
df['item_idx'] = df['item_id'].map(item2idx)

n_users = len(unique_users)
n_items = len(unique_items)
print(f'  {n_users:,} users, {n_items:,} items after remapping')

# 保存映射
np.save(OUT_DIR / 'user_mapping.npy', np.array(unique_users))
np.save(OUT_DIR / 'item_mapping.npy', np.array(unique_items))
print(f'  Saved user_mapping.npy ({n_users}) and item_mapping.npy ({n_items})')

# ---------------------------------------------------------------------------
# Step 6: Leave-one-out 划分
# ---------------------------------------------------------------------------
print('[6/8] Building train/valid/test splits (leave-one-out)...')

# 按 user + 行顺序分组
groups = df.groupby('user_idx', sort=False)

train_rows, valid_rows, test_rows = [], [], []

for user_idx, group in groups:
    group = group.reset_index(drop=True)
    seq = group['item_idx'].tolist()
    rewards = group['reward'].tolist()

    if len(seq) < 3:
        continue

    # history 列表
    history_all = seq[:-1]   # 去掉最后一个（test target）

    # Test: 最后一个 item 为 target，前面所有为 history
    test_rows.append({
        'user_id': user_idx,
        'history': history_all,
        'target': seq[-1],
        'target_reward': rewards[-1],
    })

    # Valid: 倒数第二个为 target
    valid_rows.append({
        'user_id': user_idx,
        'history': seq[:-2],
        'target': seq[-2],
        'target_reward': rewards[-2],
    })

    # Train: 滑动窗口，每条记录 (history[0:i], item[i], reward[i])
    for i in range(1, len(seq) - 2):
        train_rows.append({
            'user_id': user_idx,
            'history': seq[:i],
            'target': seq[i],
            'target_reward': rewards[i],
        })

train_df = pd.DataFrame(train_rows)
valid_df = pd.DataFrame(valid_rows)
test_df  = pd.DataFrame(test_rows)

# 截断/填充 history 到 max_len（只截断，填充在 dataset.py 中做）
def truncate(seq, max_len):
    return seq[-max_len:] if len(seq) > max_len else seq

train_df['history'] = train_df['history'].apply(lambda x: truncate(x, MAX_LEN))
valid_df['history'] = valid_df['history'].apply(lambda x: truncate(x, MAX_LEN))
test_df['history']  = test_df['history'].apply(lambda x: truncate(x, MAX_LEN))

print(f'  train: {len(train_df):,} samples')
print(f'  valid: {len(valid_df):,} samples')
print(f'  test:  {len(test_df):,} samples')
print(f'  target_reward stats (train):')
print(f'    mean={train_df.target_reward.mean():.3f}, '
      f'25th={train_df.target_reward.quantile(0.25):.3f}, '
      f'75th={train_df.target_reward.quantile(0.75):.3f}')

# ---------------------------------------------------------------------------
# Step 7: 保存 parquet
# ---------------------------------------------------------------------------
print('[7/8] Saving parquet files...')
train_df.to_parquet(OUT_DIR / 'train.parquet', index=False, engine='fastparquet')
valid_df.to_parquet(OUT_DIR / 'valid.parquet', index=False, engine='fastparquet')
test_df.to_parquet(OUT_DIR / 'test.parquet',   index=False, engine='fastparquet')

# test_good: 去掉 bottom 20% reward 的测试样本（target_reward < 20th percentile）
# 用于评估模型对高质量互动的召回情况
test_reward_20pct = test_df['target_reward'].quantile(0.2)
test_good_df = test_df[test_df['target_reward'] >= test_reward_20pct].copy()
test_good_df.to_parquet(OUT_DIR / 'test_good.parquet', index=False, engine='fastparquet')

print(f'  test (full):    {len(test_df):,} samples')
print(f'  test_good (top80%): {len(test_good_df):,} samples  '
      f'(reward >= {test_reward_20pct:.3f})')
print(f'  Saved to {OUT_DIR}')

# ---------------------------------------------------------------------------
# Step 8: 提取 item embedding 子集，保存为 item_emb.parquet
# ---------------------------------------------------------------------------
print('[8/8] Extracting item embeddings for filtered items...')
emb_data = np.load(DATA_DIR / 'item_embeddings.npz')
emb_item_ids  = emb_data['item_id']   # shape: (19627601,)
emb_vectors   = emb_data['embedding'] # shape: (19627601, 64), float16

# 建立 item_id → embedding 的查找表
print('  Building embedding lookup...')
emb_id2idx = {int(v): i for i, v in enumerate(emb_item_ids)}

# 按照 item_mapping 的顺序提取 embedding（item_mapping 中索引 i 对应 item2idx=i+1）
embeddings_out = []
missing = 0
for orig_item_id in unique_items:
    if int(orig_item_id) in emb_id2idx:
        idx = emb_id2idx[int(orig_item_id)]
        embeddings_out.append(emb_vectors[idx].astype(np.float32))
    else:
        # 用零向量填充（罕见情况）
        embeddings_out.append(np.zeros(64, dtype=np.float32))
        missing += 1

embeddings_out = np.array(embeddings_out)  # shape: (n_items, 64)
print(f'  Embeddings shape: {embeddings_out.shape}, missing: {missing}')

# 保存为 npy（shape: n_items x 64）和一个 mapping csv
# fastparquet 无法保存 numpy array 列，改用 npy
np.save(OUT_DIR / 'item_emb.npy', embeddings_out)
# 额外保存 item_idx → item_id_orig 的映射 csv（调试用）
pd.DataFrame({
    'item_idx': list(range(1, n_items + 1)),
    'item_id_orig': unique_items,
}).to_csv(OUT_DIR / 'item_idx_map.csv', index=False)
print(f'  Saved item_emb.npy (shape={embeddings_out.shape})')

# ---------------------------------------------------------------------------
# 完成
# ---------------------------------------------------------------------------
print()
print('=== Preprocessing complete ===')
print(f'Output dir: {OUT_DIR}')
print(f'  user_mapping.npy : {n_users} users')
print(f'  item_mapping.npy : {n_items} items')
print(f'  train.parquet    : {len(train_df):,} samples')
print(f'  valid.parquet    : {len(valid_df):,} samples')
print(f'  test.parquet     : {len(test_df):,} samples')
print(f'  item_emb.npy     : {n_items} items x 64 dim')
print()
print('Next step: run rqvae to generate SID codes')
print('  cd ../rqvae && python generate_code.py --dataset VK-LSVD')
