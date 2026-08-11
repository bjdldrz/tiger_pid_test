"""
experiment_reward_drop.py
=========================
实验 C：基于 reward 的质量过滤，模拟 RSFT 效果。

Drop 方式：只保留 target_reward >= threshold 的训练样本（top 80% 完播率）
对比组  ：baseline（复用已有 baseline，不重复训练）

reward 字段：parquet 中的 'target_reward' 列（完播率分位数，0~1）
  - threshold=0.2 → 保留 top 80%（去掉完播率最低的 20%）

要求：train.parquet 必须包含 'target_reward' 列（VK-LSVD processed/ 已有）

使用方法：
  cd model
  python experiment_reward_drop.py [--epochs 50] [--device cuda] [--dataset VK-LSVD]
  python experiment_reward_drop.py --reward_thresholds 0.2 0.4 0.6
"""

import argparse
import logging
import os
import sys

import pandas as pd
import torch
import torch.optim as optim

from dataset import GenRecDataset, item2code
from dataloader import GenRecDataLoader
from main import (TIGER, train, evaluate, set_seed,
                  setup_device, wrap_model_multigpu,
                  save_baseline, load_baseline)

# threshold=0.2 means keep samples with reward >= 0.2 (top 80%)
REWARD_THRESHOLDS = [0.2]

DATASET_CONFIGS = {
    'Beauty': {
        'dataset_path': '../data/Beauty',
        'code_path': '../data/Beauty/Beauty_t5_rqvae.npy',
        'results_path': '../results/Beauty_reward_drop_experiment.csv',
        'baseline_path': '../results/Beauty_baseline.json',
        'log_path': '../results/Beauty_reward_drop_experiment.log',
        'save_dir': '../results/Beauty_reward_drop_ckpts',
    },
    'VK-LSVD': {
        'dataset_path': '../data/VK-LSVD/processed',
        'code_path': '../data/VK-LSVD/processed/VK_rqvae.npy',
        'results_path': '../results/VK_reward_drop_experiment.csv',
        'baseline_path': '../results/VK_baseline.json',
        'log_path': '../results/VK_reward_drop_experiment.log',
        'save_dir': '../results/VK_reward_drop_ckpts',
    },
}

MODEL_CONFIG = dict(
    num_layers=4, num_decoder_layers=4,
    d_model=128, d_ff=1024, num_heads=6, d_kv=64,
    dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0,
    feed_forward_proj='relu',
    batch_size=256, infer_size=96,
    lr=1e-4, max_len=20,
    topk_list=[5, 10, 20], beam_size=30,
    seed=2025,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VK-LSVD',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--reward_thresholds', type=float, nargs='+',
                        default=REWARD_THRESHOLDS,
                        help='Reward thresholds: keep samples with reward >= threshold. '
                             '0.2 = keep top 80%%, 0.5 = keep top 50%%.')
    return parser.parse_args()


def run_experiment(filter_kwargs, tag, config, device,
                   total_pids, code_to_item, test_pid_set,
                   validation_dataloader, test_dataloader):
    set_seed(config['seed'])
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/train.parquet',
        code_path=config['code_path'],
        mode='train', max_len=config['max_len'],
        seed=config['seed'], **filter_kwargs,
    )
    n_train = len(train_dataset)
    train_pid_set = train_dataset.get_train_pid_set()
    train_pid_coverage = len(train_pid_set) / total_pids
    train_coverage_over_test = len(train_pid_set & test_pid_set) / len(test_pid_set)

    logging.info(f"[{tag}] samples={n_train}, pid_cov={train_pid_coverage:.4f}, "
                 f"cov_over_test={train_coverage_over_test:.4f}")
    print(f"[{tag}] samples={n_train}, pid_coverage={train_pid_coverage:.4f}, "
          f"coverage_over_test={train_coverage_over_test:.4f}")

    train_dl = GenRecDataLoader(train_dataset, config['batch_size'], shuffle=True)
    model = wrap_model_multigpu(TIGER(config), device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_val_ndcg = 0.0
    best_test_recalls = best_test_ndcgs = None
    best_recalled_pids = set()

    for epoch in range(config['num_epochs']):
        loss = train(model, train_dl, optimizer, device)
        _, ndcgs, _ = evaluate(model, validation_dataloader,
                               config['topk_list'], config['beam_size'], device)
        if ndcgs['NDCG@20'] > best_val_ndcg:
            best_val_ndcg = ndcgs['NDCG@20']
            best_test_recalls, best_test_ndcgs, best_recalled_pids = evaluate(
                model, test_dataloader,
                config['topk_list'], config['beam_size'], device,
                code_to_item=code_to_item)
            os.makedirs(config['save_dir'], exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(config['save_dir'], f"{tag}.pth"))
        logging.info(f"  epoch {epoch+1} loss={loss:.4f} NDCG@20={ndcgs['NDCG@20']:.4f}")

    recalled_pid_coverage = len(best_recalled_pids) / total_pids
    result = {
        'experiment': tag,
        'n_train_samples': n_train,
        'train_pid_coverage': train_pid_coverage,
        'train_coverage_over_test': train_coverage_over_test,
        'recalled_pid_coverage': recalled_pid_coverage,
        'recalled_pid_count': len(best_recalled_pids),
        'total_pids': total_pids,
    }
    for k in config['topk_list']:
        result[f'Recall@{k}'] = best_test_recalls[f'Recall@{k}'] if best_test_recalls else 0.0
        result[f'NDCG@{k}'] = best_test_ndcgs[f'NDCG@{k}'] if best_test_ndcgs else 0.0
    print(f"[{tag}] recalled_pid_cov={recalled_pid_coverage:.4f} "
          f"Recall@10={result['Recall@10']:.4f}")
    return result


def main():
    args = parse_args()
    dc = DATASET_CONFIGS[args.dataset]
    config = {**MODEL_CONFIG, **dc,
              'num_epochs': args.epochs, 'device': args.device}

    os.makedirs(os.path.dirname(dc['log_path']), exist_ok=True)
    logging.basicConfig(filename=dc['log_path'], level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    device = setup_device(args.device)
    logging.info(f"device={device}, dataset={args.dataset}")

    item_to_code, code_to_item = item2code(config['code_path'])
    total_pids = len(item_to_code)

    valid_ds = GenRecDataset(dc['dataset_path'] + '/valid.parquet',
                             config['code_path'], 'evaluation', config['max_len'])
    test_ds  = GenRecDataset(dc['dataset_path'] + '/test.parquet',
                             config['code_path'], 'evaluation', config['max_len'])
    valid_dl = GenRecDataLoader(valid_ds, int(config['infer_size']), shuffle=False)
    test_dl  = GenRecDataLoader(test_ds,  int(config['infer_size']), shuffle=False)
    test_pid_set = test_ds.get_train_pid_set()

    all_results = []

    # ---- Baseline: load or train ----
    baseline = load_baseline(dc['baseline_path'])
    if baseline is None:
        logging.info("No baseline found, training baseline...")
        baseline = run_experiment({}, 'baseline', config, device,
                                  total_pids, code_to_item, test_pid_set,
                                  valid_dl, test_dl)
        save_baseline(baseline, dc['baseline_path'])
    all_results.append(baseline)

    # ---- Reward threshold experiments ----
    for threshold in args.reward_thresholds:
        pct_keep = int((1 - threshold) * 100)
        tag = f'reward_top{pct_keep:03d}pct'
        result = run_experiment(
            {'reward_threshold': threshold}, tag, config, device,
            total_pids, code_to_item, test_pid_set, valid_dl, test_dl)
        all_results.append(result)
        df = pd.DataFrame(all_results)
        df.to_csv(config['results_path'], index=False)
        print(df.to_string(index=False))

    pd.DataFrame(all_results).to_csv(config['results_path'], index=False)
    logging.info("=== experiment_reward_drop complete ===")


if __name__ == '__main__':
    main()
