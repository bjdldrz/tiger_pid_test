"""
experiment_pid_drop.py
======================
验证实验：直接控制 PID 覆盖率对 TIGER 召回率的影响。

与 experiment_drop.py（随机 drop 样本）的区别：
  本实验在 PID 维度精确控制覆盖率——随机选取 pid_keep_ratio 比例的 PID 保留，
  将所有 target 为被排除 PID 的训练样本全部移除，从而精确控制训练集 PID 覆盖率。

实验统计指标（与 experiment_drop.py 完全一致）：
    - 训练集 target PID 覆盖率（精确等于 pid_keep_ratio）
    - 模型 beam search 能生成的 PID 覆盖率（召回 PID 覆盖率）
    - Recall@5 / Recall@10 / Recall@20
    - NDCG@5 / NDCG@10 / NDCG@20

结果保存至：../results/pid_drop_experiment.csv

使用方法：
  cd model
  python experiment_pid_drop.py [--epochs 50] [--device cuda]
  python experiment_pid_drop.py --pid_keep_ratios 1.0 0.8 0.6 0.4 0.2
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
from main import TIGER, train, evaluate, set_seed

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
PID_KEEP_RATIOS = [1.0, 0.8, 0.6, 0.4, 0.2]

DEFAULT_CONFIG = dict(
    # Model
    num_layers=4,
    num_decoder_layers=4,
    d_model=128,
    d_ff=1024,
    num_heads=6,
    d_kv=64,
    dropout_rate=0.1,
    vocab_size=1025,
    pad_token_id=0,
    eos_token_id=0,
    feed_forward_proj='relu',
    # Training
    batch_size=256,
    infer_size=96,
    lr=1e-4,
    max_len=20,
    # Eval
    topk_list=[5, 10, 20],
    beam_size=30,
    # Paths
    dataset_path='../data/Beauty',
    code_path='../data/Beauty/Beauty_t5_rqvae.npy',
    results_path='../results/pid_drop_experiment.csv',
    save_dir='../results/pid_drop_ckpts',
    log_path='../results/pid_drop_experiment.log',
    # Reproducibility
    seed=2025,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PID-level drop experiment for TIGER")
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per run (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda, mps, or cpu')
    parser.add_argument('--pid_keep_ratios', type=float, nargs='+', default=PID_KEEP_RATIOS,
                        help='List of PID keep ratios to test (e.g. 1.0 0.8 0.6 0.4 0.2)')
    return parser.parse_args()


def run_single_experiment(pid_keep_ratio, config, device, total_pids, code_to_item,
                           validation_dataloader, test_dataloader, test_pid_set=None):
    """Train one model with the given pid_keep_ratio and return metrics."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting experiment: pid_keep_ratio={pid_keep_ratio:.2f}")
    logging.info(f"{'='*60}")

    set_seed(config['seed'])

    # Build training dataset with PID-level drop
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len'],
        pid_keep_ratio=pid_keep_ratio,
        seed=config['seed'],
    )

    train_pid_coverage = train_dataset.get_train_pid_coverage(total_pids)
    n_train = len(train_dataset)
    # Coverage over test set: how many test-target PIDs appear in training
    train_pid_set = train_dataset.get_train_pid_set()
    if test_pid_set:
        train_coverage_over_test = len(train_pid_set & test_pid_set) / len(test_pid_set)
    else:
        train_coverage_over_test = None
    logging.info(
        f"Train samples: {n_train}  |  Train PID coverage (all): {train_pid_coverage:.4f}  "
        f"|  Train coverage over test: {train_coverage_over_test:.4f}"
    )
    print(f"[pid_keep={pid_keep_ratio:.2f}] train_samples={n_train}, pid_coverage={train_pid_coverage:.4f}, "
          f"coverage_over_test={train_coverage_over_test:.4f}")

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize model
    model = TIGER(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Training loop — keep best checkpoint by val NDCG@20
    best_val_ndcg = 0.0
    best_test_recalls = None
    best_test_ndcgs = None
    best_recalled_pids = set()

    for epoch in range(config['num_epochs']):
        train_loss = train(model, train_dataloader, optimizer, device)
        avg_recalls, avg_ndcgs, _ = evaluate(
            model, validation_dataloader,
            config['topk_list'], config['beam_size'], device,
            code_to_item=None  # skip PID tracking on validation for speed
        )
        val_ndcg20 = avg_ndcgs['NDCG@20']
        logging.info(
            f"Epoch {epoch+1}/{config['num_epochs']} | loss={train_loss:.4f} | val_NDCG@20={val_ndcg20:.4f}"
        )

        if val_ndcg20 > best_val_ndcg:
            best_val_ndcg = val_ndcg20
            # Evaluate on test set with full PID tracking
            best_test_recalls, best_test_ndcgs, best_recalled_pids = evaluate(
                model, test_dataloader,
                config['topk_list'], config['beam_size'], device,
                code_to_item=code_to_item
            )
            logging.info(f"  New best! test_Recall@10={best_test_recalls['Recall@10']:.4f}")

            # Save checkpoint
            os.makedirs(config['save_dir'], exist_ok=True)
            ckpt_path = os.path.join(
                config['save_dir'], f"pid_keep{int(pid_keep_ratio*100):03d}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)

    recalled_pid_coverage = len(best_recalled_pids) / total_pids

    result = {
        'pid_keep_ratio': pid_keep_ratio,
        'n_train_samples': n_train,
        'train_pid_coverage': train_pid_coverage,              # train target PIDs / all items
        'train_coverage_over_test': train_coverage_over_test,  # train target PIDs ∩ test target PIDs / test target PIDs
        'recalled_pid_coverage': recalled_pid_coverage,
        'recalled_pid_count': len(best_recalled_pids),
        'total_pids': total_pids,
    }
    for k in config['topk_list']:
        result[f'Recall@{k}'] = best_test_recalls[f'Recall@{k}'] if best_test_recalls else 0.0
        result[f'NDCG@{k}'] = best_test_ndcgs[f'NDCG@{k}'] if best_test_ndcgs else 0.0

    logging.info(f"Result: {result}")
    print(f"[pid_keep={pid_keep_ratio:.2f}] recalled_pid_coverage={recalled_pid_coverage:.4f}, "
          f"Recall@10={result['Recall@10']:.4f}")
    return result


def main():
    args = parse_args()
    config = dict(DEFAULT_CONFIG)
    config['num_epochs'] = args.epochs
    config['device'] = args.device

    os.makedirs(os.path.dirname(config['log_path']), exist_ok=True)
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Also log to stdout
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    if config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Config: {config}")

    # Load item→code mapping (shared across all runs)
    item_to_code, code_to_item = item2code(config['code_path'])
    total_pids = len(item_to_code)
    logging.info(f"Total PIDs: {total_pids}")

    # Build validation and test dataloaders (shared, no drop)
    validation_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/valid.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len'],
    )
    test_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/test.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len'],
    )
    validation_dataloader = GenRecDataLoader(
        validation_dataset, batch_size=config['infer_size'], shuffle=False
    )
    test_dataloader = GenRecDataLoader(
        test_dataset, batch_size=config['infer_size'], shuffle=False
    )

    # Pre-compute test target PID set (used for coverage_over_test metric)
    test_pid_set = test_dataset.get_train_pid_set()
    logging.info(f"Unique test target PIDs: {len(test_pid_set)}")

    # Run experiments
    all_results = []
    for pid_keep_ratio in args.pid_keep_ratios:
        result = run_single_experiment(
            pid_keep_ratio=pid_keep_ratio,
            config=config,
            device=device,
            total_pids=total_pids,
            code_to_item=code_to_item,
            validation_dataloader=validation_dataloader,
            test_dataloader=test_dataloader,
            test_pid_set=test_pid_set,
        )
        all_results.append(result)

        # Save intermediate results after each run
        df = pd.DataFrame(all_results)
        df.to_csv(config['results_path'], index=False)
        print(f"\nResults saved to {config['results_path']}")
        print(df.to_string(index=False))

    # Final summary
    df = pd.DataFrame(all_results)
    df.to_csv(config['results_path'], index=False)
    logging.info("\n=== Final Results ===")
    logging.info(df.to_string(index=False))
    print("\n=== Experiment Complete ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
