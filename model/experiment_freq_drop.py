"""
experiment_freq_drop.py
=======================
验证实验：基于 PID 频次的尾部过滤对 TIGER 召回率的影响。

实验逻辑：
  统计每个 PID 在训练集中作为 target 的出现次数，
  按频次升序排列，去掉出现次数最少的 freq_drop_ratio 比例的 PID
  及其所有训练样本，模拟 RSFT 对低质量 item 的过滤效果。

与随机 Drop 实验的区别：
  - 随机 drop：随机丢弃训练样本，PID 覆盖率间接下降
  - PID drop：随机选择要排除的 PID，精确控制覆盖率
  - 频次 drop（本实验）：系统性地删除低频（低质量）PID，
    模拟 RSFT 的效果，保留的 PID 偏向高频高质量 item

结果保存至：../results/freq_drop_experiment.csv

使用方法：
  cd model
  python experiment_freq_drop.py [--epochs 50] [--device cuda]
  python experiment_freq_drop.py --freq_drop_ratios 0.0 0.2 0.4
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
# Default: baseline (0.0) vs drop bottom-20% lowest-frequency PIDs (0.2)
FREQ_DROP_RATIOS = [0.0, 0.2]

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
    beam_size=20,
    # Paths
    dataset_path='../data/Beauty',
    code_path='../data/Beauty/Beauty_t5_rqvae.npy',
    results_path='../results/freq_drop_experiment.csv',
    save_dir='../results/freq_drop_ckpts',
    log_path='../results/freq_drop_experiment.log',
    # Reproducibility
    seed=2025,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Frequency-based tail drop experiment for TIGER")
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per run (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda, mps, or cpu')
    parser.add_argument('--freq_drop_ratios', type=float, nargs='+', default=FREQ_DROP_RATIOS,
                        help='List of freq drop ratios (e.g. 0.0 0.2 0.4)')
    return parser.parse_args()


def run_single_experiment(freq_drop_ratio, config, device, total_pids, code_to_item,
                           validation_dataloader, test_dataloader, test_pid_set=None):
    """Train one model with the given freq_drop_ratio and return metrics."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting experiment: freq_drop_ratio={freq_drop_ratio:.2f}")
    logging.info(f"{'='*60}")

    set_seed(config['seed'])

    # Build training dataset with frequency-based tail drop
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len'],
        freq_drop_ratio=freq_drop_ratio,
        seed=config['seed'],
    )

    train_pid_coverage = train_dataset.get_train_pid_coverage(total_pids)
    n_train = len(train_dataset)
    train_pid_set = train_dataset.get_train_pid_set()
    if test_pid_set:
        train_coverage_over_test = len(train_pid_set & test_pid_set) / len(test_pid_set)
    else:
        train_coverage_over_test = None

    logging.info(
        f"Train samples: {n_train}  |  Train PID coverage (all): {train_pid_coverage:.4f}  "
        f"|  Train coverage over test: {train_coverage_over_test:.4f}"
    )
    print(f"[freq_drop={freq_drop_ratio:.2f}] train_samples={n_train}, "
          f"pid_coverage={train_pid_coverage:.4f}, "
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
            best_test_recalls, best_test_ndcgs, best_recalled_pids = evaluate(
                model, test_dataloader,
                config['topk_list'], config['beam_size'], device,
                code_to_item=code_to_item
            )
            logging.info(f"  New best! test_Recall@10={best_test_recalls['Recall@10']:.4f}")

            os.makedirs(config['save_dir'], exist_ok=True)
            ckpt_path = os.path.join(
                config['save_dir'], f"freq_drop{int(freq_drop_ratio*100):03d}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)

    recalled_pid_coverage = len(best_recalled_pids) / total_pids

    result = {
        'freq_drop_ratio': freq_drop_ratio,
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

    logging.info(f"Result: {result}")
    print(f"[freq_drop={freq_drop_ratio:.2f}] recalled_pid_coverage={recalled_pid_coverage:.4f}, "
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
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    if config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    item_to_code, code_to_item = item2code(config['code_path'])
    total_pids = len(item_to_code)
    logging.info(f"Total PIDs: {total_pids}")

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

    # Pre-compute test target PID set
    test_pid_set = test_dataset.get_train_pid_set()
    logging.info(f"Unique test target PIDs: {len(test_pid_set)}")

    all_results = []
    for freq_drop_ratio in args.freq_drop_ratios:
        result = run_single_experiment(
            freq_drop_ratio=freq_drop_ratio,
            config=config,
            device=device,
            total_pids=total_pids,
            code_to_item=code_to_item,
            validation_dataloader=validation_dataloader,
            test_dataloader=test_dataloader,
            test_pid_set=test_pid_set,
        )
        all_results.append(result)

        df = pd.DataFrame(all_results)
        df.to_csv(config['results_path'], index=False)
        print(f"\nResults saved to {config['results_path']}")
        print(df.to_string(index=False))

    df = pd.DataFrame(all_results)
    df.to_csv(config['results_path'], index=False)
    logging.info("\n=== Final Results ===")
    logging.info(df.to_string(index=False))
    print("\n=== Experiment Complete ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
