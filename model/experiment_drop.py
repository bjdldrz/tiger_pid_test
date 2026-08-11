"""
experiment_drop.py
==================
实验 A：随机 Drop 训练样本，验证样本量减少对 Recall@K 的影响。

Drop 方式：随机丢弃 20% 训练样本（交互记录粒度）
对比组  ：baseline（无 drop）

Baseline 复用：
  若 results/{dataset}_baseline.json 已存在，直接加载，不重复训练。
  所有三个实验共享同一份 baseline。

多 GPU：自动检测，DataParallel 透明包装。

使用方法：
  cd model
  python experiment_drop.py [--epochs 50] [--device cuda] [--dataset Beauty]
  python experiment_drop.py --drop_ratios 0.2 0.4   # 不含 0.0，使用已有 baseline
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from dataset import GenRecDataset, item2code
from dataloader import GenRecDataLoader
from main import (TIGER, train, evaluate, set_seed,
                  setup_device, wrap_model_multigpu,
                  save_baseline, load_baseline)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DROP_RATIOS = [0.2]   # baseline (0.0) is auto-handled separately

DATASET_CONFIGS = {
    'Beauty': {
        'dataset_path': '../data/Beauty',
        'code_path': '../data/Beauty/Beauty_t5_rqvae.npy',
        'results_path': '../results/Beauty_drop_experiment.csv',
        'baseline_path': '../results/Beauty_baseline.json',
        'log_path': '../results/Beauty_drop_experiment.log',
        'save_dir': '../results/Beauty_drop_ckpts',
    },
    'VK-LSVD': {
        'dataset_path': '../data/VK-LSVD/processed',
        'code_path': '../data/VK-LSVD/processed/VK_rqvae.npy',
        'results_path': '../results/VK_drop_experiment.csv',
        'baseline_path': '../results/VK_baseline.json',
        'log_path': '../results/VK_drop_experiment.log',
        'save_dir': '../results/VK_drop_ckpts',
    },
}

MODEL_CONFIG = dict(
    num_layers=4, num_decoder_layers=4,
    d_model=128, d_ff=1024, num_heads=6, d_kv=64,
    dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0,
    feed_forward_proj='relu',
    batch_size=256, infer_size=96,
    lr=1e-4, max_len=50,
    topk_list=[5, 10, 20], beam_size=30,
    seed=2025,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without val NDCG@20 improvement)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--drop_ratios', type=float, nargs='+', default=DROP_RATIOS,
                        help='Drop ratios to test (baseline 0.0 is handled automatically)')
    return parser.parse_args()


def run_experiment(filter_kwargs, tag, config, device,
                   total_pids, code_to_item, test_pid_set,
                   validation_dataloader, test_dataloader,
                   test_good_dataloader=None, test_bad_dataloader=None):
    """Train one model with given filter kwargs and return metrics dict."""
    set_seed(config['seed'])

    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len'],
        seed=config['seed'],
        **filter_kwargs,
    )

    n_train = len(train_dataset)
    train_pid_set = train_dataset.get_train_pid_set()
    train_pid_coverage = len(train_pid_set) / total_pids
    train_coverage_over_test = len(train_pid_set & test_pid_set) / len(test_pid_set)

    logging.info(f"[{tag}] samples={n_train}, pid_cov={train_pid_coverage:.4f}, "
                 f"cov_over_test={train_coverage_over_test:.4f}")
    print(f"[{tag}] samples={n_train}, pid_coverage={train_pid_coverage:.4f}, "
          f"coverage_over_test={train_coverage_over_test:.4f}")

    train_dataloader = GenRecDataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )

    model = TIGER(config)
    model = wrap_model_multigpu(model, device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_val_ndcg = 0.0
    best_test_recalls = best_test_ndcgs = None
    best_test_good_recalls = best_test_good_ndcgs = None
    best_test_bad_recalls  = best_test_bad_ndcgs  = None
    best_recalled_pids = set()
    no_improve_count = 0

    for epoch in range(config['num_epochs']):
        loss = train(model, train_dataloader, optimizer, device)
        recalls, ndcgs, _ = evaluate(
            model, validation_dataloader,
            config['topk_list'], config['beam_size'], device
        )
        val_ndcg20 = ndcgs['NDCG@20']
        logging.info(f"  epoch {epoch+1}/{config['num_epochs']} loss={loss:.4f} val_NDCG@20={val_ndcg20:.4f} (no_improve={no_improve_count})")

        if val_ndcg20 > best_val_ndcg:
            best_val_ndcg = val_ndcg20
            no_improve_count = 0
            best_test_recalls, best_test_ndcgs, best_recalled_pids = evaluate(
                model, test_dataloader,
                config['topk_list'], config['beam_size'], device,
                code_to_item=code_to_item
            )
            if test_good_dataloader is not None:
                best_test_good_recalls, best_test_good_ndcgs, _ = evaluate(
                    model, test_good_dataloader,
                    config['topk_list'], config['beam_size'], device
                )
            if test_bad_dataloader is not None:
                best_test_bad_recalls, best_test_bad_ndcgs, _ = evaluate(
                    model, test_bad_dataloader,
                    config['topk_list'], config['beam_size'], device
                )
            os.makedirs(config['save_dir'], exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(config['save_dir'], f"{tag}.pth")
            )
            logging.info(f"  new best Recall@10={best_test_recalls['Recall@10']:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= config['patience']:
                logging.info(f"  Early stopping at epoch {epoch+1} (no improvement for {config['patience']} epochs)")
                print(f"[{tag}] Early stopping at epoch {epoch+1}")
                break

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
        result[f'Recall@{k}']      = best_test_recalls[f'Recall@{k}'] if best_test_recalls else 0.0
        result[f'NDCG@{k}']        = best_test_ndcgs[f'NDCG@{k}']    if best_test_ndcgs    else 0.0
        if best_test_good_recalls:
            result[f'Good_Recall@{k}'] = best_test_good_recalls[f'Recall@{k}']
            result[f'Good_NDCG@{k}']   = best_test_good_ndcgs[f'NDCG@{k}']
        if best_test_bad_recalls:
            result[f'Bad_Recall@{k}']  = best_test_bad_recalls[f'Recall@{k}']
            result[f'Bad_NDCG@{k}']    = best_test_bad_ndcgs[f'NDCG@{k}']

    print(f"[{tag}] recalled_pid_cov={recalled_pid_coverage:.4f} "
          f"Recall@10={result['Recall@10']:.4f}"
          + (f" Good@10={result.get('Good_Recall@10', 0):.4f}" if best_test_good_recalls else "")
          + (f" Bad@10={result.get('Bad_Recall@10',  0):.4f}" if best_test_bad_recalls  else ""))
    return result


def main():
    args = parse_args()
    dc = DATASET_CONFIGS[args.dataset]
    config = {**MODEL_CONFIG, **dc,
              'num_epochs': args.epochs, 'patience': args.patience,
              'device': args.device}

    os.makedirs(os.path.dirname(dc['log_path']), exist_ok=True)
    logging.basicConfig(
        filename=dc['log_path'], level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    device = setup_device(args.device)
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    logging.info(f"device={device}, GPUs={n_gpus}, dataset={args.dataset}")

    item_to_code, code_to_item = item2code(config['code_path'])
    total_pids = len(item_to_code)

    valid_ds = GenRecDataset(dc['dataset_path'] + '/valid.parquet',
                             config['code_path'], 'evaluation', config['max_len'])
    test_ds  = GenRecDataset(dc['dataset_path'] + '/test.parquet',
                             config['code_path'], 'evaluation', config['max_len'])
    valid_dl = GenRecDataLoader(valid_ds, int(config['infer_size']), shuffle=False)
    test_dl  = GenRecDataLoader(test_ds,  int(config['infer_size']), shuffle=False)
    test_pid_set = test_ds.get_train_pid_set()

    # test_good (reward >= 0.2) / test_bad (reward < 0.2)
    def _load_split_dl(fname):
        path = dc['dataset_path'] + f'/{fname}'
        if not os.path.exists(path):
            logging.warning(f"{fname} not found, skipping")
            return None
        ds = GenRecDataset(path, config['code_path'], 'evaluation', config['max_len'])
        logging.info(f"Loaded {fname}: {len(ds)} samples")
        return GenRecDataLoader(ds, int(config['infer_size']), shuffle=False)

    test_good_dl = _load_split_dl('test_good.parquet')
    test_bad_dl  = _load_split_dl('test_bad.parquet')

    all_results = []

    # ---- Baseline (drop_ratio=0.0): load or train once ----
    baseline = load_baseline(dc['baseline_path'])
    if baseline is None:
        logging.info("No baseline found, training baseline now...")
        baseline = run_experiment(
            filter_kwargs={},
            tag='baseline',
            config=config, device=device,
            total_pids=total_pids, code_to_item=code_to_item,
            test_pid_set=test_pid_set,
            validation_dataloader=valid_dl, test_dataloader=test_dl,
            test_good_dataloader=test_good_dl, test_bad_dataloader=test_bad_dl,
        )
        save_baseline(baseline, dc['baseline_path'])
    else:
        logging.info("Baseline loaded from cache, skipping training.")

    all_results.append(baseline)

    # ---- Drop experiments ----
    for drop_ratio in args.drop_ratios:
        tag = f'drop_{int(drop_ratio*100):03d}pct'
        result = run_experiment(
            filter_kwargs={'drop_ratio': drop_ratio},
            tag=tag,
            config=config, device=device,
            total_pids=total_pids, code_to_item=code_to_item,
            test_pid_set=test_pid_set,
            validation_dataloader=valid_dl, test_dataloader=test_dl,
            test_good_dataloader=test_good_dl, test_bad_dataloader=test_bad_dl,
        )
        all_results.append(result)

        df = pd.DataFrame(all_results)
        df.to_csv(config['results_path'], index=False)
        print(df.to_string(index=False))

    df = pd.DataFrame(all_results)
    df.to_csv(config['results_path'], index=False)
    logging.info("=== experiment_drop complete ===\n" + df.to_string(index=False))


if __name__ == '__main__':
    main()
