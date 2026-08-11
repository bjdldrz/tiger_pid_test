import argparse
import collections
import json
import logging
import os

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Beauty',
                    help='Dataset name: Beauty or VK-LSVD')
parser.add_argument('--ckpt_path', type=str, default=None,
                    help='Path to checkpoint. If omitted, auto-resolved from dataset ckpt dir.')
cli_args = parser.parse_args()

dataset = cli_args.dataset

DATASET_CONFIG = {
    'Beauty': {
        'data_path': '../data/Beauty/item_emb.parquet',
        'output_file': '../data/Beauty/Beauty_t5_rqvae.npy',
        'ckpt_dir': './ckpt/Beauty',
    },
    'VK-LSVD': {
        'data_path': '../data/VK-LSVD/processed/item_emb.npy',
        'output_file': '../data/VK-LSVD/processed/VK_rqvae.npy',
        'ckpt_dir': './ckpt/VK-LSVD',
    },
}

if dataset not in DATASET_CONFIG:
    raise ValueError(f"Unknown dataset '{dataset}'. Supported: {list(DATASET_CONFIG.keys())}")

cfg = DATASET_CONFIG[dataset]

# Resolve checkpoint path
if cli_args.ckpt_path:
    ckpt_path = cli_args.ckpt_path
else:
    ckpt_dir = cfg['ckpt_dir']
    if os.path.isdir(ckpt_dir):
        subdirs = sorted(os.listdir(ckpt_dir))
        if subdirs:
            ckpt_path = os.path.join(ckpt_dir, subdirs[-1], 'best_collision_model.pth')
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    else:
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"Please train RQ-VAE first:\n"
            f"  cd rqvae && python main.py --dataset {dataset}\n"
            f"Or pass --ckpt_path <path>"
        )

output_file = cfg['output_file']

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Load checkpoint and model
# ---------------------------------------------------------------------------
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]

data = EmbDataset(cfg['data_path'])
print(f"Dataset: {dataset}, items: {len(data)}, embedding dim: {data.dim}")

model = RQVAE(in_dim=data.dim,
              num_emb_list=args.num_emb_list,
              e_dim=args.e_dim,
              layers=args.layers,
              dropout_prob=args.dropout_prob,
              bn=args.bn,
              loss_type=args.loss_type,
              quant_loss_weight=args.quant_loss_weight,
              kmeans_init=args.kmeans_init,
              kmeans_iters=args.kmeans_iters,
              sk_epsilons=args.sk_epsilons,
              sk_iters=args.sk_iters,
              )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data, num_workers=0,
                         batch_size=64, shuffle=False,
                         pin_memory=False)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))
        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0

tt = 0
while True:
    if tt >= 30 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)

        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ", len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate", (tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

codes = []
for key, value in all_indices_dict.items():
    code = [int(item.split('_')[1].strip('>')) for item in value]
    codes.append(code)

codes_array = np.array(codes)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]

if len(duplicates) > 0:
    print("Resolving duplicates in codes...")
    for duplicate in duplicates:
        duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
        for i, idx in enumerate(duplicate_indices):
            codes_array[idx, -1] = i

new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = new_unique_codes[new_counts > 1]

if len(duplicates) > 0:
    print("There still have duplicates:", duplicates)
else:
    print("There are no duplicates in the codes after resolution.")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
print(f"Saving codes to {output_file}")
print(f"the first 5 codes: {codes_array[:5]}")
np.save(output_file, codes_array)
