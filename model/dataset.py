import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset

def process_data(file_path, mode, max_len, PAD_TOKEN=0):
    """
    Process parquet data based on mode ('train' or 'evaluation').

    Args:
        file_path (str): Path to the parquet file.
        mode (str): Mode of operation ('train' or 'evaluation').
        max_len (int): Maximum length for padding or truncation.

    Returns:
        list: Processed data.
    """
    # Load parquet data
    data = pd.read_parquet(file_path, engine='fastparquet')

    # Combine "history" and "target" columns into a single sequence
    # Important: Ensure 'history' is a list and 'target' is appended correctly
    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])

    if mode == 'train':
        # Sliding window processing
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append({
                    'history': sequence[:i],
                    'target': sequence[i]
                })
    elif mode == 'evaluation':
        # Use the last item as target and the rest as history
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            processed_data.append({
                'history': sequence[:-1],
                'target': sequence[-1]
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    # Apply padding or truncation
    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence
    
def item2code(code_path, codebook_size=256):
    """
    Convert itemID to code
    :param code_path: npy file path to store rqvae codes
    :return: dict item_to_code, code_to_item
    """
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}
    
    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i,c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, mode, max_len,
                 drop_ratio=0.0, pid_keep_ratio=1.0, seed=42, PAD_TOKEN=0):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            mode (str): Mode of operation ('train' or 'evaluation').
            max_len (int): Maximum length for padding or truncation.
            drop_ratio (float): Fraction of training samples to randomly drop (0.0~1.0).
                                Only applied when mode='train'. Default: 0.0 (no drop).
            pid_keep_ratio (float): Fraction of unique target PIDs to keep (0.0~1.0).
                                    Removes ALL samples whose target is a dropped PID.
                                    Only applied when mode='train'. Default: 1.0 (keep all PIDs).
                                    Note: pid_keep_ratio is applied first, then drop_ratio.
            seed (int): Random seed for reproducible sampling. Default: 42.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        # drop_ratio and pid_keep_ratio only apply to train mode
        self.drop_ratio = drop_ratio if mode == 'train' else 0.0
        self.pid_keep_ratio = pid_keep_ratio if mode == 'train' else 1.0
        self.seed = seed
        # Load item-to-code mapping
        self.item_to_code, self.code_to_item = item2code(code_path)
        # Process the dataset
        self.data = self._prepare_data()
        
    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Applies random drop when drop_ratio > 0 and mode == 'train'.
        Returns:
            list: Processed data with items converted to codes.
        """
        # Process the data using the process_data function
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
        )

        # PID-level drop: keep only pid_keep_ratio fraction of unique target PIDs
        # Applied BEFORE sample-level drop so PID coverage is precisely controlled
        if self.pid_keep_ratio < 1.0:
            rng_pid = random.Random(self.seed)
            all_pids = sorted(set(item['target'] for item in processed_data))
            n_keep_pids = int(len(all_pids) * self.pid_keep_ratio)
            kept_pids = set(rng_pid.sample(all_pids, n_keep_pids))
            processed_data = [item for item in processed_data if item['target'] in kept_pids]

        # Random sample-level drop: keep (1 - drop_ratio) fraction of remaining training samples
        if self.drop_ratio > 0.0:
            rng = random.Random(self.seed + 1)  # different seed from pid drop
            n_keep = int(len(processed_data) * (1.0 - self.drop_ratio))
            processed_data = rng.sample(processed_data, n_keep)

        # Convert items to codes, preserve raw_target for coverage stats
        for item in processed_data:
            item['raw_target'] = item['target']  # save original itemID before conversion
            item['history'] = [self.item_to_code.get(x, [self.PAD_TOKEN] * 4) for x in item['history']]
            item['target'] = self.item_to_code.get(item['target'], [self.PAD_TOKEN] * 4)
        return processed_data

    def get_train_pid_coverage(self, total_pids):
        """
        Compute the fraction of all PIDs that appear as target in this (possibly dropped) training set.
        Args:
            total_pids (int): Total number of unique items (PIDs) in the full dataset.
        Returns:
            float: Coverage ratio in [0, 1].
        """
        train_pids = set(item['raw_target'] for item in self.data)
        return len(train_pids) / total_pids

    def get_train_pid_set(self):
        """Return the set of target PIDs present in this training set."""
        return set(item['raw_target'] for item in self.data)

    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history' and 'target'.
        """
        return self.data[index]
    
    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)
    
if __name__ == "__main__":
    # Example usage
    dataset_path = '../data/Beauty/train.parquet'
    code_path = '../data/Beauty/Beauty_t5_rqvae.npy'
    mode = 'train'
    max_len = 20

    dataset = GenRecDataset(dataset_path, code_path, mode, max_len)
    print("Number of items in dataset:", len(dataset))
    print("First five items in dataset:", [dataset[i] for i in range(5)])

    # Test drop_ratio
    dataset_drop = GenRecDataset(dataset_path, code_path, mode, max_len, drop_ratio=0.5, seed=42)
    print(f"\nWith drop_ratio=0.5: {len(dataset_drop)} samples (original: {len(dataset)})")
    total_pids = len(dataset.item_to_code)
    print(f"Full train PID coverage: {dataset.get_train_pid_coverage(total_pids):.4f}")
    print(f"Dropped train PID coverage: {dataset_drop.get_train_pid_coverage(total_pids):.4f}")

    # Test pid_keep_ratio
    dataset_pid = GenRecDataset(dataset_path, code_path, mode, max_len, pid_keep_ratio=0.6, seed=42)
    print(f"\nWith pid_keep_ratio=0.6: {len(dataset_pid)} samples (original: {len(dataset)})")
    print(f"PID-dropped train PID coverage: {dataset_pid.get_train_pid_coverage(total_pids):.4f}")
