import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


class EmbDataset(data.Dataset):
    """
    Item embedding dataset.
    Supports two formats:
      - .parquet : must have an 'embedding' column (list/array per row), e.g. Beauty
      - .npy     : shape (n_items, emb_dim), e.g. VK-LSVD item_emb.npy
    """

    def __init__(self, data_path):
        self.data_path = data_path
        if str(data_path).endswith('.npy'):
            self.embeddings = np.load(data_path).astype(np.float32)
        else:
            self.embeddings = pd.read_parquet(data_path, engine='fastparquet')['embedding'].values
            self.embeddings = np.stack(self.embeddings, axis=0).astype(np.float32)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        return torch.FloatTensor(self.embeddings[index])

    def __len__(self):
        return len(self.embeddings)
