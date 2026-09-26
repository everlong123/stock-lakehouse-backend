"""PyTorch sequence dataset for LSTM training."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Sliding windows of scaled features with a next-close target."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int) -> None:
        if sequence_length <= 1:
            raise ValueError("sequence_length must be greater than 1.")
        if len(features) <= sequence_length:
            raise ValueError("Not enough rows to build LSTM sequences.")
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.features) - self.sequence_length

    def __getitem__(self, index: int):
        import torch

        x = self.features[index : index + self.sequence_length]
        y = self.targets[index + self.sequence_length - 1]
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)
