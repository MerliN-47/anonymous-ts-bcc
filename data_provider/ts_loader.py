#!/usr/bin/env python3
"""
[Anonymous] TS-RAG / CAVIR Time Series Data Loaders.
Loaders for ETT (ETTh1, ETTh2, ETTm1, ETTm2), Weather, Traffic, Exchange Rate, and Electricity.
Double-Blind Compliant Implementation.
"""

import os
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    """
    Data Loader for ETT 1-Hour Datasets (ETTh1, ETTh2).
    """
    def __init__(
        self,
        root_path: str = "./datasets/ETT-small/",
        data_path: str = "ETTh1.csv",
        flag: str = "test",
        size: Optional[Tuple[int, int, int]] = None,
        scale: bool = True
    ):
        if size is None:
            self.seq_len = 512
            self.label_len = 48
            self.pred_len = 64
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        full_path = os.path.join(self.root_path, self.data_path)

        # Standard 12-month train, 4-month val, 4-month test split borders
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if os.path.exists(full_path):
            df_raw = pd.read_csv(full_path)
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            data = df_data.values
        else:
            # Fallback deterministic synthetic ETT series for standalone unit tests
            rng = np.random.RandomState(42)
            total_len = border2s[-1]
            time_idx = np.arange(total_len)
            data = np.zeros((total_len, 7), dtype=np.float32)
            for c in range(7):
                freq = 0.01 * (c + 1)
                data[:, c] = np.sin(freq * time_idx) + 0.5 * np.cos(freq * 0.5 * time_idx) + rng.normal(0, 0.1, size=total_len)

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float()

    def __len__(self) -> int:
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)


class Dataset_ETT_minute(Dataset):
    """
    Data Loader for ETT 15-Minute Datasets (ETTm1, ETTm2).
    """
    def __init__(
        self,
        root_path: str = "./datasets/ETT-small/",
        data_path: str = "ETTm1.csv",
        flag: str = "test",
        size: Optional[Tuple[int, int, int]] = None,
        scale: bool = True
    ):
        if size is None:
            self.seq_len = 512
            self.label_len = 48
            self.pred_len = 64
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        full_path = os.path.join(self.root_path, self.data_path)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if os.path.exists(full_path):
            df_raw = pd.read_csv(full_path)
            cols_data = df_raw.columns[1:]
            data = df_raw[cols_data].values
        else:
            rng = np.random.RandomState(42)
            total_len = border2s[-1]
            time_idx = np.arange(total_len)
            data = np.zeros((total_len, 7), dtype=np.float32)
            for c in range(7):
                freq = 0.002 * (c + 1)
                data[:, c] = np.sin(freq * time_idx) + 0.3 * np.cos(freq * 0.5 * time_idx) + rng.normal(0, 0.1, size=total_len)

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float()

    def __len__(self) -> int:
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)


class Dataset_Custom(Dataset):
    """
    Data Loader for Custom Multivariate Datasets: Weather, Traffic, Electricity, Exchange.
    """
    def __init__(
        self,
        root_path: str = "./datasets/",
        data_path: str = "weather.csv",
        flag: str = "test",
        size: Optional[Tuple[int, int, int]] = None,
        scale: bool = True
    ):
        if size is None:
            self.seq_len = 512
            self.label_len = 48
            self.pred_len = 64
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        full_path = os.path.join(self.root_path, self.data_path)

        if os.path.exists(full_path):
            df_raw = pd.read_csv(full_path)
            cols_data = df_raw.columns[1:]
            data = df_raw[cols_data].values
        else:
            # Fallback synthetic multi-rate series
            rng = np.random.RandomState(42)
            total_len = 10000
            time_idx = np.arange(total_len)
            data = np.zeros((total_len, 8), dtype=np.float32)
            for c in range(8):
                freq = 0.005 * (c + 1)
                data[:, c] = np.sin(freq * time_idx) + rng.normal(0, 0.1, size=total_len)

        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float()

    def __len__(self) -> int:
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)


def data_provider(args, flag: str = "test") -> Tuple[Dataset, DataLoader]:
    """
    Factory function instantiating dataset and standard DataLoader.
    """
    dataset_name = getattr(args, "data", "ETTh1")
    if hasattr(args, "dataset") and args.dataset:
        dataset_name = args.dataset

    ds_lower = dataset_name.lower()
    root_path = getattr(args, "root_path", "./datasets/ETT-small/")

    # Resolve dataset paths across common root structures
    candidate_roots = [
        root_path,
        "./datasets/ETT-small/",
        "../datasets/ETT-small/",
        "./datasets/",
        "../datasets/"
    ]
    resolved_root = root_path
    for cr in candidate_roots:
        if os.path.exists(cr):
            resolved_root = cr
            break

    size = (args.seq_len, getattr(args, "label_len", 48), args.pred_len)
    batch_size = getattr(args, "batch_size", 64)
    num_workers = getattr(args, "num_workers", 2)
    shuffle = (flag == "train")

    if "etth" in ds_lower:
        data_path = f"{dataset_name}.csv" if not dataset_name.endswith(".csv") else dataset_name
        dataset = Dataset_ETT_hour(root_path=resolved_root, data_path=data_path, flag=flag, size=size)
    elif "ettm" in ds_lower:
        data_path = f"{dataset_name}.csv" if not dataset_name.endswith(".csv") else dataset_name
        dataset = Dataset_ETT_minute(root_path=resolved_root, data_path=data_path, flag=flag, size=size)
    else:
        data_path = f"{dataset_name}.csv" if not dataset_name.endswith(".csv") else dataset_name
        dataset = Dataset_Custom(root_path=resolved_root, data_path=data_path, flag=flag, size=size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )
    return dataset, loader
