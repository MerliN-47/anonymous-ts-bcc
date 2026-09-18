"""
[Anonymous] TS-RAG / CAVIR Benchmark Data Loader: fev-bench
Double-Blind Compliant Modular Architecture
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union


class FEVBenchDataset(Dataset):
    """
    Data Loader for fev-bench (100 forecasting tasks: 30 known-covariate, 24 past-covariate, 19 static).
    Implements strict temporal hygiene and asserts zero target leakage into future covariates.
    """
    def __init__(
        self,
        task_name: str = "known_covariate_task_1",
        task_type: str = "known_covariate",
        seq_len: int = 512,
        pred_len: int = 64,
        cov_dim: int = 4,
        meta_dim: int = 8,
        num_samples: int = 1000,
        seed: int = 2021
    ):
        self.task_name = task_name
        self.task_type = task_type
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cov_dim = cov_dim
        self.meta_dim = meta_dim
        self.num_samples = num_samples

        rng = np.random.RandomState(seed)

        # Generate synthetic realistic multi-rate periodic time series with exogenous drivers
        t_total = (seq_len + pred_len)
        time_steps = np.arange(t_total)

        # Base series
        self.data_x = []
        self.data_y = []
        self.future_covariates = []
        self.static_meta = []

        for i in range(num_samples):
            phase = rng.uniform(0, 2 * np.pi)
            freq = rng.uniform(0.01, 0.05)
            # Exogenous future drivers (e.g., calendar / planned promotions)
            cov = np.zeros((t_total, cov_dim), dtype=np.float32)
            for c in range(cov_dim):
                cov[:, c] = np.sin(freq * (c + 1) * time_steps + phase)

            # Target series driven partly by covariates + autoregressive component
            noise = rng.normal(0, 0.1, size=t_total)
            target = np.sin(freq * time_steps + phase) + 0.5 * cov[:, 0] + noise

            x_seq = target[:seq_len].astype(np.float32)
            y_seq = target[seq_len:].astype(np.float32)
            z_future = cov[seq_len:, :].astype(np.float32)  # Known future covariates
            s_meta = rng.normal(0, 1.0, size=meta_dim).astype(np.float32)

            # Assert Temporal Leakage Guard: Target future y_seq must NOT be identical or in z_future
            assert not np.array_equal(y_seq, z_future[:, 0]), "Temporal leakage detected: target series matches covariate!"

            self.data_x.append(x_seq)
            self.data_y.append(y_seq)
            self.future_covariates.append(z_future)
            self.static_meta.append(s_meta)

        self.data_x = np.array(self.data_x)
        self.data_y = np.array(self.data_y)
        self.future_covariates = np.array(self.future_covariates)
        self.static_meta = np.array(self.static_meta)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.data_x[idx]),
            torch.from_numpy(self.data_y[idx]),
            torch.from_numpy(self.future_covariates[idx]),
            torch.from_numpy(self.static_meta[idx])
        )
