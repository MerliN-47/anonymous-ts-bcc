"""
[Anonymous] TS-RAG / CAVIR Multimodal News / Event Loader
Double-Blind Compliant Modular Architecture
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union


class MultimodalDataset(Dataset):
    """
    Dataset for Multimodal / Text-Grounded Time Series Forecasting (Context-is-Key / TimesX).
    Pairs numerical sequences with external timestamped textual news/events.
    """
    NEWS_TEMPLATES = [
        "Federal reserve announces rate pause at 5.25%",
        "Unexpected crude oil supply disruption reported in key transit pipeline",
        "Quarterly earnings release exceeds consensus estimates by 12%",
        "Major blizzard causes nationwide transportation and freight delays",
        "Tech sector rally driven by breakthrough semiconductor manufacturing yields",
        "Consumer price index inflation printed lower than consensus at 2.9% YoY",
        "Global container shipping spot rates surged 18% week-over-week",
        "Utility grid operator reports record summer peak electricity load"
    ]

    COUNTERFACTUAL_EVENTS = [
        "SHOCK: Central bank announces emergency +100bps interest rate hike effective immediately",
        "SHOCK: Port strike shuts down maritime logistics hub",
        "SHOCK: Unexpected commodity export embargo enacted"
    ]

    def __init__(
        self,
        num_samples: int = 500,
        seq_len: int = 512,
        pred_len: int = 64,
        seed: int = 2021
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.pred_len = pred_len

        rng = np.random.RandomState(seed)
        t_total = seq_len + pred_len
        time_steps = np.arange(t_total)

        self.data_x = []
        self.data_y = []
        self.texts = []
        self.counterfactual_texts = []
        self.counterfactual_y = []

        for i in range(num_samples):
            freq = rng.uniform(0.01, 0.04)
            phase = rng.uniform(0, 2 * np.pi)
            base = np.sin(freq * time_steps + phase) + rng.normal(0, 0.05, size=t_total)

            event_idx = rng.randint(0, len(self.NEWS_TEMPLATES))
            news_text = self.NEWS_TEMPLATES[event_idx]

            cf_idx = rng.randint(0, len(self.COUNTERFACTUAL_EVENTS))
            cf_text = self.COUNTERFACTUAL_EVENTS[cf_idx]

            # In counterfactual scenario, the future exhibits a sharp negative or positive jump
            cf_jump = -0.5 if "hike" in cf_text or "strike" in cf_text else 0.5
            cf_y_seq = base[seq_len:].copy() + cf_jump

            self.data_x.append(base[:seq_len].astype(np.float32))
            self.data_y.append(base[seq_len:].astype(np.float32))
            self.texts.append(news_text)
            self.counterfactual_texts.append(cf_text)
            self.counterfactual_y.append(cf_y_seq.astype(np.float32))

        self.data_x = np.array(self.data_x)
        self.data_y = np.array(self.data_y)
        self.counterfactual_y = np.array(self.counterfactual_y)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        return {
            "x": torch.from_numpy(self.data_x[idx]),
            "y": torch.from_numpy(self.data_y[idx]),
            "text": self.texts[idx],
            "cf_text": self.counterfactual_texts[idx],
            "cf_y": torch.from_numpy(self.counterfactual_y[idx])
        }
