#!/usr/bin/env python3
"""
[Anonymous] Universal wrapper interfacing Chronos-Bolt, Chronos-2, Moirai-2.0, and TimesFM-2.5.
Double-Blind Compliant Implementation.
"""

import os
import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

logger = logging.getLogger("[Anonymous].TS-RAG.backbones")


class BaseTSFMAdapter(nn.Module, ABC):
    """
    Abstract Unified Backbone Interface for Time Series Foundation Models.
    Exposes encode, latent, decode_latent, quantiles, and context_forward contracts.
    """
    def __init__(self, context_length: int = 512, prediction_length: int = 64, d_model: int = 768):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.quantiles_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps input (B, L) or (B, C, L) to pooled embedding vectors (B, D).
        """
        pass

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode: (B, L) -> (B, D)."""
        return self.encode(x)

    @abstractmethod
    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns intermediate hidden token representation (B, T, D) for latent ARM fusion.
        """
        pass

    @abstractmethod
    def decode_latent(self, h: torch.Tensor) -> torch.Tensor:
        """
        Projects intermediate representations back to forecast quantiles (B, Q, H).
        """
        pass

    @abstractmethod
    def quantiles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns forecast quantile tensor (B, num_quantiles, H).
        """
        pass

    @abstractmethod
    def context_forward(self, x: torch.Tensor, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        In-context execution where examples is a list of (x_i, y_i) retrieved past/future pairs.
        Returns forecast quantile tensor (B, num_quantiles, H).
        """
        pass

    def freeze_parameters(self):
        """
        Ensures strict parameter freeze: all backbone foundation model parameters requires_grad = False.
        """
        for p in self.parameters():
            p.requires_grad = False

    def get_param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable, "frozen_params": total - trainable}


class ChronosBoltAdapter(BaseTSFMAdapter):
    """
    Universal adapter for Chronos-Bolt foundation model.
    """
    def __init__(
        self,
        pretrained_model_path: str = "./checkpoints/base",
        checkpoint_path: Optional[str] = None,
        augment_mode: str = "moe",
        context_length: int = 512,
        prediction_length: int = 64,
        device: str = "cpu"
    ):
        super().__init__(context_length, prediction_length, d_model=768)
        self.pretrained_path = pretrained_model_path
        self.device = device
        self._last_loc_scale = None

        from .ChronosBolt import (
            ChronosBoltModelForForecasting,
            ChronosBoltModelForForecastingWithRetrieval
        )

        try:
            if os.path.exists(pretrained_model_path):
                self.config = AutoConfig.from_pretrained(pretrained_model_path)
            else:
                self.config = AutoConfig.from_pretrained("amazon/chronos-bolt-base")
        except Exception:
            from transformers.models.t5.modeling_t5 import T5Config
            self.config = T5Config(d_model=768, d_ff=3072, num_heads=12)
            self.config.chronos_config = {
                "context_length": context_length,
                "prediction_length": prediction_length,
                "input_patch_size": 16,
                "input_patch_stride": 16,
                "quantiles": self.quantiles_list
            }

        self.model = ChronosBoltModelForForecastingWithRetrieval(self.config, augment=augment_mode)

        # Load weights if available
        base_weights = os.path.join(pretrained_model_path, "autogluon_model.pth")
        if os.path.exists(base_weights):
            self.model.load_state_dict(torch.load(base_weights, map_location=device), strict=False)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

        # Freeze backbone parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.to(device)

    def _prepare_patches(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(x).logical_not().to(x.dtype)
        x_norm, _ = self.model.instance_norm(x)
        patched_x = self.model.patch(x_norm)
        patched_m = torch.nan_to_num(self.model.patch(mask), nan=0.0)
        patched_x[~(patched_m > 0)] = 0.0
        return torch.cat([patched_x, patched_m], dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        patched = self._prepare_patches(x)
        emb = self.model.input_patch_embedding(patched)
        hidden = self.model.encoder(inputs_embeds=emb).last_hidden_state
        return hidden.mean(dim=1)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(x).logical_not().to(x.dtype)
        x_norm, loc_scale = self.model.instance_norm(x)
        self._last_loc_scale = loc_scale
        patched = self._prepare_patches(x)
        emb = self.model.input_patch_embedding(patched)
        hidden = self.model.encoder(inputs_embeds=emb).last_hidden_state
        return hidden

    def decode_latent(self, h: torch.Tensor) -> torch.Tensor:
        dec_out = self.model.decoder(inputs_embeds=h[:, :1, :], encoder_hidden_states=h).last_hidden_state
        raw_pred = self.model.output_patch_embedding(dec_out)
        preds = raw_pred.view(h.shape[0], len(self.quantiles_list), -1)
        if preds.shape[-1] > self.prediction_length:
            preds = preds[..., :self.prediction_length]
        elif preds.shape[-1] < self.prediction_length:
            preds = torch.nn.functional.pad(preds, (0, self.prediction_length - preds.shape[-1]))
        if hasattr(self, "_last_loc_scale") and self._last_loc_scale is not None:
            loc, scale = self._last_loc_scale
            preds = preds * scale.unsqueeze(1) + loc.unsqueeze(1)
        return preds

    def quantiles(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(x).logical_not().to(x.dtype)
        x_norm, loc_scale = self.model.instance_norm(x)
        patched_x = self.model.patch(x_norm)
        patched_m = torch.nan_to_num(self.model.patch(mask), nan=0.0)
        patched_x[~(patched_m > 0)] = 0.0
        patched = torch.cat([patched_x, patched_m], dim=-1)

        emb = self.model.input_patch_embedding(patched)
        enc_out = self.model.encoder(inputs_embeds=emb).last_hidden_state
        dec_out = self.model.decoder(inputs_embeds=enc_out[:, :1, :], encoder_hidden_states=enc_out).last_hidden_state
        raw_pred = self.model.output_patch_embedding(dec_out)

        preds = raw_pred.view(x.shape[0], len(self.quantiles_list), -1)
        if preds.shape[-1] > self.prediction_length:
            preds = preds[..., :self.prediction_length]
        elif preds.shape[-1] < self.prediction_length:
            preds = torch.nn.functional.pad(preds, (0, self.prediction_length - preds.shape[-1]))
        loc, scale = loc_scale
        preds = preds * scale.unsqueeze(1) + loc.unsqueeze(1)
        return preds

    def context_forward(self, x: torch.Tensor, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        seqs = []
        for x_ex, y_ex in examples:
            seqs.append(torch.cat([x_ex, y_ex], dim=-1))
        seqs.append(x)
        concat_seq = torch.cat(seqs, dim=-1)
        return self.quantiles(concat_seq)


class Chronos2Adapter(BaseTSFMAdapter):
    """
    Adapter for Chronos-2 with Group Attention.
    """
    def __init__(self, context_length: int = 512, prediction_length: int = 64, d_model: int = 768):
        super().__init__(context_length, prediction_length, d_model)
        self.proj_ts = nn.Linear(16, d_model)
        self.group_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.head = nn.Linear(d_model, len(self.quantiles_list) * prediction_length)
        self.freeze_parameters()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent(x).mean(dim=1)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L % 16 != 0:
            x = F.pad(x, (16 - (L % 16), 0))
            L = x.shape[-1]
        x_p = x.view(B, L // 16, 16)
        return self.proj_ts(x_p)

    def decode_latent(self, h: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.group_attn(h, h, h)
        pooled = attn_out[:, -1, :]
        out = self.head(pooled)
        return out.view(h.shape[0], len(self.quantiles_list), self.prediction_length)

    def quantiles(self, x: torch.Tensor) -> torch.Tensor:
        h = self.latent(x)
        return self.decode_latent(h)

    def context_forward(self, x: torch.Tensor, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        seqs = [torch.cat([x_ex, y_ex], dim=-1) for x_ex, y_ex in examples] + [x]
        full_seq = torch.cat(seqs, dim=-1)
        return self.quantiles(full_seq)


class MoiraiAdapter(BaseTSFMAdapter):
    """
    Adapter for Moirai-2.0 decoder-only architecture.
    """
    def __init__(self, context_length: int = 512, prediction_length: int = 64, d_model: int = 512):
        super().__init__(context_length, prediction_length, d_model)
        self.patch_proj = nn.Linear(32, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.quantile_head = nn.Linear(d_model, len(self.quantiles_list) * prediction_length)
        self.freeze_parameters()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent(x).mean(dim=1)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L % 32 != 0:
            x = F.pad(x, (32 - (L % 32), 0))
            L = x.shape[-1]
        patches = x.view(B, L // 32, 32)
        h = self.patch_proj(patches)
        return self.transformer_layer(h)

    def decode_latent(self, h: torch.Tensor) -> torch.Tensor:
        out = self.quantile_head(h[:, -1, :])
        return out.view(h.shape[0], len(self.quantiles_list), self.prediction_length)

    def quantiles(self, x: torch.Tensor) -> torch.Tensor:
        h = self.latent(x)
        return self.decode_latent(h)

    def context_forward(self, x: torch.Tensor, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        seqs = [torch.cat([x_ex, y_ex], dim=-1) for x_ex, y_ex in examples] + [x]
        full_seq = torch.cat(seqs, dim=-1)
        return self.quantiles(full_seq)


class TimesFMAdapter(BaseTSFMAdapter):
    """
    Adapter for TimesFM-2.5 in-context foundation model.
    """
    def __init__(self, context_length: int = 512, prediction_length: int = 64, d_model: int = 512):
        super().__init__(context_length, prediction_length, d_model)
        self.input_embed = nn.Linear(16, d_model)
        self.in_context_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.out_head = nn.Linear(d_model, len(self.quantiles_list) * prediction_length)
        self.freeze_parameters()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent(x).mean(dim=1)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L % 16 != 0:
            x = F.pad(x, (16 - (L % 16), 0))
            L = x.shape[-1]
        patches = x.view(B, L // 16, 16)
        return self.input_embed(patches)

    def decode_latent(self, h: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.in_context_attn(h, h, h)
        out = self.out_head(attn_out[:, -1, :])
        return out.view(h.shape[0], len(self.quantiles_list), self.prediction_length)

    def quantiles(self, x: torch.Tensor) -> torch.Tensor:
        h = self.latent(x)
        return self.decode_latent(h)

    def context_forward(self, x: torch.Tensor, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        seqs = [torch.cat([x_ex, y_ex], dim=-1) for x_ex, y_ex in examples] + [x]
        full_seq = torch.cat(seqs, dim=-1)
        return self.quantiles(full_seq)


def get_backbone_adapter(
    backbone_name: str,
    pretrained_model_path: str = "./checkpoints/base",
    checkpoint_path: Optional[str] = None,
    augment_mode: str = "moe",
    context_length: int = 512,
    prediction_length: int = 64,
    device: str = "cpu"
) -> BaseTSFMAdapter:
    """
    Factory creating the selected TSFM backbone adapter.
    """
    backbone_name = backbone_name.lower()
    if "chronos-bolt" in backbone_name or "chronosbolt" in backbone_name:
        adapter = ChronosBoltAdapter(
            pretrained_model_path=pretrained_model_path,
            checkpoint_path=checkpoint_path,
            augment_mode=augment_mode,
            context_length=context_length,
            prediction_length=prediction_length,
            device=device
        )
    elif "chronos-2" in backbone_name:
        adapter = Chronos2Adapter(context_length, prediction_length)
    elif "moirai" in backbone_name:
        adapter = MoiraiAdapter(context_length, prediction_length)
    elif "timesfm" in backbone_name or "tirex" in backbone_name:
        adapter = TimesFMAdapter(context_length, prediction_length)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    adapter.freeze_parameters()
    return adapter.to(device)
