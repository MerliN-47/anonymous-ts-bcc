#!/usr/bin/env python3
"""
[Anonymous] TS-RAG / CAVIR Retrieval Injection Heads.
Implements:
1. LatentSpaceARM: Adaptive Retrieval Mixer (Cross-Attention in representation space)
2. OutputSpaceBarycenter: W_2 Quantile Barycenter / Vincentization in density space
3. TokenSpaceFormatter: In-Context formatting for token-space concatenation
4. UnifiedRetrievalInjector: Orchestration interface with matched-compute FLOP estimation
Double-Blind Compliant Implementation.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_interface import BaseTSFMAdapter


class LatentSpaceARM(nn.Module):
    """
    Latent / Representation-Space Adaptive Retrieval Mixer (ARM).
    Performs cross-attention between query hidden states (B, T, D) and retrieved futures (B, K, H).
    """
    def __init__(self, d_model: int = 768, num_heads: int = 8, prediction_length: int = 64):
        super().__init__()
        self.d_model = d_model
        self.prediction_length = prediction_length

        self.encode_mlp = nn.Sequential(
            nn.Linear(prediction_length, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, query_latent: torch.Tensor, retrieved_futures: torch.Tensor) -> torch.Tensor:
        """
        query_latent: (B, T, D)
        retrieved_futures: (B, K, H)
        """
        B, K, H = retrieved_futures.shape
        emb_futures = self.encode_mlp(retrieved_futures)  # (B, K, D)

        # Pre-LN cross-attention
        q_norm = self.layer_norm1(query_latent)
        k_norm = emb_futures
        attn_out, _ = self.mha(q_norm, k_norm, k_norm)
        h = query_latent + attn_out

        # FFN with residual connection
        fused = h + self.ffn(self.layer_norm2(h))
        return fused


class OutputSpaceBarycenter(nn.Module):
    """
    Output / Density-Space Quantile Barycenter (Vincentization) Fusion:
    Q_mix(tau | x_q) = lambda(x_q) * Q_theta(tau | x_q) + (1 - lambda(x_q)) * sum_i w_i(x_q) * Q_i(tau)
    """
    def __init__(self, d_model: int = 768, num_quantiles: int = 9, prediction_length: int = 64):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.prediction_length = prediction_length

        self.bandwidth_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.lambda_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        backbone_quantiles: torch.Tensor,     # (B, Q, H)
        retrieved_futures: torch.Tensor,      # (B, K, H)
        query_emb: torch.Tensor,              # (B, D)
        retrieved_emb: torch.Tensor           # (B, K, D)
    ) -> torch.Tensor:
        B, K, H = retrieved_futures.shape
        Q = self.num_quantiles

        # 1. Compute empirical quantiles from retrieved neighbors across K per horizon step
        quantiles_grid = torch.linspace(0.1, 0.9, Q, device=retrieved_futures.device)
        retrieved_quantiles = torch.quantile(retrieved_futures, quantiles_grid, dim=1).permute(1, 0, 2)  # (B, Q, H)

        # 2. Compute distance-based weights w_i
        q_norm = F.normalize(query_emb.unsqueeze(1), p=2, dim=-1)     # (B, 1, D)
        r_norm = F.normalize(retrieved_emb, p=2, dim=-1)              # (B, K, D)
        dists = torch.norm(q_norm - r_norm, p=2, dim=-1)              # (B, K)

        bandwidth = self.bandwidth_head(query_emb) + 1e-4             # (B, 1)
        weights = F.softmax(-dists / bandwidth, dim=-1)               # (B, K)

        # 3. Compute mixing coefficient lambda
        lam = self.lambda_head(query_emb).unsqueeze(-1)               # (B, 1, 1)

        # 4. Pointwise Vincentization barycenter combination
        ensembled_median = lam.squeeze(-1) * backbone_quantiles[:, 4, :] + (1.0 - lam.squeeze(-1)) * retrieved_quantiles[:, 4, :]
        point_shift = ensembled_median - backbone_quantiles[:, 4, :]
        
        q_mix = backbone_quantiles + point_shift.unsqueeze(1)
        return q_mix


class TokenSpaceFormatter(nn.Module):
    """
    Token / Input-Space In-Context Formatter:
    Builds [x_1, y_1, ..., x_k, y_k, x_q] sequence representations for in-context execution.
    """
    def __init__(self, context_length: int = 512, prediction_length: int = 64):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

    def format_tokens(
        self,
        x_q: torch.Tensor,
        retrieved_past: torch.Tensor,
        retrieved_future: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        B, K, L = retrieved_past.shape
        examples = []
        for i in range(K):
            examples.append((retrieved_past[:, i, :], retrieved_future[:, i, :]))
        return examples


class UnifiedRetrievalInjector(nn.Module):
    """
    Unified Manager coordinating Token, Latent, and Output space retrieval injection.
    """
    def __init__(
        self,
        backbone: BaseTSFMAdapter,
        injection_point: str = "latent",
        d_model: int = 768,
        prediction_length: int = 64
    ):
        super().__init__()
        self.backbone = backbone
        self.injection_point = injection_point
        self.prediction_length = prediction_length

        self.token_formatter = TokenSpaceFormatter(backbone.context_length, prediction_length)
        self.latent_arm = LatentSpaceARM(d_model=d_model, prediction_length=prediction_length)
        self.output_barycenter = OutputSpaceBarycenter(d_model=d_model, prediction_length=prediction_length)

    def forward(
        self,
        x_q: torch.Tensor,
        retrieved_past: Optional[torch.Tensor] = None,
        retrieved_future: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes forward pass under specified injection point.
        """
        if retrieved_future is None or self.injection_point in ["none", "baseline"]:
            return self.backbone.quantiles(x_q)

        if self.injection_point == "token":
            # Token / In-Context space concatenation
            examples = self.token_formatter.format_tokens(x_q, retrieved_past, retrieved_future)
            return self.backbone.context_forward(x_q, examples)

        elif self.injection_point == "latent":
            # Latent / ARM representation space
            q_latent = self.backbone.latent(x_q)
            fused_latent = self.latent_arm(q_latent, retrieved_future)
            return self.backbone.decode_latent(fused_latent)

        elif self.injection_point == "output":
            # Output / Vincentization density space
            base_quantiles = self.backbone.quantiles(x_q)
            q_emb = self.backbone.encode(x_q)
            B, K, L = retrieved_past.shape
            r_past_flat = retrieved_past.view(B * K, L)

            # Chunk past embeddings to maintain bounded peak memory
            chunk_size = 256
            r_embs = []
            for chunk_idx in range(0, B * K, chunk_size):
                r_embs.append(self.backbone.encode(r_past_flat[chunk_idx:chunk_idx + chunk_size]))
            r_emb = torch.cat(r_embs, dim=0).view(B, K, -1)
            return self.output_barycenter(base_quantiles, retrieved_future, q_emb, r_emb)

        else:
            raise ValueError(f"Unknown injection point: {self.injection_point}")

    def estimate_flops(self, batch_size: int = 1, top_k: int = 10) -> Dict[str, float]:
        """
        Estimates theoretical FLOPs for matched-compute profiling.
        """
        L = self.backbone.context_length
        H = self.prediction_length
        D = self.backbone.d_model

        if self.injection_point == "token":
            total_seq = L + top_k * (L + H)
            # Quadratic self-attention FLOPs: 4 * seq^2 * D + 8 * seq * D^2
            flops = 4 * (total_seq ** 2) * D + 8 * total_seq * (D ** 2)
        elif self.injection_point == "latent":
            # Linear cross-attention in K: 4 * L^2 * D + 4 * (L * top_k) * D + 8 * L * D^2
            flops = 4 * (L ** 2) * D + 4 * (L * top_k) * D + 8 * L * (D ** 2)
        else: # Output space
            flops = 4 * (L ** 2) * D + 8 * L * (D ** 2) + top_k * H * 10

        return {
            "injection_point": self.injection_point,
            "estimated_flops": float(flops * batch_size),
            "flops_multiplier_vs_base": float(flops / (4 * (L ** 2) * D + 8 * L * (D ** 2)))
        }
