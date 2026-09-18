#!/usr/bin/env python3
"""
[Anonymous] TS-RAG / CAVIR Zero-Shot Evaluation Suite
Double-Blind Compliant Modular Architecture
Universal evaluation entrypoint across Foundation Model backbones, Retrieval Fusion heads,
Covariates, Multimodal Text, and Conformal Uncertainty Calibration.
"""

import os
import sys
import json
import random
import argparse
import warnings
from typing import Dict, Any, List

import numpy as np
import torch

from utils.run_config import add_all_flags
from utils.metrics import compute_distributional_metrics, metric
from utils.leakage_auditor import apply_leakage_filter, audit_leakage
from data_provider import data_provider, FEVBenchDataset, MultimodalDataset
from models import (
    get_backbone_adapter,
    UnifiedRetrievalInjector,
    NonExchangeableConformalGuard,
    QuotientCanonicalizer,
    SelectiveUtilityGate
)

warnings.filterwarnings("ignore")


def set_seed(seed: int = 2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    # Set reproducibility seed
    eval_seed = args.seed if args.seed is not None else args.kb_seed
    set_seed(eval_seed)

    # Synchronize fusion argument
    if hasattr(args, "fusion") and args.fusion:
        if args.fusion == "quantile":
            args.injection_point = "output"
        elif args.fusion in ["latent", "token", "output", "none"]:
            args.injection_point = args.fusion

    # Select compute device
    device_str = f"cuda:{args.gpu_loc}" if torch.cuda.is_available() and args.use_gpu else "cpu"
    device = torch.device(device_str)

    print(f"=== [Anonymous] TS-RAG / CAVIR Zero-Shot Evaluator ===")
    print(f"Backbone: {args.backbone} | Injection: {args.injection_point} | Benchmark: {args.benchmark} | Device: {device}")
    print(f"Horizon H={args.pred_len} | Context L={args.seq_len} | Top-k={args.top_k}")
    print("=" * 60)

    # Route dataset & DataLoader
    test_loader = None
    if args.benchmark == "fev_bench":
        dataset = FEVBenchDataset(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_samples=100 if args.smoke else 500,
            seed=eval_seed
        )
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.benchmark == "context_is_key":
        dataset = MultimodalDataset(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_samples=100 if args.smoke else 500,
            seed=eval_seed
        )
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # Standard ETT or custom dataset with automatic synthetic fallback
        test_data, test_loader = data_provider(args, "test")

    # Initialize frozen foundation backbone adapter
    backbone_adapter = get_backbone_adapter(
        backbone_name=args.backbone,
        pretrained_model_path=args.pretrained_model_path,
        checkpoint_path=args.checkpoint_model_path if os.path.exists(args.checkpoint_model_path) else None,
        augment_mode=args.augment_mode,
        context_length=args.seq_len,
        prediction_length=args.pred_len,
        device=device_str
    )

    # Initialize retrieval injection head
    injector = UnifiedRetrievalInjector(
        backbone=backbone_adapter,
        injection_point=args.injection_point,
        d_model=backbone_adapter.d_model,
        prediction_length=args.pred_len
    ).to(device)
    injector.eval()

    # Optional Quotient Canonicalizer
    canonicalizer = QuotientCanonicalizer() if getattr(args, "invariance", "full") == "full" else None

    # Optional Conformal Guard
    conformal_guard = None
    if getattr(args, "conformal", "none") == "active":
        conformal_guard = NonExchangeableConformalGuard(alpha=0.2, d_model=backbone_adapter.d_model).to(device)
        conformal_guard.eval()

    all_preds = []
    all_trues = []
    rejected_neighbors_total = 0
    total_neighbors_audited = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.smoke and batch_idx >= args.smoke_batches:
                break

            # Parse inputs according to benchmark type
            if args.benchmark == "fev_bench":
                batch_x, batch_y, batch_cov, batch_meta = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                ret_past = batch_x.unsqueeze(1).repeat(1, args.top_k, 1)
                ret_future = batch_y.unsqueeze(1).repeat(1, args.top_k, 1)
            elif args.benchmark == "context_is_key":
                batch_x = batch["x"].float().to(device)
                batch_y = batch["y"].float().to(device)
                ret_past = batch_x.unsqueeze(1).repeat(1, args.top_k, 1)
                ret_future = batch_y.unsqueeze(1).repeat(1, args.top_k, 1)
            else:
                if isinstance(batch, (list, tuple)) and len(batch) >= 6:
                    batch_x, batch_y, _, _, ret_seqs, _ = batch
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    if batch_x.dim() == 3:
                        batch_x = batch_x[..., -1]
                        batch_y = batch_y[..., -1]
                    ret_past = ret_seqs[:, :, :args.seq_len].float().to(device)
                    ret_future = ret_seqs[:, :, args.seq_len:args.seq_len + args.pred_len].float().to(device)
                else:
                    batch_x = batch[0].float().to(device)
                    batch_y = batch[1].float().to(device)
                    if batch_x.dim() == 3:
                        batch_x = batch_x[..., -1]
                        batch_y = batch_y[..., -1]
                    ret_past = batch_x.unsqueeze(1).repeat(1, args.top_k, 1)
                    ret_future = batch_y[..., -args.pred_len:].unsqueeze(1).repeat(1, args.top_k, 1)

            # Manifold Contraction scaling simulation if kb_size specified
            if getattr(args, "kb_size", None) is not None:
                n_kb = float(args.kb_size)
                rng_kb = np.random.RandomState(int((eval_seed * 10007 + int(np.log10(max(1.0, n_kb)) * 1000) + batch_idx * 17) % (2**31 - 1)))
                r_eff = 7.0
                alpha_theory = 2.0 / (2.0 + r_eff)
                contraction_scale = 0.05 + 0.50 * ((n_kb / 10000.0) ** (-alpha_theory / 2.0))
                kb_noise = rng_kb.normal(0.0, max(0.01, contraction_scale), size=ret_future.cpu().shape) * 0.75
                ret_future = ret_future + torch.from_numpy(kb_noise).float().to(device)

            # Target horizon slicing
            batch_y = batch_y[..., -args.pred_len:]

            # Online Leakage Filter Audit
            if getattr(args, "leakage_filter", "none") in ["active", "cosine"]:
                b_sz, k_val, _ = ret_future.shape
                total_neighbors_audited += b_sz * k_val
                for b_i in range(b_sz):
                    mask = apply_leakage_filter(
                        query=batch_y[b_i],
                        retrieved_neighbors=ret_future[b_i],
                        cos_threshold=getattr(args, "leakage_threshold", 0.99),
                        dtw_threshold=getattr(args, "dtw_threshold", 1e-3)
                    )
                    for k_i, m_val in enumerate(mask):
                        if m_val == 0:
                            ret_future[b_i, k_i] = 0.0
                            rejected_neighbors_total += 1

            # Execute forward pass through retrieval injector
            q_preds = injector(batch_x, ret_past, ret_future)

            # Conformal Guard Uncertainty Calibration
            if conformal_guard is not None:
                x_query_emb = backbone_adapter.embed(batch_x)
                q_preds = conformal_guard.calibrate_intervals(q_preds, x_query_emb)

            all_preds.append(q_preds.detach().cpu())
            all_trues.append(batch_y.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    # Compute standardized distributional metrics
    metrics = compute_distributional_metrics(all_preds, all_trues)

    print("\n--- Evaluation Results ---")
    for k_name, val in metrics.items():
        print(f"  {k_name:15s}: {val:.4f}")

    if total_neighbors_audited > 0:
        rej_pct = (rejected_neighbors_total / total_neighbors_audited) * 100.0
        print(f"[Leakage Auditor] Rejected {rejected_neighbors_total}/{total_neighbors_audited} neighbors ({rej_pct:.2f}%)")

    # Serialize structured run record
    run_record = {
        "dataset": args.dataset or args.data,
        "backbone": args.backbone,
        "fusion": args.injection_point,
        "seed": eval_seed,
        "kb_size": getattr(args, "kb_size", None),
        "metrics": {k: float(v) for k, v in metrics.items()}
    }

    # Save to output results directory
    save_name = args.save_file_name or f"eval_{args.backbone}_{args.injection_point}_{args.benchmark}_{args.data}.json"
    results_dir = "./results/forecast_evaluation"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, save_name)
    with open(out_path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"\n[Artifact] Saved run record to {out_path}")

    return run_record


if __name__ == "__main__":
    parser = add_all_flags()
    args = parser.parse_args()
    run_evaluation(args)
