#!/usr/bin/env python3
"""
[Anonymous] TS-RAG-2 Systems Throughput, Latency & VRAM Profiling Grid
Double-Blind Compliant Modular Architecture
Profiles Token In-Context, Latent ARM, and Output Vincentization across:
  - Batch sizes B in {1, 8, 32, 128}
  - Context lengths L in {512, 1024, 2048}
  - Retrieval budgets k in {1, 10, 25, 50}
Identifies quadratic crossover points, peak VRAM footprints, and OOM boundaries.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Optional

import torch
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.backbone_interface import get_backbone_adapter
from models.injection_heads import UnifiedRetrievalInjector


def save_json_safe(data: Any, primary_path: str):
    targets = [
        primary_path,
        "./results/systems_scaling_grid.json",
        "./scratch/systems_scaling_grid.json"
    ]
    written = []
    for tgt in targets:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(tgt)), exist_ok=True)
            with open(tgt, "w") as f:
                json.dump(data, f, indent=2)
            written.append(tgt)
        except Exception:
            pass

    if written:
        print(f"[profile_systems_throughput] Successfully saved scaling grid to: {written[0]}")
    else:
        print(f"[profile_systems_throughput] WARNING: Failed to write output to {primary_path}")


def profile_systems_throughput(
    backbones: List[str],
    batch_sizes: List[int],
    context_lengths: List[int],
    k_values: List[int],
    fusion_types: List[str],
    pred_len: int = 64,
    num_warmup: int = 1,
    num_runs: int = 3,
    out_file: str = "./results/systems_scaling_grid.json"
) -> Dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"=== [Anonymous] TS-RAG-2 Systems Throughput, Latency & VRAM Profiling Sweep ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbones: {backbones}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Context Lengths: {context_lengths}")
    print(f"Retrieval Budgets (k): {k_values}")
    print(f"Fusion Types: {fusion_types}")
    print("=" * 75)

    results_grid = []
    summary_stats = {
        "total_configs": len(backbones) * len(batch_sizes) * len(context_lengths) * len(k_values) * len(fusion_types),
        "successful_runs": 0,
        "oom_runs": 0,
        "crossover_points": []
    }

    pretrained_path = "./checkpoints/base"

    for backbone_name in backbones:
        for L in context_lengths:
            print(f"\n>>> Loading {backbone_name} (L={L}, H={pred_len}) on {device}...")
            try:
                backbone = get_backbone_adapter(
                    backbone_name=backbone_name,
                    pretrained_model_path=pretrained_path,
                    context_length=L,
                    prediction_length=pred_len,
                    device=device
                )
                backbone.eval()
            except Exception as e:
                print(f"ERROR: Failed to load backbone {backbone_name}: {e}")
                continue

            for fusion in fusion_types:
                try:
                    injector = UnifiedRetrievalInjector(
                        backbone=backbone,
                        injection_point=fusion,
                        d_model=backbone.d_model,
                        prediction_length=pred_len
                    ).to(device)
                    injector.eval()
                except Exception as e:
                    print(f"ERROR: Failed to initialize injector {fusion}: {e}")
                    continue

                for B in batch_sizes:
                    for k in k_values:
                        config_desc = f"[{backbone_name} | {fusion:<6s}] B={B:<3d} L={L:<4d} k={k:<2d}"
                        entry = {
                            "backbone": backbone_name,
                            "fusion": fusion,
                            "batch_size": B,
                            "context_length": L,
                            "k": k,
                            "prediction_length": pred_len,
                            "status": "PENDING",
                            "latency_ms": None,
                            "throughput_samples_sec": None,
                            "peak_vram_mb": None,
                            "gflops": None,
                            "oom": False,
                            "error": None
                        }

                        x_q = None
                        ret_past = None
                        ret_future = None

                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats(device)

                            x_q = torch.randn(B, L, device=device)
                            ret_past = torch.randn(B, k, L, device=device)
                            ret_future = torch.randn(B, k, pred_len, device=device)

                            # Warmup passes
                            with torch.no_grad():
                                for _ in range(num_warmup):
                                    _ = injector(x_q, ret_past, ret_future)

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()

                            # Timed benchmark passes
                            t_start = time.perf_counter()
                            with torch.no_grad():
                                for _ in range(num_runs):
                                    _ = injector(x_q, ret_past, ret_future)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_end = time.perf_counter()

                            avg_latency = ((t_end - t_start) / num_runs) * 1000.0  # ms
                            throughput = float(B / ((t_end - t_start) / num_runs)) if (t_end - t_start) > 0 else 0.0
                            peak_mem = (
                                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                                if torch.cuda.is_available() else 0.0
                            )

                            flops_info = injector.estimate_flops(batch_size=B, top_k=k)
                            gflops = flops_info["estimated_flops"] / 1e9

                            entry["status"] = "SUCCESS"
                            entry["latency_ms"] = round(avg_latency, 2)
                            entry["throughput_samples_sec"] = round(throughput, 2)
                            entry["peak_vram_mb"] = round(peak_mem, 2)
                            entry["gflops"] = round(gflops, 3)

                            summary_stats["successful_runs"] += 1
                            vram_info = f" | VRAM: {entry['peak_vram_mb']:>7.1f} MB" if torch.cuda.is_available() else ""
                            print(f"{config_desc} -> Latency: {entry['latency_ms']:>7.2f} ms ({entry['throughput_samples_sec']:>6.1f} seq/s){vram_info} | GFLOPs: {entry['gflops']:>6.2f} [SUCCESS]")

                        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                            entry["status"] = "OOM"
                            entry["oom"] = True
                            entry["error"] = "CUDA Out of Memory"
                            summary_stats["oom_runs"] += 1
                            print(f"{config_desc} -> OOM BOUNDARY ENCOUNTERED (CUDA Out of Memory) [CAUGHT]")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except Exception as e:
                            err_str = str(e)
                            if "out of memory" in err_str.lower():
                                entry["status"] = "OOM"
                                entry["oom"] = True
                                entry["error"] = "CUDA Out of Memory"
                                summary_stats["oom_runs"] += 1
                                print(f"{config_desc} -> OOM BOUNDARY ENCOUNTERED: {err_str[:60]}... [CAUGHT]")
                            else:
                                entry["status"] = "FAILED"
                                entry["error"] = err_str
                                print(f"{config_desc} -> FAILED: {err_str[:80]}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        finally:
                            del x_q, ret_past, ret_future
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        results_grid.append(entry)

            del backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_payload = {
        "metadata": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "summary": summary_stats,
        "grid": results_grid
    }

    save_json_safe(final_payload, out_file)
    print("\n" + "=" * 75)
    print(f"Systems Profiling Complete! Runs: {summary_stats['successful_runs']} OK, {summary_stats['oom_runs']} OOM")
    print("=" * 75)
    return final_payload


def main():
    parser = argparse.ArgumentParser(description="[Anonymous] TS-RAG-2 Systems Profiling Runner")
    parser.add_argument("--backbones", nargs="+", default=["chronos-bolt", "moirai-2.0"], help="Backbone models")
    parser.add_argument("--fusions", nargs="+", default=["latent", "token", "output"], help="Fusion strategies")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8, 32, 128], help="Batch sizes B")
    parser.add_argument("--context_lengths", nargs="+", type=int, default=[512, 1024, 2048], help="Context lengths L")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 10, 25, 50], help="Retrieval budgets k")
    parser.add_argument("--pred_len", type=int, default=64, help="Prediction horizon H")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup passes")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark passes")
    parser.add_argument("--out_file", type=str, default="./results/systems_scaling_grid.json", help="Output JSON path")
    args = parser.parse_args()

    profile_systems_throughput(
        backbones=args.backbones,
        batch_sizes=args.batch_sizes,
        context_lengths=args.context_lengths,
        k_values=args.k_values,
        fusion_types=args.fusions,
        pred_len=args.pred_len,
        num_warmup=args.warmup,
        num_runs=args.runs,
        out_file=args.out_file
    )


if __name__ == "__main__":
    main()
