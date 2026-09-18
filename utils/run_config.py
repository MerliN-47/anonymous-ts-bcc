"""
[Anonymous] TS-RAG / CAVIR Configuration & CLI Schema
Double-Blind Compliant Modular Architecture
"""

import argparse


def add_all_flags(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """
    Standardized CLI flags for TS-RAG / CAVIR baseline, ablations, and multimodal extensions.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="[Anonymous] TS-RAG / CAVIR Foundation Model Retrieval-Augmentation Suite")

    # Model & Backbone configuration
    parser.add_argument("--model_id", type=str, default="ChronosBoltRetrieve", help="Model run identifier")
    parser.add_argument("--model", type=str, default="ChronosBoltRetrieve", help="Model family name")
    parser.add_argument(
        "--backbone",
        type=str,
        default="chronos-bolt",
        choices=["chronos-bolt", "chronos-2", "moirai-2.0", "timesfm-2.5", "tirex"],
        help="Base time series foundation model backbone"
    )
    parser.add_argument(
        "--injection_point",
        type=str,
        default="latent",
        choices=["latent", "token", "output", "none"],
        help="Retrieval injection location: token (in-context), latent (ARM), output (quantile barycenter)"
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Directory to save/load checkpoints")
    parser.add_argument("--pretrained_model_path", type=str, default="./checkpoints/base", help="Path to base pretrained weights")
    parser.add_argument("--checkpoint_model_path", type=str, default="None", help="Path to fine-tuned or head checkpoint")

    # Sequence lengths
    parser.add_argument("--seq_len", type=int, default=512, help="Context sequence length (L)")
    parser.add_argument("--context_length", type=int, default=512, help="Alias for seq_len")
    parser.add_argument("--pred_len", type=int, default=64, help="Prediction horizon length (H)")
    parser.add_argument("--prediction_length", type=int, default=64, help="Alias for pred_len")
    parser.add_argument("--label_len", type=int, default=48, help="Start token length for Informer/Autoformer style models")
    parser.add_argument("--lookback_length", type=int, default=512, help="Retrieval query lookback window")
    parser.add_argument("--retrieve_lookback_length", type=int, default=512, help="Alias for retrieval lookback length")

    # Retrieval parameters
    parser.add_argument("--top_k", type=int, default=10, help="Retrieval budget: number of retrieved neighbors (k)")
    parser.add_argument("--retrieval_database_dir", type=str, default="./retrieval_database/", help="Directory storing retrieval database")
    parser.add_argument("--retrieval_database_path", type=str, default="./retrieval_database/", help="Path to retrieval database")
    parser.add_argument("--dimension", type=int, default=768, help="Embedding representation dimension")
    parser.add_argument("--embedding_model_type", type=str, default="chronos", help="Embedding model backbone")
    parser.add_argument("--embedding_tuning", type=str, default=None, help="Embedding tuning regime")
    parser.add_argument("--mode", type=str, default="only_self_train", help="Retrieval indexing mode")
    parser.add_argument(
        "--index_type",
        type=str,
        default="flat",
        choices=["flat", "ivf_pq", "hnsw"],
        help="FAISS index backend"
    )

    # KB Scaling & Regime
    parser.add_argument("--kb_size_exp", type=int, default=5, help="Exponent for KB size (e.g., 5 for 10^5)")
    parser.add_argument("--kb_size", type=int, default=None, help="Direct integer KB sequence count (e.g., 10000)")
    parser.add_argument("--kb_seed", type=int, default=2021, help="Seed for KB subsampling/splitting")
    parser.add_argument(
        "--kb_regime",
        type=str,
        default="retrieval_shot",
        choices=["strict_zeroshot", "retrieval_shot", "cross_domain", "poisoned"],
        help="Data split & purity regime"
    )
    parser.add_argument("--kb_fraction", type=float, default=1.0, help="Fraction of KB retained")
    parser.add_argument("--poison_ratio", type=float, default=0.0, help="Adversarial noise/poison ratio")

    # Covariate & Multivariate flags
    parser.add_argument(
        "--covariate_mode",
        type=str,
        default="none",
        choices=["none", "future_known", "static_meta", "full_covariate", "cross_channel"],
        help="Conditioning mode for retrieval query & fusion"
    )
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="channel_independent",
        choices=["channel_independent", "channel_block", "group_attention"],
        help="Multivariate channel indexing scheme"
    )

    # Multimodal Text flags
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="none",
        choices=["none", "bge-large-en", "mpnet-base"],
        help="Text embedding model for multimodal event streams"
    )
    parser.add_argument(
        "--text_role",
        type=str,
        default="none",
        choices=["none", "retrieval_only", "fusion_only", "dual"],
        help="Isolates retrieval vs fusion contribution of text"
    )

    # Fusion architecture flags
    parser.add_argument("--augment_mode", type=str, default="moe", help="Fusion mode: moe, gate, none")
    parser.add_argument("--align", type=str, default="none", choices=["none", "affine", "affine_conditioned"])
    parser.add_argument("--gate", type=str, default="softmax", choices=["softmax", "abstain", "none"])
    parser.add_argument("--fusion", type=str, default="latent", choices=["latent", "quantile", "token", "output", "none"])
    parser.add_argument("--gate_distance", type=str, default="none", choices=["none", "raw", "log", "kernel"])
    parser.add_argument("--loss", type=str, default="pinball", choices=["pinball", "point", "wql", "crps"])

    # Representation Invariance & Leakage Filtering
    parser.add_argument("--invariance", type=str, default="full", choices=["full", "partial"], help="Representation scale-invariance")
    parser.add_argument("--leakage_filter", type=str, default="none", choices=["none", "active", "cosine"], help="Online leakage filtering")
    parser.add_argument("--leakage_threshold", type=float, default=0.99, help="Cosine similarity threshold for leakage filtering")
    parser.add_argument("--dtw_threshold", type=float, default=1e-3, help="DTW distance threshold for sequence duplication")

    # Conformal Prediction & Uncertainty
    parser.add_argument("--conformal", "--conformal_guard", dest="conformal", type=str, default="none", choices=["none", "active"], help="Enable Conformal Guard")
    parser.add_argument("--cf_shift", type=str, default="none", choices=["none", "active"], help="Enable counterfactual / variance shift test")

    # Evaluation & Benchmark
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ett",
        choices=["ett", "gift_eval", "fev_bench", "context_is_key"],
        help="Evaluation benchmark dataset suite"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset alias (etth1, etth2, ettm1, ettm2, weather, traffic, etc.)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for evaluation sweeps")
    parser.add_argument(
        "--eval_metrics",
        type=str,
        default="mse_mae",
        choices=["mse_mae", "distributional"],
        help="Output evaluation metrics: mse_mae or distributional"
    )
    parser.add_argument("--report_distributional", action="store_true", default=False, help="Report CRPS/WQL/Coverage metrics")

    # Dataset file specifications
    parser.add_argument("--root_path", type=str, default="./datasets/ETT-small/", help="Root directory of dataset files")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="Data filename")
    parser.add_argument("--data", type=str, default="ETTh1", help="Dataset identifier")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"], help="Forecasting task: M, S, or MS")
    parser.add_argument("--target", type=str, default="OT", help="Target variable name in S or MS task")
    parser.add_argument("--freq", type=str, default="h", help="Frequency for time features encoding (s, t, h, d, b, w, m)")

    # Training & Execution
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training/testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader worker count")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Optimizer learning rate")
    parser.add_argument("--train_epochs", type=int, default=10, help="Training epoch count")
    parser.add_argument("--train_steps", type=int, default=10000, help="Max optimization steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay parameter")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--gpu_loc", type=int, default=0, help="GPU device ID")
    parser.add_argument("--use_gpu", type=bool, default=True, help="Enable GPU execution if available")
    parser.add_argument("--save_file_name", type=str, default=None, help="Filename for evaluation results")

    # Smoke test flags
    parser.add_argument("--smoke", action="store_true", default=False, help="Run quick smoke verification test")
    parser.add_argument("--smoke_batches", type=int, default=20, help="Number of batches in smoke mode")

    # Legacy & compatibility arguments
    parser.add_argument("--decay_fac", type=float, default=0.5)
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--tmax", type=int, default=20)
    parser.add_argument("--cos", type=int, default=1)
    parser.add_argument("--metadata_frequency", type=str, default="hour")
    parser.add_argument("--metadata_database_name", type=str, default="ETTh1")

    return parser
