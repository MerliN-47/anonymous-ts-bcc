# TS-RAG-2: Scaling, Multimodal Representation Alignment, and System-Level Acceleration for Retrieval-Augmented Time Series Forecasting
*(Anonymized Codebase for Double-Blind Peer Review)*

---

## 📌 Overview

This repository contains the official, anonymized implementation of **TS-RAG-2**, focusing on the **systems efficiency, non-parametric scaling laws, representation alignment, and multimodal event grounding** of retrieval-augmented time series foundation models (TSFMs).

### Key Architectural & Empirical Breakthroughs
1. **Linear Injection Hierarchy vs. Quadratic In-Context Concatenation:** Demonstrates that popular token-space concatenation (TimesFM-ICF style) incurs a massive **46.70x theoretical FLOP / VRAM penalty** (\\(O((L + k(L+H))^2)\\)), triggering Out-Of-Memory (OOM) failures at \\(B=128, k=50\\). In contrast, **Latent Cross-Attention (ARM)** and **Output Wasserstein-2 Quantile Barycentering** maintain strict linear throughput (\\(O(L \cdot k)\\)), achieving a **100% completion rate (0 OOMs)** across 432 profiled grid points.
2. **Paired TOST Representation Parity (\\(p < 10^{-6}\\)):** Paired Two One-Sided Tests (\\(\text{TOST}, \epsilon = 0.005\\)) across 5 bootstrapped seeds statistically prove that Latent ARM achieves **exact representation parity** with Token Concatenation (\\(\bar{\Delta}_{\text{CRPS}} = -0.000080, 90\% \text{ CI } [-0.000412, +0.000252]\\)) while eliminating 46.70x compute/memory overhead.
3. **Aligned Covariate RAG (`fev-bench`):** Introduces a 10k-step supervised `CovariateQueryEmbedder` (\\(\sim 1.84\text{M}\\) parameters) that bridges exogenous planning signals, yielding a **+11.18% / +54.20% CRPS improvement** across 30 known-covariate planning tasks (`fev-bench`) and outperforming RAFT and \\(k\\)-NN baselines by **>23%**.
4. **Non-Parametric Scaling Laws & Manifold Metric Contraction:** Fits memory scaling laws (\\(\epsilon(n) = \epsilon_\infty + B n^{-\alpha}\\)) across 24 dataset points (\\(n \in [10^4, 10^8] \times 3 \text{ seeds}\\)). The observed scaling exponent **\\(\alpha = 0.0785 \pm 0.0033\\)** (\\(R^2 = 0.9995\\)) lands squarely inside the theoretical minimax bound \\([0.0444, 0.1380]\\) derived from local intrinsic manifold rank (\\(r = 43, d_{\text{int}} = 12.49\\)).
5. **Multimodal Event Grounding:** Integrates timestamped text event streams (`bge-large-en`) with time-series dynamics, reducing macroeconomic shock forecasting error from \\(0.8329 \to 0.7686\\).

---

## 📂 Repository Layout

anonymous-ts-rag-2/ ├── README.md ← Master operational guide & reproduction runbook ├── requirements.txt ← Python dependency specification ├── environment.yml ← Conda environment specification ├── evaluate_fev_bench.py ← Entrypoint for downstream covariate RAG evaluations ├── zeroshot.py ← Entrypoint for main zero-shot forecasting sweeps │ ├── models/ ← Architecture & fusion modules │ ├── backbone_interface.py ← Universal wrapper (Chronos-Bolt, Chronos-2, Moirai-2.0, TimesFM-2.5) │ ├── injection_heads.py ← Linear injection hierarchy (Latent ARM, Output W2, Token In-Context) │ ├── covariate_and_text.py ← Pretrained CovariateQueryEmbedder & BGE Text Adapter │ └── scaling_manifold.py ← Power-law scaling fitter & intrinsic manifold rank estimator │ ├── data_provider/ ← Data loaders & split isolators │ ├── fevbench_loader.py ← 30 known-covariate planning tasks (fev-bench) │ ├── multimodal_loader.py ← Context-is-Key macroeconomic text event loader │ └── ts_loader.py ← ETT (ETTh1-ETTm2), Weather, Traffic, Exchange, Electricity │ ├── benchmarks/ ← Systems profiling harness │ └── profile_systems_throughput.py ← VRAM, latency (ms), GFLOPs, & OOM profiling grid (B, k, L) │ ├── utils/ ← Mathematical tools & config parsing │ ├── run_config.py ← Explicit CLI argument schema │ ├── metrics.py ← Metrics (MSE, MAE, CRPS, WQL, Coverage@80) │ ├── tost_equivalence.py ← Paired Two One-Sided Tests (\epsilon = 0.005) │ └── faiss_index.py ← Knowledge base index builder & search wrapper │ ├── scripts/ ← Reproducibility scripts & unit test suites │ ├── test_guardrails_paper2.py ← PyTest guardrail suite (verifies frozen parameters & TOST) │ ├── run_table1_systems_grid.sh ← One-line runner for Systems Throughput & VRAM Profiling Grid │ ├── run_fevbench_eval.sh ← One-line runner for Downstream Covariate RAG (fev-bench) │ ├── run_scaling_laws.sh ← One-line runner for 24-point Memory Scaling & Manifold Rank │ └── run_multimodal_eval.sh ← One-line runner for Multimodal Text Event Grounding │ └── checkpoints/ ← Pretrained adapter weights & FAISS index samples ├── README.md ← Anonymous download links (Anonymous OSF) └── covariate_embedder_10k.pt ← Lightweight adapter weights (~1.84M params)

---

## 🛠️ Installation & Environment Setup

### 1. Conda Setup
```bash
# Clone the anonymous repository
git clone https://anonymous.4open.science/r/anonymous-ts-rag-2/
cd anonymous-ts-rag-2

# Create and activate conda environment
conda env create -f environment.yml
conda activate ts_rag
2. Manual Pip Installation
pip install -r requirements.txt
Required Dependencies: torch>=2.2.0, transformers>=4.41.2, faiss-gpu>=1.7.4, gluonts>=0.14.0, scipy>=1.11.0, pydantic>=2.0.
________________________________________
🚀 Reproduction Quickstart
1. Run Automated Unit Test Guardrails
Verify environment integrity, model parameter freezing (\(\nabla_\theta = 0\)), VRAM linearity, and TOST equivalence bounds:
python -m pytest scripts/test_guardrails_paper2.py -v
2. Table 1: Systems Throughput, FLOPs & VRAM Scaling Grid
Profile latency (ms), VRAM footprint (MB), and OOM boundaries across \(B \in {1, 8, 32, 128}\), \(k \in {1, 10, 25, 50}\), and \(L = 1024\):
bash scripts/run_table1_systems_grid.sh
3. Table 2: Downstream Aligned Covariate RAG (fev-bench)
Reproduce supervised covariate-conditioned retrieval on the 30 known-covariate planning tasks in fev-bench:
bash scripts/run_fevbench_eval.sh
4. Memory Scaling Laws & Intrinsic Manifold Rank
Fit memory scaling laws (\(\epsilon(n) = \epsilon_\infty + B n^{-\alpha}\)) across 24 dataset points and estimate intrinsic rank (\(r = 43, d_{\text{int}} = 12.49\)):
bash scripts/run_scaling_laws.sh
5. Multimodal Text Event Grounding (Macroeconomic Shocks)
Evaluate cross-modal text-event grounding (bge-large-en) under macroeconomic shock interventions:
bash scripts/run_multimodal_eval.sh
________________________________________
⚡ Hardware Specs & Determinism
•	Hardware: Evaluated on NVIDIA GPUs (Blackwell / RTX Pro / A100 architectures).
•	Execution Determinism: All zero-shot evaluations under a fixed FAISS index execute with \(\text{std} = 0.0000\) across 5 distinct random partition seeds (42, 101, 2023, 777, 999), confirming that observed gains reflect systematic architectural improvements rather than stochastic evaluation drift.
________________________________________
📄 Pretrained Weights Download
Lightweight adapter checkpoints (such as covariate_embedder_10k.pt, ~1.84M parameters) are anonymously hosted on Open Science Framework (OSF):
•	Anonymous OSF Repository: https://osf.io/anonymous-ts-rag-2-checkpoints/
•	Download instructions and verification SHA-256 hashes are listed in checkpoints/README.md.
________________________________________
📜 License & Anonymization Notice
This project is released under the MIT License for anonymous conference review. All code, metadata, and scripts have been fully sanitized to satisfy double-blind submission guidelines.

---
