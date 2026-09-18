"""
[Anonymous] TS-RAG-2 / CAVIR Scaling Laws & Manifold Geometry
Double-Blind Compliant Modular Architecture
Implements:
1. Non-parametric power-law scaling law fitter: epsilon(n) = eps_inf + B * n^(-alpha)
2. Local intrinsic manifold dimension estimation (Two-NN, Facco et al., 2017)
3. Global effective manifold rank (PCA 95% variance explained)
4. Theoretical convergence rate comparison: alpha <= 2 / (2 + d_int)
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit


def power_law_func(n: np.ndarray, eps_inf: float, B: float, alpha: float) -> np.ndarray:
    """
    Non-parametric retrieval power-law scaling function:
    epsilon(n) = eps_inf + B * n^(-alpha)
    """
    return eps_inf + B * np.power(n, -alpha)


class ManifoldRankEstimator:
    """
    Estimator for intrinsic data manifold dimension and effective rank.
    """
    @staticmethod
    def estimate_two_nn(data: np.ndarray) -> float:
        """
        Two-NN local intrinsic dimension estimator (Facco et al., 2017).
        Fits empirical ratio of second-to-first nearest neighbor distances.
        """
        from sklearn.neighbors import NearestNeighbors
        data = np.asarray(data, dtype=np.float64)
        n_samples = min(2000, len(data))
        if n_samples < 10:
            return 1.0

        if len(data) > n_samples:
            indices = np.random.choice(len(data), size=n_samples, replace=False)
            sub_data = data[indices]
        else:
            sub_data = data

        nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(sub_data)
        distances, _ = nbrs.kneighbors(sub_data)

        r1 = distances[:, 1]
        r2 = distances[:, 2]

        valid = (r1 > 1e-8) & (r2 > 1e-8)
        if np.sum(valid) < 5:
            return 1.0

        mu = r2[valid] / r1[valid]
        mu = np.sort(mu)
        f = np.arange(1, len(mu) + 1) / float(len(mu))

        # Linear fit of -log(1 - f) vs log(mu) through origin
        x = np.log(mu)
        y = -np.log(1.0 - f + 1e-12)

        d_int = float(np.sum(x * y) / (np.sum(x ** 2) + 1e-8))
        return float(max(1.0, d_int))

    @staticmethod
    def estimate_pca_rank(data: np.ndarray, variance_threshold: float = 0.95) -> int:
        """
        Computes the effective rank r: the minimal number of principal components
        explaining variance_threshold (default 95%) of empirical variance.
        """
        data = np.asarray(data, dtype=np.float64)
        centered = data - np.mean(data, axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var_exp = np.cumsum(s ** 2) / (np.sum(s ** 2) + 1e-12)
        rank = int(np.searchsorted(var_exp, variance_threshold) + 1)
        return max(1, rank)

    @staticmethod
    def compute_theoretical_bound(dimension: float) -> float:
        """
        Theoretical non-parametric convergence rate bound:
        alpha^* = 2 / (2 + d_int)
        """
        return float(2.0 / (2.0 + dimension))


class ScalingLawFitter:
    """
    Fits empirical scaling laws across memory size n:
    epsilon(n) = eps_inf + B * n^(-alpha)
    """
    def __init__(self, metric_key: str = "CRPS"):
        self.metric_key = metric_key

    def fit(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fits power law across empirical evaluation points.
        Each point must have 'n_kb' (or 'kb_size') and the metric key (e.g. 'CRPS' or 'MSE').
        """
        n_vals = []
        err_vals = []
        seeds = []

        for p in data_points:
            n = p.get("n_kb", p.get("kb_size"))
            err = p.get(self.metric_key)
            if n is not None and err is not None:
                n_vals.append(float(n))
                err_vals.append(float(err))
                seeds.append(p.get("seed", 2021))

        n_vals = np.array(n_vals, dtype=np.float64)
        err_vals = np.array(err_vals, dtype=np.float64)
        seeds = np.array(seeds)

        if len(n_vals) < 4:
            raise ValueError(f"Insufficient data points ({len(n_vals)}) to fit 3-parameter power law.")

        unique_seeds = sorted(list(set(seeds)))

        # Parameter initialization
        p0 = [min(err_vals) * 0.9, (max(err_vals) - min(err_vals)) * 10.0, 0.15]
        bounds = ([0.0, 0.0, 0.001], [np.inf, np.inf, 2.0])

        # Global fit across all empirical points
        popt, pcov = curve_fit(power_law_func, n_vals, err_vals, p0=p0, bounds=bounds, maxfev=10000)
        eps_inf_fit, B_fit, alpha_fit = popt
        perr = np.sqrt(np.diag(pcov))
        eps_inf_se, B_se, alpha_se = perr

        # Goodness-of-fit R^2
        err_pred = power_law_func(n_vals, *popt)
        ss_res = np.sum((err_vals - err_pred) ** 2)
        ss_tot = np.sum((err_vals - np.mean(err_vals)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

        # Per-seed bootstrap statistics
        per_seed_alphas = []
        per_seed_fits = {}
        for s in unique_seeds:
            mask = (seeds == s)
            s_n = n_vals[mask]
            s_err = err_vals[mask]
            if len(s_n) >= 4:
                try:
                    s_popt, _ = curve_fit(power_law_func, s_n, s_err, p0=p0, bounds=bounds, maxfev=10000)
                    per_seed_alphas.append(float(s_popt[2]))
                    per_seed_fits[str(s)] = {
                        "eps_inf": round(float(s_popt[0]), 4),
                        "B": round(float(s_popt[1]), 4),
                        "alpha": round(float(s_popt[2]), 4)
                    }
                except Exception:
                    pass

        mean_seed_alpha = float(np.mean(per_seed_alphas)) if per_seed_alphas else float(alpha_fit)
        std_seed_alpha = float(np.std(per_seed_alphas, ddof=1)) if len(per_seed_alphas) > 1 else float(alpha_se)
        se_seed_alpha = std_seed_alpha / np.sqrt(len(per_seed_alphas)) if per_seed_alphas else float(alpha_se)

        return {
            "metric": self.metric_key,
            "n_points": len(n_vals),
            "global_fit": {
                "eps_inf": round(float(eps_inf_fit), 4),
                "eps_inf_se": round(float(eps_inf_se), 4),
                "B": round(float(B_fit), 4),
                "B_se": round(float(B_se), 4),
                "alpha": round(float(alpha_fit), 4),
                "alpha_se": round(float(alpha_se), 4),
                "R2": round(r2, 4)
            },
            "per_seed_fits": per_seed_fits,
            "bootstrap_statistics": {
                "mean_alpha": round(mean_seed_alpha, 4),
                "std_alpha": round(std_seed_alpha, 4),
                "se_alpha": round(se_seed_alpha, 4),
                "confidence_interval_95": [
                    round(mean_seed_alpha - 1.96 * se_seed_alpha, 4),
                    round(mean_seed_alpha + 1.96 * se_seed_alpha, 4)
                ]
            }
        }


def fit_scaling_power_law(data_points: List[Dict[str, Any]], metric_key: str = "CRPS") -> Dict[str, Any]:
    fitter = ScalingLawFitter(metric_key=metric_key)
    return fitter.fit(data_points)


def estimate_two_nn_dimension(data: np.ndarray) -> float:
    return ManifoldRankEstimator.estimate_two_nn(data)


def estimate_pca_effective_rank(data: np.ndarray, variance_threshold: float = 0.95) -> int:
    return ManifoldRankEstimator.estimate_pca_rank(data, variance_threshold=variance_threshold)
