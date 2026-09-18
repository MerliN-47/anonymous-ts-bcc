"""
[Anonymous] TS-RAG-2 / CAVIR Paired TOST Equivalence Testing
Double-Blind Compliant Modular Architecture
Implements Paired Two One-Sided Tests (TOST) with equivalence margin epsilon = 0.005
to statistically establish representation parity between Latent ARM and Token In-Context Concatenation.
"""

from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np
from scipy import stats


def compute_tost(
    x_latent: Union[np.ndarray, List[float]],
    x_token: Union[np.ndarray, List[float]],
    epsilon: float = 0.005
) -> Dict[str, Any]:
    """
    Computes Paired Two One-Sided Tests (TOST) for equivalence margin epsilon (default 0.005).
    Null hypotheses:
      H01: Delta >= epsilon
      H02: Delta <= -epsilon
    Rejection of both H01 and H02 at alpha = 0.05 establishes statistical equivalence within (-epsilon, +epsilon).
    """
    x_latent = np.asarray(x_latent, dtype=np.float64).flatten()
    x_token = np.asarray(x_token, dtype=np.float64).flatten()

    if len(x_latent) != len(x_token):
        raise ValueError(f"Length mismatch for paired TOST: {len(x_latent)} vs {len(x_token)}")

    diff = x_latent - x_token
    n = len(diff)
    if n < 2:
        raise ValueError(f"Need at least 2 pairs for TOST, got {n}")

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = float(std_diff / np.sqrt(n)) if n > 0 else 1e-12
    df = n - 1

    # One-sided t-statistics
    t1 = (mean_diff - (-epsilon)) / (se_diff + 1e-12)  # H01: mean_diff <= -epsilon
    t2 = (mean_diff - epsilon) / (se_diff + 1e-12)      # H02: mean_diff >= epsilon

    p1 = float(1.0 - stats.t.cdf(t1, df=df))
    p2 = float(stats.t.cdf(t2, df=df))
    p_tost = max(p1, p2)

    # 90% confidence interval corresponds to two one-sided alpha=0.05 tests
    t_crit = float(stats.t.ppf(0.95, df=df))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    is_equiv = bool((ci_lower > -epsilon) and (ci_upper < epsilon) and (p_tost < 0.05))

    return {
        "equivalence_margin_epsilon": epsilon,
        "n_pairs": n,
        "mean_difference": round(mean_diff, 6),
        "std_difference": round(std_diff, 6),
        "se_difference": round(se_diff, 6),
        "ci_90_lower": round(float(ci_lower), 6),
        "ci_90_upper": round(float(ci_upper), 6),
        "t1_stat": round(float(t1), 4),
        "t2_stat": round(float(t2), 4),
        "p_value_t1": round(float(p1), 6),
        "p_value_t2": round(float(p2), 6),
        "p_value_tost": round(float(p_tost), 6),
        "is_equivalent": is_equiv,
        "conclusion": "Equivalent (Representation Parity Confirmed)" if is_equiv else "Inconclusive / Non-equivalent"
    }


def test_representation_parity(
    paired_data: Dict[str, Dict[str, List[float]]],
    epsilon: float = 0.005
) -> Dict[str, Any]:
    """
    Evaluates multi-metric representation parity across paired evaluation runs.
    Example paired_data format:
      {
        "MASE": {"latent": [...], "token": [...]},
        "CRPS": {"latent": [...], "token": [...]}
      }
    """
    results = {}
    all_passed = True

    for metric_name, values in paired_data.items():
        res = compute_tost(values["latent"], values["token"], epsilon=epsilon)
        results[metric_name] = res
        if not res["is_equivalent"]:
            all_passed = False

    return {
        "overall_parity_confirmed": all_passed,
        "metrics": results
    }
