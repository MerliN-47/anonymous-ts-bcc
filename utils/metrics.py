"""
[Anonymous] TS-RAG / CAVIR Evaluation Metrics Suite
Double-Blind Compliant Modular Architecture
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union


def RSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.sum((true - pred) ** 2)) / (np.sqrt(np.sum((true - true.mean()) ** 2)) + 1e-8))


def CORR(pred: np.ndarray, true: np.ndarray) -> float:
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return float((u / (d + 1e-8)).mean(-1))


def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(100 * (pred - true) / (true + 1e-8))))


def MSPE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.square((pred - true) / (true + 1e-8))))


def SMAPE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)))


def ND(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)) / (np.mean(np.abs(true)) + 1e-8))


def pinball_loss(
    quantile_preds: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> np.ndarray:
    """
    Computes per-quantile pinball loss:
    L_tau(y, q) = 2 * (tau * (y - q) * I(y >= q) + (1 - tau) * (q - y) * I(y < q))
    """
    if isinstance(quantile_preds, torch.Tensor):
        quantile_preds = quantile_preds.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    if true.ndim == 2 and quantile_preds.ndim == 3:
        true = np.expand_dims(true, 1)

    losses = []
    for i, tau in enumerate(quantiles):
        q = quantile_preds[:, i:i+1, :] if quantile_preds.ndim == 3 else quantile_preds[..., i:i+1]
        err = true - q
        loss = 2.0 * np.maximum(tau * err, (tau - 1.0) * err)
        losses.append(float(np.mean(loss)))
    return np.array(losses)


def CRPS(
    quantile_preds: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> float:
    """
    Continuous Ranked Probability Score via numerical quantile integration.
    """
    ploss = pinball_loss(quantile_preds, true, quantiles)
    return float(np.mean(ploss))


def WQL(
    quantile_preds: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> float:
    """
    Weighted Quantile Loss normalized by total sum of absolute true targets.
    """
    if isinstance(quantile_preds, torch.Tensor):
        quantile_preds = quantile_preds.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    if true.ndim == 2 and quantile_preds.ndim == 3:
        true_exp = np.expand_dims(true, 1)
    else:
        true_exp = true

    num_sum = 0.0
    den_sum = np.sum(np.abs(true)) + 1e-8

    for i, tau in enumerate(quantiles):
        q = quantile_preds[:, i:i+1, :] if quantile_preds.ndim == 3 else quantile_preds[..., i:i+1]
        err = true_exp - q
        loss = 2.0 * np.maximum(tau * err, (tau - 1.0) * err)
        num_sum += np.sum(loss)

    return float(num_sum / (len(quantiles) * den_sum))


def MASE(
    pred: np.ndarray,
    true: np.ndarray,
    train_history: Optional[np.ndarray] = None,
    seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).
    """
    mae_model = np.mean(np.abs(pred - true))
    if train_history is not None and len(train_history) > seasonality:
        scale = np.mean(np.abs(train_history[seasonality:] - train_history[:-seasonality]))
    else:
        scale = np.mean(np.abs(true[:, 1:] - true[:, :-1])) if true.shape[-1] > 1 else 1.0
    return float(mae_model / (scale + 1e-8))


def Coverage(
    quantile_preds: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.8,
    quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> float:
    """
    Computes empirical coverage of central alpha-prediction interval.
    For alpha=0.8, verifies whether true target lies between tau=0.1 and tau=0.9.
    """
    if isinstance(quantile_preds, torch.Tensor):
        quantile_preds = quantile_preds.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    lower_tau = round((1.0 - alpha) / 2.0, 4)
    upper_tau = round(1.0 - lower_tau, 4)

    if lower_tau in quantiles and upper_tau in quantiles:
        low_idx = quantiles.index(lower_tau)
        high_idx = quantiles.index(upper_tau)
        low_q = quantile_preds[:, low_idx, :]
        high_q = quantile_preds[:, high_idx, :]
        covered = (true >= low_q) & (true <= high_q)
        return float(np.mean(covered.astype(float)))
    else:
        low_q = np.quantile(quantile_preds, lower_tau, axis=1)
        high_q = np.quantile(quantile_preds, upper_tau, axis=1)
        covered = (true >= low_q) & (true <= high_q)
        return float(np.mean(covered.astype(float)))


def DieboldMarianoTest(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for predictive accuracy equality.
    Returns: (DM statistic, two-sided p-value)
    """
    from scipy import stats
    d = errors_1.flatten() - errors_2.flatten()
    n = len(d)
    if n < 2:
        return 0.0, 1.0
    mean_d = np.mean(d)

    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        gamma_sum += 2.0 * gamma_k

    var_d = (gamma0 + gamma_sum) / n
    if var_d <= 0:
        return 0.0, 1.0
    dm_stat = float(mean_d / np.sqrt(var_d))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_value


def metric(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    nd = ND(pred, true)
    return mae, mse, rmse, mape, mspe, smape, nd


def compute_distributional_metrics(
    quantile_preds: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    train_history: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive metric computation across point and distributional forecasts.
    """
    if isinstance(quantile_preds, torch.Tensor):
        quantile_preds = quantile_preds.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    # Median prediction (tau=0.5 or center)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else quantile_preds.shape[1] // 2
    point_pred = quantile_preds[:, median_idx, :]

    mae, mse, rmse, mape, mspe, smape, nd = metric(point_pred, true)
    crps = CRPS(quantile_preds, true, quantiles)
    wql = WQL(quantile_preds, true, quantiles)
    mase = MASE(point_pred, true, train_history)
    cov80 = Coverage(quantile_preds, true, alpha=0.8, quantiles=quantiles)
    cov60 = Coverage(quantile_preds, true, alpha=0.6, quantiles=quantiles)
    cov40 = Coverage(quantile_preds, true, alpha=0.4, quantiles=quantiles)
    cov20 = Coverage(quantile_preds, true, alpha=0.2, quantiles=quantiles)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "SMAPE": smape,
        "ND": nd,
        "CRPS": crps,
        "WQL": wql,
        "MASE": mase,
        "Coverage@80": cov80,
        "Coverage@60": cov60,
        "Coverage@40": cov40,
        "Coverage@20": cov20,
    }
