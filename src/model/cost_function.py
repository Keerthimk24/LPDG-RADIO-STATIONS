"""
Custom cost function for LightGBM — asymmetric loss that penalises
false negatives (missing a broken gateway) more than false positives
(unnecessary visit).

Cost structure:
  - False Positive (visit healthy gateway): €380 wasted
  - False Negative (miss broken gateway): €600 per week compounding

The ratio 600/380 ≈ 1.58 — FN is ~1.6x more costly than FP.
"""

from __future__ import annotations

import numpy as np


def asymmetric_cost_objective(y_pred: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
    """Custom objective function for LightGBM.

    Penalises under-prediction (false negatives) at 600/380 ≈ 1.58x the rate
    of over-prediction (false positives).

    Args:
        y_pred: Current model predictions.
        dtrain: LightGBM Dataset with true labels.

    Returns:
        Tuple of (gradient, hessian).
    """
    y_true = dtrain.get_label()
    residual = y_true - y_pred

    # Asymmetric weights
    cost_fn = 600.0  # False negative cost
    cost_fp = 380.0  # False positive cost

    # Gradient: steeper for under-prediction
    grad = np.where(
        residual > 0,
        -cost_fn * residual,   # Under-predicting (FN direction)
        -cost_fp * residual,   # Over-predicting (FP direction)
    )

    # Hessian (constant per side)
    hess = np.where(
        residual > 0,
        cost_fn * np.ones_like(residual),
        cost_fp * np.ones_like(residual),
    )

    return grad, hess


def cost_metric(y_pred: np.ndarray, dtrain) -> tuple[str, float, bool]:
    """Custom evaluation metric: total cost in euros.

    For use as eval_metric in LightGBM. Lower is better.
    """
    y_true = dtrain.get_label()

    # Convert continuous prediction to binary (threshold at 0.5)
    y_binary = (y_pred > 0.5).astype(int)

    # False positives: predicted broken but actually fine
    fp = ((y_binary == 1) & (y_true == 0)).sum()

    # False negatives: predicted fine but actually broken
    fn = ((y_binary == 0) & (y_true == 1)).sum()

    total_cost = fp * 380.0 + fn * 600.0

    return "total_cost_eur", total_cost, False  # False = lower is better


def compute_visit_value(probability: float, cost_fp: float = 380.0, cost_fn: float = 600.0) -> float:
    """Compute the expected value of visiting a gateway.

    Visit_value = P(broken) × cost_fn - cost_fp
    Positive value means visiting is worthwhile.

    Args:
        probability: Estimated probability the gateway is broken.
        cost_fp: Cost of an unnecessary visit.
        cost_fn: Cost of leaving a broken gateway for one week.

    Returns:
        Expected net cost savings from visiting.
    """
    return probability * cost_fn - (1 - probability) * cost_fp


def rank_by_cost(
    gateway_ids: np.ndarray,
    probabilities: np.ndarray,
    budget: int = 15,
    cost_fp: float = 380.0,
    cost_fn: float = 600.0,
) -> list[dict]:
    """Rank gateways by expected cost savings and select top-N.

    Args:
        gateway_ids: Array of gateway IDs.
        probabilities: Predicted probabilities of being broken.
        budget: Maximum visits per week.
        cost_fp: Cost of unnecessary visit.
        cost_fn: Cost of missed broken gateway per week.

    Returns:
        List of dicts with gateway_id, rank, score, probability.
    """
    values = np.array([compute_visit_value(p, cost_fp, cost_fn) for p in probabilities])

    # Sort by expected value, descending (highest value = most worth visiting)
    indices = np.argsort(values)[::-1]

    ranked = []
    for rank, idx in enumerate(indices[:budget], 1):
        ranked.append({
            "gateway_id": gateway_ids[idx],
            "rank": rank,
            "score": float(values[idx]),
            "probability": float(probabilities[idx]),
        })

    return ranked
