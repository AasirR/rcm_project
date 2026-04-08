"""
src/evaluation/metrics.py

Shared evaluation utilities used across Notebooks 03, 04, and 05.
Provides a single evaluate() function that returns a consistent metrics dict
for any binary classifier, plus a cost_analysis() function for the business layer.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, brier_score_loss,
)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute a standard set of binary classification metrics.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, brier, threshold,
                    precision, recall, f1, tp, fp, fn, tn,
                    fpr (array), tpr (array), prec (array), rec (array)
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    return {
        'roc_auc':   roc_auc_score(y_true, y_prob),
        'pr_auc':    average_precision_score(y_true, y_prob),
        'brier':     brier_score_loss(y_true, y_prob),
        'threshold': threshold,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'fpr': fpr, 'tpr': tpr, 'prec': prec, 'rec': rec,
    }


def cost_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 300.0,
    fp_cost: float = 25.0,
    n_thresholds: int = 500,
) -> pd.DataFrame:
    """
    Sweep thresholds and compute business cost metrics at each point.

    Returns a DataFrame with columns:
        threshold, tp, fp, fn, tn, cost_fn, cost_fp, total_cost,
        net_savings, precision, recall, f1, flagged
    """
    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    baseline_cost = y_true.sum() * fn_cost  # catch-nothing baseline
    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cost_fn    = fn * fn_cost
        cost_fp    = fp * fp_cost
        total_cost = cost_fn + cost_fp
        net_savings = baseline_cost - total_cost

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

        rows.append({
            'threshold': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'cost_fn': cost_fn, 'cost_fp': cost_fp,
            'total_cost': total_cost, 'net_savings': net_savings,
            'precision': precision, 'recall': recall, 'f1': f1,
            'flagged': tp + fp,
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(cost_df: pd.DataFrame, criterion: str = 'min_cost') -> float:
    """
    Return the threshold that optimises the chosen criterion.
    criterion: 'min_cost' | 'max_savings' | 'max_f1'
    """
    if criterion == 'min_cost':
        return cost_df.loc[cost_df['total_cost'].idxmin(), 'threshold']
    if criterion == 'max_savings':
        return cost_df.loc[cost_df['net_savings'].idxmax(), 'threshold']
    if criterion == 'max_f1':
        return cost_df.loc[cost_df['f1'].idxmax(), 'threshold']
    raise ValueError(f"Unknown criterion: {criterion}")
