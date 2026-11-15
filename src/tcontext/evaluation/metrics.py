"""
Classification Metrics Suite for Multi-Model Evaluation.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None
) -> dict[str, Any]:
    """Calculate comprehensive predictive performance metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision_bin),
        "recall": float(recall_bin),
        "f1": float(f1_bin),
        "macro_precision": float(precision_mac),
        "macro_recall": float(recall_mac),
        "macro_f1": float(f1_mac),
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            elif y_prob.ndim == 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = None
            if auc is not None:
                metrics["roc_auc"] = float(auc)
        except Exception:
            metrics["roc_auc"] = None

    return metrics
