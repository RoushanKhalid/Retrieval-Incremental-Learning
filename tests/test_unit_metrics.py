"""
Unit tests for evaluation metrics suite: Accuracy, Precision, Recall, Macro F1, Binary F1, ROC-AUC, and Confusion Matrix.
"""

import numpy as np

from tcontext.evaluation.metrics import calculate_classification_metrics


def test_perfect_predictions():
    """Verify metrics on 100% correct predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])

    res = calculate_classification_metrics(y_true, y_pred, y_prob)

    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["confusion_matrix"] == [[2, 0], [0, 2]]


def test_imperfect_predictions():
    """Verify metrics with mixed false positives and false negatives."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # 1 FP (index 1), 1 FN (index 2)

    res = calculate_classification_metrics(y_true, y_pred)

    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5
    assert res["macro_f1"] == 0.5
    assert res["confusion_matrix"] == [[1, 1], [1, 1]]


def test_edge_case_zero_division():
    """Verify handling when a class has zero positive predictions."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])

    res = calculate_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_roc_auc_1d_and_2d_probabilities():
    """Verify ROC-AUC accepts both 1D and 2D probability formats."""
    y_true = np.array([0, 0, 1, 1])
    y_prob_2d = np.array([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])
    y_prob_1d = np.array([0.2, 0.3, 0.8, 0.9])

    res_2d = calculate_classification_metrics(y_true, y_true, y_prob_2d)
    res_1d = calculate_classification_metrics(y_true, y_true, y_prob_1d)

    assert res_2d["roc_auc"] == 1.0
    assert res_1d["roc_auc"] == 1.0
