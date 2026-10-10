"""
Cost-sensitive threshold analysis for the credit default classifier.


A classifier tuned to maximize accuracy or F1 is not necessarily the
right operating point for deployment. This module searches for the decision threshold that
minimizes expected cost on the validation set, then reports that
threshold's performance on the test set, alongside the default 0.5
threshold for comparison.

This is new functionality added after the initial extraction from the
coursework notebook.
"""


from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score



def expected_cost(
    y_true, y_prob, threshold: float, cost_fn: float = 5.0, cost_fp: float = 1.0
) -> float:
    """
    Total cost of operating at a given decision threshold.

    cost_fn : cost of a false negative (a missed defaulter). Defaults
        to 5x the cost of a false positive. This ratio is an
        assumption, not a measured value.
        
    cost_fp : cost of a false positive (a wrongly flagged good
        customer).
    """
    
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return cost_fn * fn + cost_fp * fp



def find_optimal_threshold(
    y_val,
    y_val_prob,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    thresholds: np.ndarray | None = None,
):
    """
    Scan thresholds on the validation set and return the one that
    minimizes expected cost.

    Returns
    -------
    best_threshold : float
    
    cost_curve : pd.DataFrame
        One row per threshold tested, with its expected cost. Used
        for plotting  and for sanity-checking.
    """
    
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    costs = [
        expected_cost(y_val, y_val_prob, t, cost_fn=cost_fn, cost_fp=cost_fp)
        for t in thresholds
    ]
    
    cost_curve = pd.DataFrame({"threshold": thresholds, "expected_cost": costs})
    best_threshold = float(cost_curve.loc[cost_curve["expected_cost"].idxmin(), "threshold"])
    
    
    return best_threshold, cost_curve



def evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """
    Precision / recall / Macro-F1 / confusion-matrix breakdown at a
    given decision threshold, rather than scikit-learn's implicit
    default of 0.5.
    """
    
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()


    return {
        "threshold": round(threshold, 3),
        "precision_class1": round(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0), 3
        ),
        "recall_class1": round(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0), 3
        ),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }



def compare_default_vs_optimal_threshold(
    y_val,
    y_val_prob,
    y_test,
    y_test_prob,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> pd.DataFrame:
    """
    Find the cost-optimal threshold on the validation set, 
    then report test-set performance at both the
    default (0.5) and the optimal threshold, side by side.
    """
    
    best_threshold, _ = find_optimal_threshold(
        y_val, y_val_prob, cost_fn=cost_fn, cost_fp=cost_fp
    )

    default_result = evaluate_at_threshold(y_test, y_test_prob, 0.5)
    optimal_result = evaluate_at_threshold(y_test, y_test_prob, best_threshold)


    return pd.DataFrame(
        [
            {"Threshold type": "Default (0.5)", **default_result},
            {
                "Threshold type": f"Cost-optimal (FN:FP = {cost_fn:g}:{cost_fp:g})",
                **optimal_result,
            },
        ]
    )
    
    

def cost_sensitivity_analysis(
    y_val,
    y_val_prob,
    y_test,
    y_test_prob,
    cost_ratios: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0),
) -> pd.DataFrame:
    """
    Re-run the threshold search for several false-negative/false-positive
    cost ratios (cost_fp is fixed at 1) to show how much the
    recommended threshold depends on the assumed ratio.
 
    The threshold is always selected on the validation set and
    evaluated on the test set. The expected cost reported is the
    test-set cost at the selected threshold, computed with the same
    ratio used to select it.
 
    Returns
    -------
    pd.DataFrame
        One row per ratio: selected threshold, test precision/recall,
        FN/FP counts, and test expected cost.
    """
    
    rows = []
    
    for ratio in cost_ratios:
        threshold, _ = find_optimal_threshold(
            y_val, y_val_prob, cost_fn=ratio, cost_fp=1.0
        )
        result = evaluate_at_threshold(y_test, y_test_prob, threshold)
        
        rows.append(
            {
                "cost_fn : cost_fp": f"{ratio:g} : 1",
                "optimal_threshold": result["threshold"],
                "precision_class1": result["precision_class1"],
                "recall_class1": result["recall_class1"],
                "false_negatives": result["false_negatives"],
                "false_positives": result["false_positives"],
                "test_expected_cost": ratio * result["false_negatives"]
                + result["false_positives"],
            }
        )
        
    return pd.DataFrame(rows)   