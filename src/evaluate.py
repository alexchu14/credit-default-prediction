"""
Structural extraction of the evaluation cells from the baseline
notebook. 
"""
 
from __future__ import annotations
 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
 
 
 
 
def evaluate_model(model, X_test_scaled, y_test):
    """
    Compute the full metric set for one model on the test set:
    Accuracy, Precision/Recall (class 1 = default), Macro-F1, AUROC,
    AUPRC.
    """
    y_pred = model.predict(X_test_scaled)
 
 
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.decision_function(X_test_scaled)
 
 
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision (Class 1)": round(precision_score(y_test, y_pred, pos_label=1), 3),
        "Recall (Class 1)": round(recall_score(y_test, y_pred, pos_label=1), 3),
        "Macro-F1": round(f1_score(y_test, y_pred, average="macro"), 3),
        "AUROC": round(roc_auc_score(y_test, y_prob), 3),
        "AUPRC": round(average_precision_score(y_test, y_prob), 3),
    }
    
    return metrics, y_prob
 
 

 
def build_metrics_table(
    best_models: dict, X_test_scaled, y_test, model_col_name: str = "Model"
):
    """
    Loop over all tuned models, evaluate each on the test set, and
    return the main comparison table  plus a dict of per-model probability
    scores (needed for the overlaid ROC / PR plots).
    """
    
    rows = []
    model_proba = {}
 
 
    for name, model in best_models.items():   
        metrics, y_prob = evaluate_model(model, X_test_scaled, y_test)
        rows.append({model_col_name: name, **metrics})
        model_proba[name] = y_prob
 
 
    return pd.DataFrame(rows), model_proba