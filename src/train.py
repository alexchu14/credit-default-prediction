"""
Training / hyperparameter-tuning routine.

Reproduces the manual grid-search loop from the baseline notebook.
"""

from __future__ import annotations

import copy

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from src.data import load_raw_data, print_class_report, stratified_split
from src.features import build_preprocessor
from src.models import get_models_and_params


def tune_models(
    models_and_params: dict,
    X_train_scaled,
    y_train,
    X_val_scaled,
    y_val,
    verbose_prefix: str = "Training",
):
    """
    Manual grid search over each model's hyperparameter grid, selecting
    the configuration with the best macro-F1 score on the validation
    set.
    """
    
    best_models = {}
    table_hyperparam = []


    for model_name, config in models_and_params.items():
        print(f"{verbose_prefix} {model_name}...")

        best_score = -1
        best_params = None
        best_model = None


        for params in ParameterGrid(config["params"]):
            model = clone(config["model"])
            model.set_params(**params)
            model.fit(X_train_scaled, y_train)

            y_val_pred = model.predict(X_val_scaled)
            current_score = f1_score(y_val, y_val_pred, average="macro")


            if current_score > best_score:
                best_score = current_score
                best_params = params
                best_model = copy.deepcopy(model)

        best_model.set_params(**best_params)
        best_model.fit(X_train_scaled, y_train)
        best_models[model_name] = best_model


        table_hyperparam.append(
            {
                "Model": model_name,
                "Selected Hyperparameters": str(best_params),
                " Score (F1)": round(best_score, 4),
            }
        )


    return best_models, table_hyperparam



def run_baseline_pipeline(random_state: int = 42) -> dict:
    """
    End-to-end reproduction of the baseline notebook's
    data -> split -> preprocess -> tune pipeline .
    """
    
    X, y = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, random_state=random_state
    )

    print_class_report(y_train, "Train")
    print_class_report(y_val, "Validation")
    print_class_report(y_test, "Test")


    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)


    models_and_params = get_models_and_params(random_state=random_state)
    best_models, table_hyperparam = tune_models(
        models_and_params,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        verbose_prefix="Training",
    )

    return {
        "preprocessor": preprocessor,
        "models_and_params": models_and_params,
        "best_models": best_models,
        "table_hyperparam": table_hyperparam,
    }



def run_smote_pipeline(baseline_results: dict, random_state: int = 42):
    """
    Resample the (already scaled) training set with SMOTE, 
    retune the same 4 models, and return the
    SMOTE-variant models. Reuses the scaled data / preprocessor 
    already computed by run_baseline_pipeline (no leakage, no recomputation).
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(
        baseline_results["X_train_scaled"], baseline_results["y_train"]
    )


    best_models_smote, table_hyperparam_smote = tune_models(
        baseline_results["models_and_params"],
        X_train_smote,
        y_train_smote,
        baseline_results["X_val_scaled"],
        baseline_results["y_val"],
        verbose_prefix="Tuning (SMOTE):",
    )

    return best_models_smote, table_hyperparam_smote



if __name__ == "__main__":
    from src.evaluate import build_metrics_table

    baseline = run_baseline_pipeline()
    print(baseline["table_hyperparam"])

    metrics_table, model_proba = build_metrics_table(
        baseline["best_models"], baseline["X_test_scaled"], baseline["y_test"]
    )
    print(metrics_table)

    best_models_smote, table_hyperparam_smote = run_smote_pipeline(baseline)
    metrics_table_smote, _ = build_metrics_table(
        best_models_smote,
        baseline["X_test_scaled"],
        baseline["y_test"],
        model_col_name="Model (with SMOTE)",
    )
    print(metrics_table_smote)