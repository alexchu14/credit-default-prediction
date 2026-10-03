"""
Model factory: the four scikit-learn classifiers and their
hyperparameter search grids.
"""
 
 
from __future__ import annotations
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
 
 
 
 
def get_models_and_params(random_state: int = 42) -> dict:
    """
    Return the model + hyperparameter-grid configuration 
    (10-20 combinations per model, as required by the assignment).
    """
    
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=random_state),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"],
                "class_weight": [None, "balanced"],
            },
        },
        
        "Support Vector Machine": {
            "model": SVC(max_iter=3000, tol=0.1, random_state=random_state),
            "params": {
                "C": [0.01, 0.1, 1, 5],
                "kernel": ["rbf"],
                "gamma": ["scale", "auto"],
                "class_weight": [None, "balanced"],
            },
        },
        
        "Random Forest": {
            "model": RandomForestClassifier(random_state=random_state),
            "params": {
                "n_estimators": [10, 50, 100],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5],
            },
        },
        
        "Multi-Layer Perceptron": {
            "model": MLPClassifier(max_iter=1000, random_state=random_state),
            "params": {
                "hidden_layer_sizes": [(10,), (100,), (50, 50)],
                "alpha": [0.0001, 0.01],
                "learning_rate_init": [0.001, 0.01],
            },
        },
    }