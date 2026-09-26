"""Visualization functions — EDA figures and evaluation figures."""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)



def plot_class_distribution(y_train):
    categories=[0,1]
    Train_count = [np.sum(y_train == 0), np.sum(y_train == 1)]
    bars = plt.bar(categories, Train_count )
    
    plt.title("Target distribution (train)")
    plt.xticks(categories, [' 0 (Non-default)', ' 1 (Default)'])
    plt.ylabel("Count")
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),f'{bar.get_height()}\n({round(bar.get_height()/len(y_train),2)}%)', ha='center', va='bottom')


     
def plot_correlation_heatmap(X_train): 
    corr_matrix = X_train.corr()

    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    plt.title('Correlation heatmap (train)')
    plt.show()
        


def plot_feature_histograms(X_train):
    X_train.hist(bins=30,figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    
    
    
def plot_confusion_matrix(model, X_test_scaled, y_test, title: str = "Confusion Matrix"):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()
 
 
 
 
def plot_roc_curves(model_proba: dict, y_test):
    
    for name, y_prob in model_proba.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
        auroc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUROC = {auroc:.3f})")
 
 
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overlaid ROC Curves on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()
 
 
 
def plot_pr_curves(model_proba: dict, y_test):

    for name, y_prob in model_proba.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=1)
        auprc = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f"{name} (AUPRC = {auprc:.3f})")
 
 
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overlaid Precision-Recall Curves on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()