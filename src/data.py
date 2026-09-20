"""
Data loading and splitting utilities for the credit default prediction
project.
 
This module is a structural extraction of the data-loading and
stratified-splitting cells from the baseline coursework notebook. 
It changes nothing about the computation.
"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
 



def load_raw_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns
    X : pd.DataFrame
        Feature matrix (23 columns, named x1..x23 by OpenML).
    y : pd.Series
        Binary target (1 = default, 0 = non-default).
    """
    data = fetch_openml(data_id=42477, as_frame=True)
    X = data.data
    y = data.target.astype(int)
    
    return X, y
 


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
):
    # Split into stratified train/validation/test sets (70/15/15)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
 

def print_class_report(y_split: pd.Series, split_name: str) -> None:
    
    counts = np.bincount(y_split.values)
    print(f"{split_name} class report:")
    print(f"Non-Default(0) : {counts[0]}")
    print(f"Default(1) : {counts[1]}")
    print(f"Total : {len(y_split)} \n")









