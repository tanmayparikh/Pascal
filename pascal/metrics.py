from __future__ import annotations

import pandas as pd
import numpy as np

"""
                     Condition: A        Not A

  Test says “A”       True positive   |   False positive
                      ----------------------------------
  Test says “Not A”   False negative  |    True negative
"""

def f1_score(y_true: pd.Series|np.ndarray, y_pred: pd.Series|np.ndarray) -> pd.DataFrame:
    """Calculates the f1 score using the true and predicted labels

    Args:
        y_true (pd.Series | np.ndarray): True labels from the dataset
        y_pred (pd.Series | np.ndarray): Predicted labels from the model

    Returns:
        pd.DataFrame: Precision Recall and F1 score for each label
    """
    if type(y_true) == pd.Series:
        y_true = y_true.to_numpy()
        
    if type(y_pred) == pd.Series:
        y_pred = y_pred.to_numpy()
        
    if y_true.shape != y_pred.shape:
        raise ValueError("Y True/Pred must have the same shape")
    
    unique_labels = np.unique(y_true)
    
    out = pd.DataFrame(columns=["Label", "Precision", "Recall", "F1-Score"])
    
    for i, label in enumerate(unique_labels):
        true_pos = np.sum(np.logical_and(y_true == label, y_pred == label))
        false_pos = np.sum(np.logical_and(y_true != label, y_pred == label))
        false_neg = np.sum(np.logical_and(y_true == label, y_pred != label))


        if true_pos != 0 and (false_pos != 0 or false_neg != 0):
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
        
        out.loc[i] = [label, precision, recall, f1]
    
    out.set_index("Label", inplace=True)
    return out

def mean_abs_error(y_true: pd.Series|np.ndarray, y_pred: pd.Series|np.ndarray) -> float:
    """Calculate the mean absolute error

    Args:
        y_true (pd.Series | np.ndarray): Ground truth
        y_pred (pd.Series | np.ndarray): Predicted values

    Returns:
        float: Mean absolute error
    """
    if type(y_true) == pd.Series:
        y_true = y_true.to_numpy()
        
    if type(y_pred) == pd.Series:
        y_pred = y_pred.to_numpy()
        
    if y_true.shape != y_pred.shape:
        raise ValueError("Y True/Pred must have the same shape")
    
    return np.sum(np.abs(y_true - y_pred)) / y_true.shape[0]

def coef_determination(y_true: pd.Series|np.ndarray, y_pred: pd.Series|np.ndarray) -> float:
    """Calculate the R^2 coefficient or Coefficient of Determination

    Args:
        y_true (pd.Series | np.ndarray): Ground truth
        y_pred (pd.Series | np.ndarray): Predicted values

    Returns:
        float: R^2 coefficient
    """
    if type(y_true) == pd.Series:
        y_true = y_true.to_numpy()
        
    if type(y_pred) == pd.Series:
        y_pred = y_pred.to_numpy()
        
    if y_true.shape != y_pred.shape:
        raise ValueError("Y True/Pred must have the same shape")    
    
    residuals = y_true - y_pred
    mean = y_true.mean()
    
    # Calculate the residual sum of squares
    ss_res = np.sum(np.power(residuals, 2))
    
    # Calculate the total sum of squares
    ss_tot = np.sum(np.power(y_true - mean, 2))
    
    return 1 - (ss_res / ss_tot)

def pearson_correlation_coef(x:np.ndarray|pd.Series, y:np.ndarray|pd.Series):
    """Calculates the Pearson Coreelation Coefficient

    Args:
        x (np.ndarray | pd.Series): Input variable 1
        y (np.ndarray | pd.Series): Input variable 2
    """
    
    if type(x) == pd.Series:
        x = x.to_numpy()
        
    if type(y) == pd.Series:
        y = y.to_numpy()
    
    if x.shape != y.shape:
        raise ValueError("Input data must have the same shape")

    cov = np.cov(np.concatenate((x, y), axis=1), rowvar=False)
    return cov / (x.std() * y.std())