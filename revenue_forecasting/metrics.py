import numpy as np
from typing import Union
from sklearn.metrics import mean_absolute_error, mean_squared_error


def bias(y_true: Union[list, np.array], y_pred: Union[list, np.array]) -> float:
    """
    Calculate the bias between true and predicted values. The bias is calculated
    as the mean of the differences between the predicted and true values. A positive
    bias indicates that predictions are higher than the true values, while a negative
    bias indicates lower predictions.

    Args:
        y_true (list, np.array): True values.
        y_pred (list, np.array): Predicted values.

    Returns:
        float: The bias between true and predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_pred - y_true)


def bias_percentage(y_true: Union[list, np.array], y_pred: Union[list, np.array]) -> float:
    """
    Calculate the bias between true and predicted values. The bias is calculated
    as the mean of the differences between the predicted and true values. A positive
    bias indicates that predictions are higher than the true values, while a negative
    bias indicates lower predictions.

    Args:
        y_true (list, np.array): True values.
        y_pred (list, np.array): Predicted values.

    Returns:
        float: The bias between true and predicted values.
    """
    return bias(y_true, y_pred) / np.sum(np.abs(y_true))


def mae(y_true: Union[list, np.array], y_pred: Union[list, np.array]) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.
    The MAE is the average of absolute differences between predicted and true
    values. Lower MAE values indicate better predictive performance.

    Args:
        y_true (list, np.array): True values.
        y_pred (list, np.array): Predicted values.

    Returns:
        float: The MAE between true and predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return mean_absolute_error(y_true, y_pred)


def mae_percentage(y_true: Union[list, np.array], y_pred: Union[list, np.array]) -> float:
    """
    Calculate the Mean Absolute Error (MAE) percentage between true and predicted values.

    Args:
        y_true (list, np.array): True values.
        y_pred (list, np.array): Predicted values.

    Returns:
        float: The MAE percentage between true and predicted values.
    """
    return mean_absolute_error(y_true, y_pred) / np.mean(y_true)
