from typing import Tuple

import numpy as np
from scipy import stats
import torch


## Uncertainty calibration
# confidence calibration
def prep_reliability_diagram(true, preds, uncertainties, number_quantiles):
    """
    AUTHOR: JKH
    """
    true, preds, uncertainties = (
        np.array(true),
        np.array(preds),
        np.array(uncertainties),
    )

    # confidence intervals
    # four_sigma = 0.999936657516334
    perc = np.arange(0, 1+1 / number_quantiles, 1 / number_quantiles)
    count_arr = np.vstack(
        [
            np.abs(true - preds)
            <= stats.norm.interval(q, loc=np.zeros(len(preds)), scale=uncertainties)[1]
            for q in perc
        ]
    )
    count = np.mean(count_arr, axis=1)

    # ECE
    ECE = np.mean(np.abs(count - perc))

    # Sharpness
    Sharpness = np.std(uncertainties, ddof=1) / np.mean(uncertainties)

    return count, perc, ECE, Sharpness



def confidence_based_calibration(
    y_pred: np.array, uncertainties: np.array, y_ref_mean=0, quantiles=10
) -> Tuple[np.array, np.array]:
    """
    AUTHOR: RM
    Calculate confidence interval based calibration.
    See scalia et al. pg.2703 prior to Eq. (9), description of confidence based calibration.
    For each quantile compute the fraction of observations (+/- var) within the upper and lower-bound of the interval.
    One key underlying assumption prediction and variance describe a Gaussian distribution.

    Parameters:
        y_pred (np.ndarray): predicted labels
        uncertainties (np.ndarray): predictive variance
    Returns:
        Tuple[np.ndarray, np.ndarray]: fractions and quantiles over which fractions were computed.
    """
    assert len(y_pred) == len(uncertainties)
    N = len(y_pred)
    quantiles = np.arange(0, 1+1 / quantiles, 1 / quantiles)
    fractions = []
    for sigma_q in quantiles:
        upper_bound, lower_bound = y_ref_mean + sigma_q, y_ref_mean - sigma_q
        interval_count = np.sum(
            ((y_pred + uncertainties) <= upper_bound)
            & ((y_pred - uncertainties) >= lower_bound)
        )
        fractions.append(interval_count / N)
    return np.array(fractions), np.array(quantiles)


# def error_based_calibration(y_trues, y_pred, uncertainties):
#     raise NotImplementedError("Error Based Calibration is not yet implemented.")

def error_based_calibration(y_true, y_pred, var_pred, num_bins=10):
    """
    Author: Jacob KH
    Computes error-based calibration by binning predictions based on uncertainty and comparing 
    empirical error to predicted uncertainty in each bin.

    Parameters:
    y_true: Tensor of true values, shape [batch_size]
    y_pred: Tensor of predicted means from the model, shape [batch_size]
    sigma_pred: Tensor of predicted standard deviations (uncertainty), shape [batch_size]
    num_bins: Number of bins to use for grouping by uncertainty

    Returns:
    calibration_results: Dictionary with bins as keys and a tuple of (empirical error, average predicted uncertainty) as values.
    """
    # Calculate absolute error for each prediction
    y_true = torch.from_numpy(y_true) if not isinstance(y_true, torch.Tensor) else y_true
    y_pred = torch.from_numpy(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    var_pred = torch.from_numpy(var_pred) if not isinstance(var_pred, torch.Tensor) else var_pred

    errors = (y_true - y_pred)**2

    # Sort predictions by increasing uncertainty (sigma_pred)
    sorted_indices = torch.argsort(var_pred)
    sorted_errors = errors[sorted_indices]
    sorted_var = var_pred[sorted_indices]
    
    # Split into bins based on sorted uncertainties
    bin_size = len(sorted_var) // num_bins
    binned_errors = torch.split(sorted_errors, bin_size)
    binned_var = torch.split(sorted_var, bin_size)

    # Calculate average error and average uncertainty for each bin
    avg_empirical_error = [torch.mean(bin_error).item()**0.5 for bin_error in binned_errors]
    avg_predicted_uncertainty = [torch.mean(bin_var).item()**0.5 for bin_var in binned_var]

    return avg_empirical_error, avg_predicted_uncertainty


def expected_calibration_error(
    loss_fractions: np.ndarray, conf_interval_values: np.ndarray
) -> float:
    """
    Equation (10) ECE ratio of absolute difference between (loss_fractions in confidence interval) and confidence interval
    """
    assert len(loss_fractions) == len(conf_interval_values)
    return np.sum(np.abs(loss_fractions - conf_interval_values)) / len(loss_fractions)


def max_calibration_error(
    loss_fractions: np.ndarray, conf_interval_values: np.ndarray
) -> float:
    """
    Equation (10) MCE ratio of absolute difference  between loss-fraction in interval and interval
    """
    assert len(loss_fractions) == len(conf_interval_values)
    return np.max(np.abs(loss_fractions - conf_interval_values))


def expected_normalized_calibration_error(
    losses: np.ndarray, uncertainties: np.ndarray, n_quantiles=10
) -> float:
    """
    Eq. (11) Expected normalized calibration error
    """
    unc_quantiles = np.arange(0.1, 1+1 / n_quantiles, 1 / n_quantiles)
    m_vars = np.nan_to_num(
        np.array(
            [
                np.sqrt(np.mean(np.array([u for u in uncertainties if u <= k])))
                for k in unc_quantiles
            ]
        )
    )
    m_losses = np.nan_to_num(
        np.array(
            [
                np.mean(
                    np.array([l for l, unc in zip(losses, uncertainties) if unc <= k])
                )
                for k in unc_quantiles
            ]
        )
    )
    ence = np.sum(np.abs(np.array(m_vars) - np.array(m_losses))) / len(unc_quantiles)
    return ence
