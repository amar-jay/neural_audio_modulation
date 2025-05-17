from sklearn.metrics import mean_squared_error
import numpy as np


def signal_to_noise_ratio(original, compressed):
    """Calculate the Signal-to-Noise Ratio (SNR) between the original and compressed signals."""
    if len(original) == 0 or len(compressed) == 0:
        raise ValueError("Input signals must not be empty.")

    noise = original - compressed
    snr = 10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    return snr


def mean_squared_error_metric(original, compressed):
    """Calculate the Mean Squared Error (MSE) between the original and compressed signals."""
    if len(original) == 0 or len(compressed) == 0:
        raise ValueError("Input signals must not be empty.")

    return mean_squared_error(original, compressed)
