import wandb
import torch
from sklearn.metrics import mean_squared_error
import numpy as np
from pypesq import pesq

from pystoi import stoi
from ..data.preprocessing import transform_audio


def signal_to_noise_ratio(original, compressed):
    """Calculate the Signal-to-Noise Ratio (SNR) between the original and compressed signals."""
    if len(original) == 0 or len(compressed) == 0:
        raise ValueError("Input signals must not be empty.")

    noise = original - compressed
    snr = 10 * torch.log10(torch.sum(original**2) / torch.sum(noise**2))
    return snr


def mean_squared_error_metric(original, compressed):
    """Calculate the Mean Squared Error (MSE) between the original and compressed signals."""
    if len(original) == 0 or len(compressed) == 0:
        raise ValueError("Input signals must not be empty.")

    # Ensure tensors are flat for scikit-learn
    if isinstance(original, torch.Tensor):
        original_flat = original.flatten().detach().numpy()
    else:
        original_flat = original.flatten()

    if isinstance(compressed, torch.Tensor):
        compressed_flat = compressed.flatten().detach().numpy()
    else:
        compressed_flat = compressed.flatten()

    return mean_squared_error(original_flat, compressed_flat)


def peak_signal_to_noise_ratio(original, compressed):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original and compressed signals."""
    if len(original) == 0 or len(compressed) == 0:
        raise ValueError("Input signals must not be empty.")

    # Ensure tensors are flat for scikit-learn
    if isinstance(original, torch.Tensor):
        original_flat = original.flatten().detach().numpy()
    else:
        original_flat = original.flatten()

    if isinstance(compressed, torch.Tensor):
        compressed_flat = compressed.flatten().detach().numpy()
    else:
        compressed_flat = compressed.flatten()

    mse = mean_squared_error(original_flat, compressed_flat)
    if mse == 0:
        return float("inf")  # No noise, infinite PSNR

    max_pixel_value = original.max()
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def perceptual_difference(original, compressed):
    """Calculate the Perceptual Difference (PD) between the original and compressed signals."""
    if len(original) == 0 or len(compressed) == 0:
        raise ValueError("Input signals must not be empty.")

    # Placeholder for perceptual difference calculation
    # In practice, this would involve more complex audio processing
    return np.abs(original - compressed).mean()


def calculate_metrics(original, compressed, sample_rate, config, embedding_loss, perplexity, loss, prefix=""):
    """Calculate various metrics between the original and compressed signals."""
    compressed = transform_audio(
        config["data"]["transform"]["min_max"], config["data"]["transform"]["softmax"]
    )(compressed)
    snr = signal_to_noise_ratio(original, compressed)
    mse = mean_squared_error_metric(original, compressed)
    psnr = peak_signal_to_noise_ratio(original, compressed)
    pd = perceptual_difference(original, compressed)
    # PESQ (only 8kHz or 16kHz supported)
    if original.shape[0] == 1:
        original = original.squeeze()
    if compressed.shape[0] == 1:
        compressed = compressed.squeeze()
    # print(f"Original: {original.shape}, Compressed: {compressed.shape}")
    # pesq_score = pesq(original, compressed,sample_rate)

    # STOI
    stoi_score = stoi(original, compressed, sample_rate, extended=False)

    return {
        f"{prefix}SNR": snr,
        f"{prefix}MSE": mse,
        f"{prefix}PSNR": psnr,
        f"{prefix}PD": pd,
        # f"{prefix}PESQ": pesq_score,
        f"{prefix}STOI": stoi_score,
        f"{prefix}embedding_loss": embedding_loss,
        f"{prefix}perplexity": perplexity,
        f"{prefix}loss": loss,
    }
