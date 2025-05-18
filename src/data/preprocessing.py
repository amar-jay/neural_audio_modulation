from scipy.io import wavfile
import numpy as np
import librosa
import torch
from torchaudio import transforms as T


def normalize_audio(data):
    """Normalize audio data to the range [-1, 1]."""
    return data / np.max(np.abs(data))


def augment_audio(data, augmentation_type="noise", noise_factor=0.005):
    """Apply augmentation to audio data."""
    if augmentation_type == "noise":
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        return np.clip(augmented_data, -1, 1)
    return data


def trim_audio(data, max_length):
    """Trim audio data to a maximum length."""
    if len(data) > max_length:
        return data[:max_length]
    return data


def pad_audio(data, target_length):
    """Pad audio data to a target length."""
    if len(data) < target_length:
        padding = target_length - len(data)
        return np.pad(data, (0, padding), "constant")
    return data


def extract_features(data, sample_rate):
    """Extract features from audio data."""
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    return mfccs.T


def save_audio(file_path, data, sample_rate):
    """Save audio data to a file."""
    wavfile.write(file_path, sample_rate, data.astype(np.int16))


def clip_audio(data, sequence_length):
    """Clip audio data to a specific sequence length."""
    if data.size(-1) < sequence_length:
        # Pad with zeros if too short
        pad_length = sequence_length - data.size(-1)
        data = torch.nn.functional.pad(data, (0, pad_length))
        return data[..., :sequence_length]
    else:
        start = torch.randint(0, data.size(-1) - sequence_length + 1, (1,)).item()
        return data[..., start : start + sequence_length]

def transform_audio(**kwargs):
    """
    Args:
        - min_max (bool): Normalize audio data to [0, 1].
        - softmax (bool): Apply softmax to audio data.
        - sequence_length (int): Length of the audio sequence.
        - sample_rate (int): Target sample rate for resampling.
    """

    def func(data, original_rate):
        if kwargs["min_max"]:
            data = (data - data.min()) / (data.max() - data.min())
        if kwargs["softmax"]:
            data = torch.nn.functional.softmax(data, dim=0)
        data = resample_audio(data, original_rate, kwargs.get("sample_rate", 44100))
        if kwargs["sequence_length"]:
            data = clip_audio(data, kwargs.get("sequence_length", 220500))
        return data

    return func


def resample_audio(data, original_rate, target_rate):
    """Resample audio data to a target sample rate."""
    if original_rate != target_rate:
        data = T.Resample(orig_freq=original_rate, new_freq=target_rate)(data)
    return data
