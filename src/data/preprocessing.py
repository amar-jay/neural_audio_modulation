from scipy.io import wavfile
import numpy as np
import librosa
import torch


def load_audio(file_path):
    """Load an audio file."""
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data


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


def transform_audio(min_max=False, softmax=True):
    def func(data):
        if min_max:
            data = (data - data.min()) / (data.max() - data.min())
        if softmax:
            data = torch.nn.functional.softmax(data, dim=0)
        return data

    return func
